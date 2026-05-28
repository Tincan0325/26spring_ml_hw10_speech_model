"""Model A inference entry point.

Usage:
    from model_a.run import run_model_a
    result = run_model_a("input.wav")
"""
import contextlib
import logging
import os
import warnings

# Silence framework chatter that names the underlying models — must run BEFORE
# transformers / faster-whisper are imported.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MODELSCOPE_LOG_LEVEL", "40")  # ERROR
warnings.filterwarnings("ignore")
for _name in (
    "transformers", "faster_whisper", "huggingface_hub", "modelscope",
    "accelerate", "diffusers", "cosyvoice", "matcha", "wetext",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _silenced():
    """Suppress stdout/stderr at the OS level — covers C-extension prints too."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


import torch

_MODEL_A_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level cache — loaded once, reused for every call in the loop.
_whisper_model = None
_llama_model = None
_llama_tokenizer = None


def _stage1(audio_path: str) -> str:
    """First stage: speech -> text."""
    global _whisper_model
    with _silenced():
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8")
        segments, _ = _whisper_model.transcribe(audio_path, language="en", beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
    return transcript


def _stage2(transcript: str) -> str:
    """Second stage: text -> text response."""
    global _llama_model, _llama_tokenizer
    with _silenced():
        if _llama_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import transformers
            transformers.logging.set_verbosity_error()
            model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
            _llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _llama_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        messages = [
            {"role": "system", "content": "You are an assistant good to classify the gender in male and female. The input audio has a question and related audio data. Focus on the audio data to answer the question."},
            {"role": "user", "content": f"Please classify the gender in the following audio clip.\n{transcript}"},
        ]
        chat_input = _llama_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        # Newer transformers may return a BatchEncoding instead of a raw tensor
        if hasattr(chat_input, "input_ids"):
            input_ids = chat_input["input_ids"].to(_llama_model.device)
            generate_kwargs = {k: v.to(_llama_model.device) for k, v in chat_input.items()}
        else:
            input_ids = chat_input.to(_llama_model.device)
            generate_kwargs = {"input_ids": input_ids}

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        with torch.no_grad():
            output = _llama_model.generate(
                **generate_kwargs,
                max_new_tokens=80,
                do_sample=False,
                temperature=1.0,
                pad_token_id=_llama_tokenizer.eos_token_id,
            )
        response = _llama_tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
    return response


def run_model_a(audio_path: str) -> dict:
    """Run Model A on an audio file.

    Args:
        audio_path: Path to the input WAV/MP3 file.

    Returns:
        Dict with keys: transcript, response.
    """
    transcript = _stage1(audio_path)
    response = _stage2(transcript)

    return {
        "transcript": transcript,
        "response": response,
    }
