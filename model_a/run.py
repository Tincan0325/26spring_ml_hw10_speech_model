"""Model A inference entry point.

Usage:
    from model_a.run import run_model_a
    result = run_model_a("input.wav", "output.wav")
"""
import contextlib
import gc
import logging
import os
import sys
import warnings
from typing import Optional

# Silence framework chatter that names the underlying models — must run BEFORE
# transformers / faster-whisper / modelscope / cosyvoice are imported.
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


def _stage1(audio_path: str) -> str:
    """First stage: speech -> text."""
    with _silenced():
        from faster_whisper import WhisperModel

        model = WhisperModel("large-v3", device="cuda", compute_type="int8")
        segments, _ = model.transcribe(audio_path, language="en", beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()

        del model
        gc.collect()
        torch.cuda.empty_cache()
    return transcript


def _stage2(transcript: str, hf_token: Optional[str]) -> str:
    """Second stage: text -> text response."""
    with _silenced():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        transformers.logging.set_verbosity_error()

        model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        kwargs = {"token": hf_token} if hf_token else {}

        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

        messages = [
            {"role": "user", "content": transcript},
        ]
        chat_input = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        # Newer transformers may return a BatchEncoding instead of a raw tensor
        if hasattr(chat_input, "input_ids"):
            input_ids = chat_input["input_ids"].to(model.device)
            generate_kwargs = {k: v.to(model.device) for k, v in chat_input.items()}
        else:
            input_ids = chat_input.to(model.device)
            generate_kwargs = {"input_ids": input_ids}

        with torch.no_grad():
            output = model.generate(
                **generate_kwargs,
                max_new_tokens=80,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return response


def _stage3(text: str, output_path: str, cosyvoice_dir: str) -> None:
    """Third stage: text -> speech."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 requires a CUDA GPU.")

    sys.path.insert(0, cosyvoice_dir)
    sys.path.insert(0, os.path.join(cosyvoice_dir, "third_party/Matcha-TTS"))
    with _silenced():
        from cosyvoice.cli.cosyvoice import CosyVoice
        import torchaudio

        pretrained = os.path.join(cosyvoice_dir, "pretrained_models/CosyVoice-300M-SFT")
        cosy = CosyVoice(pretrained)

        chunks = []
        for chunk in cosy.inference_sft(text, "英文女"):
            chunks.append(chunk["tts_speech"])
    audio = torch.cat(chunks, dim=1)
    torchaudio.save(output_path, audio, 22050)

    del cosy
    gc.collect()
    torch.cuda.empty_cache()


def run_model_a_batch(
    audio_paths,
    output_dir: str,
    hf_token: Optional[str] = None,
    cosyvoice_dir: str = None,
    on_clip_done=None,
) -> dict:
    """Run Model A on many clips, loading each stage's weights only once.

    Args:
        audio_paths: iterable of input WAV/MP3 paths.
        output_dir: directory to write synthesised WAVs into (named ``<stem>.wav``).
        hf_token: optional Hugging Face token.
        cosyvoice_dir: support repo path (defaults to ``model_a/CosyVoice``).
        on_clip_done: optional callable ``(idx, audio_path, transcript, response, out_path)``
            invoked immediately after each output WAV is written. Used by the
            notebook to trigger per-clip download.

    Returns:
        ``{audio_path: {"transcript": ..., "response": ..., "output_audio": ...}}``
    """
    from pathlib import Path

    audio_paths = list(audio_paths)
    if cosyvoice_dir is None:
        cosyvoice_dir = os.path.join(_MODEL_A_DIR, "CosyVoice")
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Stage 1: transcribe all clips with one model load.
    with _silenced():
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8")
        transcripts = []
        for path in audio_paths:
            segments, _ = whisper_model.transcribe(path, language="en", beam_size=5)
            transcripts.append(" ".join(seg.text.strip() for seg in segments).strip())
        del whisper_model
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 2: generate responses with one LLM load.
    with _silenced():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        transformers.logging.set_verbosity_error()

        model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        kwargs = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        responses = []
        for transcript in transcripts:
            chat_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": transcript}],
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if hasattr(chat_input, "input_ids"):
                input_ids = chat_input["input_ids"].to(llm.device)
                generate_kwargs = {k: v.to(llm.device) for k, v in chat_input.items()}
            else:
                input_ids = chat_input.to(llm.device)
                generate_kwargs = {"input_ids": input_ids}
            with torch.no_grad():
                output = llm.generate(
                    **generate_kwargs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            responses.append(
                tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
            )
        del llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 3: synthesise speech with one CosyVoice load; callback per clip.
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 requires a CUDA GPU.")
    sys.path.insert(0, cosyvoice_dir)
    sys.path.insert(0, os.path.join(cosyvoice_dir, "third_party/Matcha-TTS"))
    with _silenced():
        from cosyvoice.cli.cosyvoice import CosyVoice
        import torchaudio

        pretrained = os.path.join(cosyvoice_dir, "pretrained_models/CosyVoice-300M-SFT")
        cosy = CosyVoice(pretrained)

    for idx, (path, transcript, response) in enumerate(zip(audio_paths, transcripts, responses)):
        stem = Path(path).stem
        out_path = os.path.join(output_dir, f"{stem}.wav")
        with _silenced():
            chunks = [chunk["tts_speech"] for chunk in cosy.inference_sft(response, "英文女")]
            audio = torch.cat(chunks, dim=1)
            torchaudio.save(out_path, audio, 22050)
        results[path] = {
            "transcript": transcript,
            "response": response,
            "output_audio": out_path,
        }
        if on_clip_done is not None:
            on_clip_done(idx, path, transcript, response, out_path)

    with _silenced():
        del cosy
        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_model_a(
    audio_path: str,
    output_path: str,
    hf_token: Optional[str] = None,
    cosyvoice_dir: str = os.path.join(_MODEL_A_DIR, "CosyVoice"),
) -> dict:
    """Run Model A on a single audio file.

    Args:
        audio_path: Path to the input WAV/MP3 file.
        output_path: Path where the model's spoken response will be saved.
        hf_token: Optional Hugging Face token (not required for default models).
        cosyvoice_dir: Path to the cloned CosyVoice repository.

    Returns:
        Dict with keys: transcript, response, output_audio.
    """
    transcript = _stage1(audio_path)
    response = _stage2(transcript, hf_token)
    _stage3(response, output_path, cosyvoice_dir)

    return {
        "transcript": transcript,
        "response": response,
        "output_audio": output_path,
    }
