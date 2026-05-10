"""Model B inference entry point.

Usage:
    from model_b.run import run_model_b
    result = run_model_b("input.wav", "output.wav")
"""
import gc
import json
import os
import sys
import tempfile
from typing import Optional

import torch


# Pre-recorded instruction prepended to every input clip so the model knows
# what the user is asking about the speaker in the recording.
DEFAULT_INSTRUCTION_WAV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "instruction.wav"
)


def _concat_audio(instruction_wav: str, audio_path: str) -> str:
    """Prepend the instruction clip to the user-provided clip and write to a tmp file."""
    from pydub import AudioSegment

    instruction = AudioSegment.from_file(instruction_wav).set_frame_rate(16000).set_channels(1)
    user_clip = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
    silence = AudioSegment.silent(duration=300, frame_rate=16000)
    combined = instruction + silence + user_clip

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    combined.export(tmp.name, format="wav")
    return tmp.name


def _ctc_postprocess(tokens: torch.Tensor, blank: int) -> str:
    toks = tokens.squeeze(0).tolist()
    deduped = [v for i, v in enumerate(toks) if i == 0 or v != toks[i - 1]]
    hyp = [v for v in deduped if v != blank]
    return " ".join(map(str, hyp))


def _run_inference(
    audio_path: str,
    output_path: str,
    hf_token: Optional[str],
    framework_dir: str,
    vocoder_ckpt: str,
    vocoder_cfg: str,
) -> dict:
    sys.path.insert(0, framework_dir)

    import whisper
    from omni_speech.model.builder import load_pretrained_model
    from omni_speech.conversation import conv_templates
    from omni_speech.datasets.preprocess import tokenizer_speech_token
    from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
    from fairseq import utils as fs_utils
    import soundfile as sf

    model_kwargs = {"token": hf_token} if hf_token else {}

    tokenizer, model, _ = load_pretrained_model(
        model_path="ICTNLP/Llama-3.1-8B-Omni",
        model_base=None,
        is_lora=False,
        s2s=True,
        device="cuda",
        **model_kwargs,
    )

    speech = whisper.load_audio(audio_path)
    speech = whisper.pad_or_trim(speech)
    mel = whisper.log_mel_spectrogram(speech, n_mels=128)
    speech_tensor = mel.permute(1, 0).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    speech_length = torch.LongTensor([speech_tensor.shape[1]]).to(device="cuda")

    conv = conv_templates["llama_3"].copy()
    conv.append_message(
        conv.roles[0],
        "<speech>\nPlease directly answer the questions in the user's speech.",
    )
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            speech=speech_tensor,
            speech_lengths=speech_length,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
            pad_token_id=128004,
            streaming_unit_gen=False,
        )
    output_ids, output_units = outputs

    response_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    units_str = _ctc_postprocess(output_units, blank=model.config.unit_vocab_size)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    with open(vocoder_cfg) as f:
        cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(vocoder_ckpt, cfg).cuda()
    codes = list(map(int, units_str.split()))
    x = fs_utils.move_to_cuda({"code": torch.LongTensor(codes).view(1, -1)})
    wav = vocoder(x, dur_prediction=True)
    sf.write(output_path, wav.detach().cpu().numpy(), 16000)

    del vocoder
    gc.collect()
    torch.cuda.empty_cache()

    return {"response": response_text, "output_audio": output_path}


def run_model_b(
    audio_path: str,
    output_path: str,
    hf_token: Optional[str] = None,
    framework_dir: str = "/content/LLaMA-Omni",
    instruction_wav: Optional[str] = None,
    vocoder_ckpt: str = "/content/vocoder/g_00500000",
    vocoder_cfg: str = "/content/vocoder/config.json",
) -> dict:
    """Run Model B on an audio file.

    Args:
        audio_path: Path to the input WAV/MP3 file.
        output_path: Path where the model's spoken response will be saved.
        hf_token: Optional Hugging Face token.
        framework_dir: Path to the cloned support repository.
        instruction_wav: Override the default instruction clip if needed.
        vocoder_ckpt: Path to the vocoder checkpoint.
        vocoder_cfg: Path to the vocoder config JSON.

    Returns:
        Dict with keys: response, output_audio.
    """
    instruction = instruction_wav or DEFAULT_INSTRUCTION_WAV
    combined = _concat_audio(instruction, audio_path)
    try:
        return _run_inference(
            combined, output_path, hf_token, framework_dir, vocoder_ckpt, vocoder_cfg
        )
    finally:
        if os.path.exists(combined):
            os.unlink(combined)
