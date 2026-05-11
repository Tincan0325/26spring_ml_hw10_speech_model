"""Model B inference entry point.

Usage:
    from model_b.run import run_model_b
    result = run_model_b("input.wav", "output.wav")
"""
import gc
import json
import logging
import os
import sys
import tempfile
import warnings
from typing import Optional

import torch

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def _patch_quantized_to():
    """Allow accelerate's dispatch_model to call .to(device) on bnb-quantized models.

    transformers raises ValueError when .to() is called on a 4-bit/8-bit model, but
    accelerate's dispatch_model unconditionally calls it as part of from_pretrained.
    We make it a no-op for quantized models; quantized params are already on GPU,
    and non-persistent buffers are moved manually in _run_inference below.
    """
    import transformers.modeling_utils as _mu

    if getattr(_mu.PreTrainedModel.to, "_patched_for_quantized", False):
        return
    _orig_to = _mu.PreTrainedModel.to

    def _patched_to(self, *args, **kwargs):
        if getattr(self, "is_quantized", False) or getattr(self, "is_loaded_in_4bit", False) or getattr(self, "is_loaded_in_8bit", False):
            return self
        return _orig_to(self, *args, **kwargs)

    _patched_to._patched_for_quantized = True
    _mu.PreTrainedModel.to = _patched_to


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
    _patch_quantized_to()

    import whisper
    from omni_speech.model.builder import load_pretrained_model
    from omni_speech.conversation import conv_templates
    from omni_speech.datasets.preprocess import tokenizer_speech_token
    from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
    from fairseq import utils as fs_utils
    import soundfile as sf

    model_kwargs = {"token": hf_token} if hf_token else {}

    # 4-bit load to fit Llama-3.1-8B-Omni + speech encoder + projector + vocoder on a T4 (16 GB).
    tokenizer, model, _ = load_pretrained_model(
        model_path="ICTNLP/Llama-3.1-8B-Omni",
        model_base=None,
        is_lora=False,
        s2s=True,
        load_4bit=True,
        device="cuda",
        **model_kwargs,
    )
    # accelerate's dispatch_model places quantized params on GPU but leaves
    # non-persistent buffers (e.g. LlamaRotaryEmbedding.inv_freq) on CPU,
    # which then fails at the first forward. Move them manually.
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "cuda":
            parts = name.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod.register_buffer(parts[-1], buf.cuda(), persistent=False)

    speech = whisper.load_audio(audio_path)
    speech = whisper.pad_or_trim(speech)
    mel = whisper.log_mel_spectrogram(speech, n_mels=128)
    speech_tensor = mel.permute(1, 0).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    speech_length = torch.LongTensor([speech_tensor.shape[1]]).to(device="cuda")

    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], "<speech>")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to("cuda")

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
        audio_path: Path to the input WAV/MP3 file. Expected to already contain
            the spoken instruction prefix (as produced for sample_audio_prompted).
        output_path: Path where the model's spoken response will be saved.
        hf_token: Optional Hugging Face token.
        framework_dir: Path to the cloned support repository.
        instruction_wav: If provided, prepend this clip to ``audio_path`` before
            inference. Leave as None when the input already has the instruction
            spoken at the start.
        vocoder_ckpt: Path to the vocoder checkpoint.
        vocoder_cfg: Path to the vocoder config JSON.

    Returns:
        Dict with keys: response, output_audio.
    """
    if instruction_wav is None:
        return _run_inference(
            audio_path, output_path, hf_token, framework_dir, vocoder_ckpt, vocoder_cfg
        )
    combined = _concat_audio(instruction_wav, audio_path)
    try:
        return _run_inference(
            combined, output_path, hf_token, framework_dir, vocoder_ckpt, vocoder_cfg
        )
    finally:
        if os.path.exists(combined):
            os.unlink(combined)
