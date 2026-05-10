# TA Notes (Do Not Distribute to Students)

## What each model is

- **Model A** = Cascade pipeline
  - ASR: `faster-whisper large-v3` (int8)
  - LLM: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (pre-quantized 4-bit, **not gated**)
  - TTS: `FunAudioLLM/CosyVoice-300M`

- **Model B** = End-to-end speech LM
  - `ICTNLP/Llama-3.1-8B-Omni` (Whisper encoder + Llama-3.1-8B + NAR speech decoder + HiFi-GAN)

## Pedagogical point

For a speaker-gender task, Model A has to round-trip through text. Whisper throws away
all acoustic features (pitch, timbre), so Llama can only guess from word choice. Model B
operates on the waveform directly, so it has access to the speaker's voice.

## Generating `model_b/instruction.wav` (one-time, by TA)

Model B is a speech-instruction model — it expects the user's question to be in the audio
itself. Since we want students to ask about a third-party recording, we prepend a fixed
spoken instruction to every clip. Generate `model_b/instruction.wav` once (with CosyVoice,
so it matches Model A's voice) and commit it.

```python
# Run on Colab T4 once after running model_a/setup.sh
import sys, torch, torchaudio
sys.path.insert(0, "/content/CosyVoice")
sys.path.insert(0, "/content/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice

cosy = CosyVoice("/content/CosyVoice/pretrained_models/CosyVoice-300M")
text = "Listen to the following speaker and tell me whether the voice sounds male or female."
chunks = [c["tts_speech"] for c in cosy.inference_sft(text, "英文女")]
audio = torch.cat(chunks, dim=1)
# Resample to 16 kHz mono for LLaMA-Omni
audio_16k = torchaudio.functional.resample(audio, 22050, 16000)
torchaudio.save("model_b/instruction.wav", audio_16k, 16000)
```

## Hugging Face access checklist

| Asset | Gated? | Notes |
|-------|--------|-------|
| `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | No | |
| `Systran/faster-whisper-large-v3` | No | |
| `FunAudioLLM/CosyVoice-300M` | No | |
| `ICTNLP/Llama-3.1-8B-Omni` | **Verify before class** | Try `huggingface-cli download` without a token in a fresh Colab |
| HiFi-GAN vocoder | No | Hosted on dl.fbaipublicfiles.com |

If LLaMA-Omni download fails without a token, update README to instruct students to
create a free HF token and pass it via `hf_token=`.

## VRAM check

- Model A: ~12 GB peak (fits T4)
- Model B: ~13 GB peak (fits T4 with `bfloat16`)
- Both `run.py` files run cleanup (`del`, `gc.collect()`, `torch.cuda.empty_cache()`)
  between sub-stages so a single notebook can run both back-to-back.

## Verification before class

1. Fresh Colab T4. `git clone` the repo.
2. Run `model_a/setup.sh` then `run_model_a("sample_audio/sample.wav", "out_a.wav")`.
3. Restart runtime (or rely on the cleanup logic) and run `model_b/setup.sh` then
   `run_model_b("sample_audio/sample.wav", "out_b.wav")`.
4. Confirm both produce playable WAV files and stay under 15 GB VRAM.
5. Confirm the cascade (Model A) refuses or hedges on the gender question, while
   the E2E (Model B) attempts an answer based on the voice.
