# Speech Model Comparison — HW10

Inference code for two speech models, **Model A** and **Model B**.

This repository is meant to be cloned and used from a Colab notebook provided by your TA.
Do **not** modify the code in `model_a/` or `model_b/` — your task is to compare what each
model produces, not to change them.

## Folder layout

```
model_a/                  Inference code for Model A
model_b/                  Inference code for Model B
sample_audio/             Raw LibriSpeech clips (16 kHz mono FLAC)
sample_audio_prompted/    Same clips with spoken prompt prepended
```

## Sample audio clips

Ten FLAC files are provided in `sample_audio/`, sourced from the
[LibriSpeech](https://www.openslr.org/12) **train-clean-100** subset
(CC BY 4.0 / LibriVox readers).  Each file is the first utterance of that
speaker (`<speaker_id>-<chapter_id>-0000.flac`, 16 kHz mono).

| File | Speaker ID | Gender | LibriVox reader name |
|------|-----------|--------|----------------------|
| `307-127535-0000.flac`  | 307  | M | Randy Phillips       |
| `374-180298-0000.flac`  | 374  | M | kumarei              |
| `1743-142913-0000.flac` | 1743 | M | Bryan Ness           |
| `2514-149482-0000.flac` | 2514 | M | S. Young             |
| `3240-131231-0000.flac` | 3240 | M | flakker              |
| `226-122538-0000.flac`  | 226  | F | Deb Bacon-Ziegler    |
| `587-41619-0000.flac`   | 587  | F | Joy Scaglione        |
| `1088-129236-0000.flac` | 1088 | F | Christabel           |
| `1246-124548-0000.flac` | 1246 | F | Sandra               |
| `1263-139804-0000.flac` | 1263 | F | Leonie Rose          |

Gender and speaker metadata were taken from the official
`SPEAKERS.TXT` distributed with LibriSpeech.

### Prompted clips (`sample_audio_prompted/`)

Each file in `sample_audio_prompted/` is a **16 kHz mono WAV** consisting of:

> *"Please identify the gender of the following speech."* (Edge TTS, `en-US-JennyNeural`)
> → 0.5 s silence → LibriSpeech utterance

Total duration per file is roughly 13–20 s.

#### Filename mapping

| WAV file | Gender | Speaker ID | LibriVox reader | Original FLAC |
|----------|--------|-----------|-----------------|---------------|
| `M_1.wav` | M | 307  | Randy Phillips    | `307-127535-0000.flac`  |
| `M_2.wav` | M | 374  | kumarei           | `374-180298-0000.flac`  |
| `M_3.wav` | M | 1743 | Bryan Ness        | `1743-142913-0000.flac` |
| `M_4.wav` | M | 2514 | S. Young          | `2514-149482-0000.flac` |
| `M_5.wav` | M | 3240 | flakker           | `3240-131231-0000.flac` |
| `F_1.wav` | F | 226  | Deb Bacon-Ziegler | `226-122538-0000.flac`  |
| `F_2.wav` | F | 587  | Joy Scaglione     | `587-41619-0000.flac`   |
| `F_3.wav` | F | 1088 | Christabel        | `1088-129236-0000.flac` |
| `F_4.wav` | F | 1246 | Sandra            | `1246-124548-0000.flac` |
| `F_5.wav` | F | 1263 | Leonie Rose       | `1263-139804-0000.flac` |

Original FLAC stems follow the LibriSpeech convention
`<speaker_id>-<chapter_id>-<utterance_id>` from `sample_audio/`.

## Quick start (in Colab)

In the TA-provided notebook, after selecting **Runtime → Change runtime type → T4 GPU**:

```python
!git clone https://github.com/Tincan0325/26spring_ml_hw10_speech_model.git /content/repo

# Run Model A on your audio clip
!bash /content/repo/model_a/setup.sh
!bash /content/repo/model_b/setup.sh

import sys; sys.path.insert(0, "/content/repo")

from model_a.run import run_model_a
run_model_a("/content/your_clip.wav", "/content/out_a.wav")


from model_b.run import run_model_b
run_model_b("/content/your_clip.wav", "/content/out_b.wav")
```

Each `run_model_*` function accepts:
- `audio_path` — input WAV/MP3 (English speech, ≤ 30 s, 16 kHz mono recommended)
- `output_path` — where to save the model's spoken response
- `hf_token` *(optional)* — Hugging Face token if access prompts appear

## Requirements

- Google Colab with **T4 GPU**
- ~15 minutes for first-time setup of each model (downloads weights)

## Task

Listen to both models' responses and compare them. 
