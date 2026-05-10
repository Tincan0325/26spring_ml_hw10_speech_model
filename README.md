# Speech Model Comparison — HW10

Inference code for two speech models, **Model A** and **Model B**.

This repository is meant to be cloned and used from a Colab notebook provided by your TA.
Do **not** modify the code in `model_a/` or `model_b/` — your task is to compare what each
model produces, not to change them.

## Folder layout

```
model_a/    Inference code for Model A
model_b/    Inference code for Model B
sample_audio/   Example input clip
```

## Quick start (in Colab)

In the TA-provided notebook, after selecting **Runtime → Change runtime type → T4 GPU**:

```python
!git clone https://github.com/<your-repo>/speech-model-comparison /content/repo

# Run Model A on your audio clip
!bash /content/repo/model_a/setup.sh
import sys; sys.path.insert(0, "/content/repo")
from model_a.run import run_model_a
run_model_a("/content/your_clip.wav", "/content/out_a.wav")

# Free GPU memory between models (the TA notebook handles this).

# Run Model B
!bash /content/repo/model_b/setup.sh
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

Listen to both models' responses and compare them. Refer to your TA's instructions for
the questions you need to answer.
