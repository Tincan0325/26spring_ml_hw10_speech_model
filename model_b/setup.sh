#!/bin/bash
# Setup script for Model B. Run once before invoking run_model_b().
set -e

echo "[Model B] Installing system packages..."
apt-get install -y -q libsndfile1 ffmpeg

echo "[Model B] Cloning support repository..."
if [ ! -d "/content/LLaMA-Omni" ]; then
  git clone https://github.com/ictnlp/LLaMA-Omni /content/LLaMA-Omni
fi

echo "[Model B] Installing fairseq from source..."
pip install -q git+https://github.com/pytorch/fairseq.git

echo "[Model B] Installing Python dependencies..."
pip install -q -r "$(dirname "$0")/requirements.txt"
pip install -q -e /content/LLaMA-Omni

echo "[Model B] Downloading vocoder weights..."
mkdir -p /content/vocoder
if [ ! -f "/content/vocoder/g_00500000" ]; then
  wget -q https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 \
    -O /content/vocoder/g_00500000
fi
if [ ! -f "/content/vocoder/config.json" ]; then
  wget -q https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json \
    -O /content/vocoder/config.json
fi

echo "[Model B] Setup complete."
