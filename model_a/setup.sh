#!/bin/bash
# Setup script for Model A. Run once before invoking run_model_a().
set -e

echo "[Model A] Installing system packages..."
apt-get install -y -q libsndfile1 sox ffmpeg

echo "[Model A] Cloning support repository..."
if [ ! -d "/content/CosyVoice" ]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice /content/CosyVoice
fi

echo "[Model A] Installing Python dependencies..."
pip install -q -r "$(dirname "$0")/requirements.txt"
pip install -q -r /content/CosyVoice/requirements.txt

echo "[Model A] Downloading model weights..."
pip install -q "huggingface_hub[cli]"
huggingface-cli download FunAudioLLM/CosyVoice-300M \
  --local-dir /content/CosyVoice/pretrained_models/CosyVoice-300M

echo "[Model A] Setup complete."
