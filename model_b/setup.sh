#!/bin/bash
# Setup script for Model B. Run once before invoking run_model_b().
# All command output is redirected to LOG_FILE; only short progress lines print.
set -e

LOG_FILE="${LOG_FILE:-/tmp/model_b_setup.log}"
: > "$LOG_FILE"
echo "[Model B] Logs: $LOG_FILE"

echo "[Model B] Installing dependency..."
apt-get install -y -qq libsndfile1 ffmpeg >> "$LOG_FILE" 2>&1

if [ ! -d "/content/LLaMA-Omni" ]; then
  git clone -q https://github.com/ictnlp/LLaMA-Omni /content/LLaMA-Omni >> "$LOG_FILE" 2>&1
fi

pip install -qqq git+https://github.com/pytorch/fairseq.git >> "$LOG_FILE" 2>&1
pip install -qqq -r "$(dirname "$0")/requirements.txt" >> "$LOG_FILE" 2>&1
pip install -qqq -e /content/LLaMA-Omni >> "$LOG_FILE" 2>&1

mkdir -p /content/vocoder
if [ ! -f "/content/vocoder/g_00500000" ]; then
  wget -q https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 \
    -O /content/vocoder/g_00500000 >> "$LOG_FILE" 2>&1
fi
if [ ! -f "/content/vocoder/config.json" ]; then
  wget -q https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json \
    -O /content/vocoder/config.json >> "$LOG_FILE" 2>&1
fi

echo "[Model B] Setup complete."
