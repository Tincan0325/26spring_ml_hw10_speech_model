#!/bin/bash
# Setup script for Model A. Run once before invoking run_model_a().
# All command output is redirected to LOG_FILE; only short progress lines print.
set -e

LOG_FILE="${LOG_FILE:-/tmp/model_a_setup.log}"
: > "$LOG_FILE"
echo "[Model A] Logs: $LOG_FILE"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COSYVOICE_DIR="$SCRIPT_DIR/CosyVoice"

echo "[Model A] Installing system packages..."
apt-get install -y -qq libsndfile1 sox ffmpeg >> "$LOG_FILE" 2>&1

echo "[Model A] Cloning support repository..."
if [ ! -d "$COSYVOICE_DIR" ]; then
  git clone -q --recursive https://github.com/FunAudioLLM/CosyVoice "$COSYVOICE_DIR" >> "$LOG_FILE" 2>&1
fi

echo "[Model A] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
pip install -r "$COSYVOICE_DIR/requirements.txt" >> "$LOG_FILE" 2>&1

echo "[Model A] Downloading model weights..."
pip install -q "huggingface_hub[cli]" >> "$LOG_FILE" 2>&1
huggingface-cli download FunAudioLLM/CosyVoice-300M \
  --local-dir "$COSYVOICE_DIR/pretrained_models/CosyVoice-300M" \
  >> "$LOG_FILE" 2>&1

echo "[Model A] Setup complete."
