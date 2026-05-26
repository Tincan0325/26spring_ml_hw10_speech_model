#!/bin/bash
# Setup script for Model B. Run once before invoking run_model_b().
# All command output is redirected to LOG_FILE; only short progress lines print.
set -e

LOG_FILE="${LOG_FILE:-/tmp/model_b_setup.log}"
: > "$LOG_FILE"
echo "[Model B] Logs: $LOG_FILE"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[Model B] Installing system packages..."
apt-get install -y -qq libsndfile1 ffmpeg >> "$LOG_FILE" 2>&1

echo "[Model B] Cloning DeSTA..."
if [ ! -d "/content/DeSTA2.5-Audio" ]; then
  git clone -q https://github.com/kehanlu/DeSTA2.5-Audio.git /content/DeSTA2.5-Audio >> "$LOG_FILE" 2>&1
fi

echo "[Model B] Installing DeSTA..."
pip install -e /content/DeSTA2.5-Audio >> "$LOG_FILE" 2>&1

echo "[Model B] Pinning transformers==4.49.0 (lulutils requires download_url + is_offline_mode)..."
pip install "transformers==4.49.0" >> "$LOG_FILE" 2>&1

echo "[Model B] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1

echo "[Model B] Downloading model weights..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('wilzzzz/DeSTA3-Llama-3.2-3B-distill')" >> "$LOG_FILE" 2>&1

echo "[Model B] Setup complete."
