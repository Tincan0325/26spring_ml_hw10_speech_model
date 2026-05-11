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
  git clone -q --depth 1 --recursive --shallow-submodules https://github.com/FunAudioLLM/CosyVoice "$COSYVOICE_DIR" >> "$LOG_FILE" 2>&1
fi

echo "[Model A] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
# CosyVoice's requirements pin many old versions that fail to build on Python 3.12
# or pull in services we don't need for inference (grpc server, fastapi UI, training tools).
# Filter to only the packages CosyVoice's inference path actually imports.
# `wget=` filters the pinned wget==3.2 line so we re-install the python `wget` package below.
FILTER='^(openai-whisper|grpcio|grpcio-tools|deepspeed|tensorrt|gradio|fastapi|fastapi-cli|uvicorn|tensorboard|gdown|wget=|pyarrow|torch=|torchaudio=|numpy=|protobuf=|transformers=|pydantic=|matplotlib=|networkx=|onnx=|--extra-index-url)'
grep -Ev "$FILTER" "$COSYVOICE_DIR/requirements.txt" > /tmp/cosyvoice_requirements_filtered.txt
pip install -r /tmp/cosyvoice_requirements_filtered.txt >> "$LOG_FILE" 2>&1
# wget is imported at module-load time by Matcha-TTS; onnxruntime 1.18 was built
# against numpy<2 and crashes on Colab's numpy 2.x, so pin a numpy-2-compatible version.
pip install -q wget "onnxruntime-gpu>=1.19" >> "$LOG_FILE" 2>&1

echo "[Model A] Downloading model weights..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='$COSYVOICE_DIR/pretrained_models/CosyVoice-300M-SFT')" >> "$LOG_FILE" 2>&1

echo "[Model A] Setup complete."
