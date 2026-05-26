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

echo "[Model B] Cloning support repository..."
if [ ! -d "/content/LLaMA-Omni" ]; then
  git clone -q --depth 1 https://github.com/ictnlp/LLaMA-Omni /content/LLaMA-Omni >> "$LOG_FILE" 2>&1
fi

echo "[Model B] Installing Python dependencies..."
# fairseq pins omegaconf<2.1 whose old metadata can't be parsed by modern pip,
# and LLaMA-Omni pins torch==2.1.2 which isn't available on Python 3.12.
# Install both with --no-deps and supply the deps Model B's inference path uses.
pip install --no-deps git+https://github.com/pytorch/fairseq.git >> "$LOG_FILE" 2>&1
pip install --no-deps -e /content/LLaMA-Omni >> "$LOG_FILE" 2>&1
pip install bitarray sacrebleu hydra-core omegaconf >> "$LOG_FILE" 2>&1
pip install -r "$SCRIPT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
pip install openai-whisper >> "$LOG_FILE" 2>&1
# LLaMA-Omni's generation code uses transformers internals that were removed after 4.45.
pip install "transformers==4.43.4" >> "$LOG_FILE" 2>&1

echo "[Model B] Patching framework sources..."
python3 "$SCRIPT_DIR/_apply_patches.py" >> "$LOG_FILE" 2>&1

echo "[Model B] Downloading vocoder..."
mkdir -p /content/vocoder
if [ ! -f "/content/vocoder/g_00500000" ]; then
  wget -q https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 \
    -O /content/vocoder/g_00500000 >> "$LOG_FILE" 2>&1
fi
if [ ! -f "/content/vocoder/config.json" ]; then
  wget -q https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json \
    -O /content/vocoder/config.json >> "$LOG_FILE" 2>&1
fi

echo "[Model B] Downloading speech encoder (whisper-large-v3)..."
# LLaMA-Omni's config points speech_encoder at "models/speech_encoder/large-v3.pt".
# Pre-download so the first run doesn't block on a 3 GB fetch.
mkdir -p /content/models/speech_encoder
if [ ! -f "/content/models/speech_encoder/large-v3.pt" ]; then
  python3 -c "import whisper; whisper._download(whisper._MODELS['large-v3'], '/content/models/speech_encoder', in_memory=False)" >> "$LOG_FILE" 2>&1
fi

echo "[Model B] Setup complete."
