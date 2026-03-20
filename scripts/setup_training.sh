#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod A100 — GRPO Humanizer Training Setup
# ─────────────────────────────────────────────────────────────────────────────
# Run this once after SSHing into your RunPod pod:
#   bash scripts/setup_training.sh
#
# Then launch training:
#   bash scripts/run_training.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Humanizer GRPO — RunPod Setup                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
echo "GPU:  $GPU_NAME"
echo "VRAM: ${VRAM_MB} MB"
echo ""

# ── System packages ───────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git curl wget unzip

# ── Python packages ───────────────────────────────────────────────────────────
echo "[2/5] Installing Python packages..."
pip install --quiet --upgrade pip

echo "      Installing flash-attn (compiles from source, ~10 min)..."
pip install flash-attn --no-build-isolation --quiet || \
    echo "      flash-attn failed (non-fatal, will use eager attention)"

pip install --quiet -r requirements_train.txt

# ── NLTK data ─────────────────────────────────────────────────────────────────
echo "[3/5] Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('NLTK: done')
"

# ── Pre-download Qwen2.5-3B-Instruct ─────────────────────────────────────────
echo "[4/5] Pre-downloading Qwen2.5-3B-Instruct model weights..."
echo "      (This pulls ~6GB — grab a coffee)"
python -c "
from transformers import AutoTokenizer
import huggingface_hub
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=True)
print('  Downloading model weights...')
huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct')
print('  Done.')
"

# ── Create directories ────────────────────────────────────────────────────────
echo "[5/5] Creating runtime directories..."
mkdir -p checkpoints data_cache logs

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Setup complete! Run:                                   ║"
echo "║     bash scripts/run_training.sh                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
