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
echo "║   Humanizer GRPO — RunPod A100 Setup                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
echo "GPU:  $GPU_NAME"
echo "VRAM: ${VRAM_MB} MB"
echo ""

if [ "$VRAM_MB" -lt 40000 ]; then
    echo "WARNING: Less than 40GB VRAM detected. Qwen2.5-3B + LoRA needs ~24GB."
    echo "         Flash attention and bfloat16 are enabled to minimize memory."
fi

# ── System packages ───────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git curl wget unzip chromium-browser || true
# Playwright will install its own Chromium, but system Chromium is a useful fallback

# ── Python packages ───────────────────────────────────────────────────────────
echo "[2/6] Installing Python packages..."
pip install --quiet --upgrade pip

# Flash Attention 2 first (takes longest, compile from source)
echo "      Installing flash-attn (compiles from source, ~10 min)..."
pip install flash-attn --no-build-isolation --quiet || \
    echo "      flash-attn failed (non-fatal, will use eager attention)"

pip install --quiet -r requirements_train.txt

# ── Playwright browsers ───────────────────────────────────────────────────────
echo "[3/6] Installing Playwright Chromium..."
playwright install chromium
playwright install-deps chromium

# ── NLTK data ─────────────────────────────────────────────────────────────────
echo "[4/6] Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('NLTK: done')
"

# ── Pre-download Qwen2.5-3B-Instruct ─────────────────────────────────────────
echo "[5/6] Pre-downloading Qwen2.5-3B-Instruct model weights..."
echo "      (This pulls ~6GB — grab a coffee)"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('  Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=True)
print('  Tokenizer done.')

print('  Downloading model weights...')
# Just download to cache; don't load into GPU memory yet
import huggingface_hub
huggingface_hub.snapshot_download('Qwen/Qwen2.5-3B-Instruct')
print('  Model weights cached.')
"

# ── Create directories ────────────────────────────────────────────────────────
echo "[6/6] Creating runtime directories..."
mkdir -p checkpoints data_cache logs

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Setup complete! Run:                                   ║"
echo "║     bash scripts/run_training.sh                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
