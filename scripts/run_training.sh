#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch GRPO Training
# ─────────────────────────────────────────────────────────────────────────────
# Tweak the env vars below, then: bash scripts/run_training.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
export OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
export NUM_STEPS="${NUM_STEPS:-2000}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LR="${LR:-5e-6}"
export SAVE_EVERY="${SAVE_EVERY:-100}"

# ── Reward settings ───────────────────────────────────────────────────────────
export MOCK_REWARDS="${MOCK_REWARDS:-false}"   # true = random scores for testing
export CL_WORKERS="${CL_WORKERS:-8}"           # parallel scoring workers
export RATE_LIMIT="${RATE_LIMIT:-1.0}"         # seconds between requests per worker

# CapSolver API key (required — get one at capsolver.com)
export CAPSOLVER_API_KEY="${CAPSOLVER_API_KEY:-}"

# Proxy list file — one proxy per line, e.g. http://user:pass@host:port
# Workers rotate through proxies automatically on rate limit / Cloudflare errors
export PROXY_FILE="${PROXY_FILE:-./proxies.txt}"

# ── WandB ─────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-humanizer-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen2.5-3b-grpo-$(date +%Y%m%d-%H%M)}"
# export WANDB_API_KEY="your-key-here"

# Resume from checkpoint (leave empty to start fresh)
export RESUME_FROM="${RESUME_FROM:-}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Humanizer GRPO Training                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Steps:         $NUM_STEPS"
echo "  Batch size:    $BATCH_SIZE prompts x 4 completions each"
echo "  LR:            $LR"
echo "  Output:        $OUTPUT_DIR"
echo "  Mock rewards:  $MOCK_REWARDS"
echo "  CL workers:    $CL_WORKERS"
echo "  Proxy file:    $PROXY_FILE"
echo "  WandB project: $WANDB_PROJECT"
echo ""

if [ "$MOCK_REWARDS" = "false" ]; then
    if [ -z "$CAPSOLVER_API_KEY" ]; then
        echo "ERROR: CAPSOLVER_API_KEY is not set."
        echo "       Export it or set it in this script."
        exit 1
    fi
    if [ ! -f "$PROXY_FILE" ]; then
        echo "WARNING: PROXY_FILE not found at $PROXY_FILE — running without proxies."
        echo "         Rate limits will hit fast. Set PROXY_FILE to your proxy list."
    else
        PROXY_COUNT=$(grep -c . "$PROXY_FILE" 2>/dev/null || echo 0)
        echo "  Proxies loaded: $PROXY_COUNT"
    fi
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU")
echo "  GPU: $GPU_NAME"
echo ""

# ── Make sure we're in the repo root ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Starting training... (logs also saved to logs/)"
echo ""

python -m train.grpo_trainer "$@" 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training finished. Checkpoints in: $OUTPUT_DIR"
