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
export BATCH_SIZE="${BATCH_SIZE:-2}"       # prompts per gradient step
export LR="${LR:-5e-6}"
export SAVE_EVERY="${SAVE_EVERY:-100}"

# Copyleaks reward settings
export MOCK_REWARDS="${MOCK_REWARDS:-false}"  # set true for offline testing
export CL_WORKERS="${CL_WORKERS:-2}"          # parallel browser instances
export RATE_LIMIT="${RATE_LIMIT:-4.0}"        # seconds between requests/worker
export CL_HEADLESS="${CL_HEADLESS:-true}"

# WandB — set your key here or via `wandb login` before running
export WANDB_PROJECT="${WANDB_PROJECT:-humanizer-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen2.5-3b-grpo-$(date +%Y%m%d-%H%M)}"
# export WANDB_API_KEY="your-key-here"       # or set via `wandb login`

# Resume from checkpoint (leave empty to start fresh)
export RESUME_FROM="${RESUME_FROM:-}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Humanizer GRPO Training                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Steps:         $NUM_STEPS"
echo "  Batch size:    $BATCH_SIZE prompts × 4 completions each"
echo "  LR:            $LR"
echo "  Output:        $OUTPUT_DIR"
echo "  Mock rewards:  $MOCK_REWARDS"
echo "  CL workers:    $CL_WORKERS"
echo "  WandB project: $WANDB_PROJECT"
echo ""

if [ "$MOCK_REWARDS" = "true" ]; then
    echo "  ⚠️  MOCK MODE: Copyleaks will NOT be called."
    echo "     Rewards are random — use only for pipeline testing."
    echo ""
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU")
echo "  GPU: $GPU_NAME"
echo ""

# ── Verify Playwright is installed ────────────────────────────────────────────
if [ "$MOCK_REWARDS" = "false" ]; then
    python -c "from playwright.async_api import async_playwright; print('Playwright: OK')" || {
        echo "ERROR: Playwright not installed. Run: pip install playwright && playwright install chromium"
        exit 1
    }
fi

# ── Make sure we're in the repo root ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Starting training... (logs → train.log)"
echo ""

# tee: logs go to both stdout and file
python -m train.grpo_trainer "$@" 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training finished. Checkpoints in: $OUTPUT_DIR"
