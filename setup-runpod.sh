#!/bin/bash
# RunPod Setup Script for Full DIPPER (11B)
# Run this after SSH-ing into your RunPod A100/A6000 pod
# Expected time: ~15-20 min for model download, then server starts

set -e

echo "=== AI Text Humanizer - RunPod Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers fastapi uvicorn nltk sentencepiece accelerate pydantic

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Check VRAM and decide model
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
echo "Detected VRAM: ${VRAM}MB"

if [ "${VRAM}" -ge 40000 ]; then
    echo "Sufficient VRAM for full DIPPER (11B). Downloading..."
    export HUMANIZER_MODEL=full
    
    # Pre-download the model to cache
    python -c "
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

print('Downloading T5-XXL tokenizer...')
T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')

print('Downloading DIPPER model (~45GB in FP16)...')
T5ForConditionalGeneration.from_pretrained(
    'kalpeshk2011/dipper-paraphraser-xxl',
    torch_dtype=torch.float16,
    device_map='auto'
)
print('Model downloaded successfully!')
"
else
    echo "VRAM < 40GB, using lightweight DIPPER model (~770M params)"
    export HUMANIZER_MODEL=lightweight
    
    python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('Downloading lightweight DIPPER...')
AutoTokenizer.from_pretrained('google/t5-efficient-large-nl32')
AutoModelForSeq2SeqLM.from_pretrained('SamSJackson/paraphrase-dipper-no-ctx')
print('Model downloaded!')
"
fi

echo ""
echo "=== Starting server ==="
echo "Model: ${HUMANIZER_MODEL}"
echo "API: http://0.0.0.0:8000"
echo "Docs: http://0.0.0.0:8000/docs"
echo ""

cd /workspace/humanizer-project  # or wherever you cloned the repo
uvicorn app.main:app --host 0.0.0.0 --port 8000
