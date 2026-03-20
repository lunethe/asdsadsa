"""
Run this on RunPod to see the actual checkpoint keys and model structure.
python debug_desklib.py
"""
import safetensors.torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, DebertaV2ForSequenceClassification

MODEL_ID = "desklib/ai-text-detector-v1.01"

print("=== CHECKPOINT KEYS ===")
path = hf_hub_download(MODEL_ID, "model.safetensors")
raw_sd = safetensors.torch.load_file(path)
for k, v in raw_sd.items():
    print(f"  {k:70s} {tuple(v.shape)}")

print("\n=== MODEL CONFIG ===")
config = AutoConfig.from_pretrained(MODEL_ID)
print(f"  num_labels: {config.num_labels}")
print(f"  architectures: {config.architectures}")
print(f"  id2label: {config.id2label}")

print("\n=== MODEL EXPECTED KEYS (num_labels from config) ===")
model = DebertaV2ForSequenceClassification(config)
for k, v in model.state_dict().items():
    print(f"  {k:70s} {tuple(v.shape)}")
