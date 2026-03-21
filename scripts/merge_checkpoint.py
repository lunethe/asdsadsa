"""
Merge LoRA weights into the base model for deployment.
Usage: python scripts/merge_checkpoint.py [checkpoint_dir] [output_dir]
Defaults: ./checkpoints/step_2000  →  ./merged_model
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CHECKPOINT  = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/step_2000"
OUTPUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "./merged_model"
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"

print(f"Base model:  {BASE_MODEL}")
print(f"Checkpoint:  {CHECKPOINT}")
print(f"Output:      {OUTPUT_DIR}")
print()

print("Loading base model on CPU (safe for merge)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, CHECKPOINT)

print("Merging LoRA into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print()
print(f"Done. Merged model saved to: {OUTPUT_DIR}")
print("You can now load it with AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)")
