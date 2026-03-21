"""
Test the trained checkpoint on a few sample texts.
Usage: python scripts/test_checkpoint.py [checkpoint_dir]
Default checkpoint: ./checkpoints/step_2000
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/step_2000"
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a text rewriter. Rewrite the given text so it sounds like a real "
    "person wrote it — natural, conversational, and varied. Avoid formal or "
    "robotic phrasing. Preserve all facts and meaning exactly. "
    "Output ONLY the rewritten text. No intro, no explanation, no meta-commentary."
)

USER_TEMPLATE = (
    "Rewrite this text so it sounds like a real human wrote it. "
    "Output the rewrite only — no intro, no 'Here is', no explanation.\n\n{text}"
)

SAMPLES = [
    "Artificial intelligence has significantly transformed various industries and "
    "continues to reshape how organizations operate and deliver value to their "
    "customers around the world. The rapid advancement of machine learning algorithms "
    "has enabled unprecedented capabilities in automation and decision-making.",

    "Climate change is one of the most pressing issues facing humanity today. "
    "Rising global temperatures have led to more frequent and severe weather events, "
    "including hurricanes, droughts, and floods. Scientists agree that immediate "
    "action is required to mitigate the worst effects of climate change.",

    "Regular exercise has numerous benefits for both physical and mental health. "
    "Studies have shown that engaging in at least 30 minutes of moderate physical "
    "activity per day can reduce the risk of chronic diseases, improve mood, "
    "and enhance cognitive function.",
]

print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)

print(f"Loading LoRA checkpoint: {CHECKPOINT}")
model = PeftModel.from_pretrained(model, CHECKPOINT)
model.eval()
print("Model ready.\n")
print("=" * 70)

for i, text in enumerate(SAMPLES, 1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    result = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    print(f"[Sample {i}]")
    print(f"ORIGINAL: {text[:120]}...")
    print(f"REWRITE:  {result}")
    print("=" * 70)
