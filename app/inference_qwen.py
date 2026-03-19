"""
Qwen2.5-3B Inference Module
=============================
Loads the GRPO-fine-tuned Qwen2.5-3B-Instruct model and exposes a
humanize() method compatible with the existing FastAPI server.

Set env var HUMANIZER_MODEL=qwen to use this instead of DIPPER.
Set QWEN_CHECKPOINT to a LoRA adapter directory to load a fine-tuned model.
If QWEN_CHECKPOINT is empty, loads the base Qwen2.5-3B-Instruct.

The post_process() function from app.main is applied after generation,
preserving the existing DIPPER pipeline's AI-word scrubbing and ZWC injection.
"""

import logging
import os
import time
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a text rewriter. Your job is to rewrite AI-generated text so it "
    "reads like it was written by a real person. Make it sound natural, "
    "conversational, and varied. Avoid formal or robotic phrasing. "
    "Preserve the original meaning and key facts exactly."
)

USER_TEMPLATE = (
    "Rewrite the following AI-generated text so it passes AI detection tools "
    "and reads like natural human writing. Keep all the facts and meaning. "
    "Do not add disclaimers or meta-commentary.\n\n"
    "AI text:\n{text}\n\n"
    "Human rewrite:"
)


class QwenHumanizer:
    """
    Wraps the fine-tuned Qwen2.5-3B model for inference.
    Provides the same .humanize() interface as DipperParaphraser.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        dtype = getattr(torch, torch_dtype)
        logger.info(f"Loading Qwen tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading Qwen model: {model_name} ({torch_dtype})")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        # Load LoRA adapter if a fine-tuned checkpoint is provided
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            logger.info(f"Loading LoRA adapter from: {checkpoint_dir}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)
            self.model = self.model.merge_and_unload()  # fuse LoRA into weights
            logger.info("LoRA adapter merged into base model.")

        self.model.eval()
        self.device = device
        logger.info(f"Qwen model ready in {time.time() - t0:.1f}s")

    def humanize(
        self,
        text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Rewrite `text` to sound human-written.
        Returns the raw completion (before post_process).
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0, prompt_len:]
        result = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return result.strip()
