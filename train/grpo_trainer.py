"""
GRPO Trainer — Qwen2.5-3B Humanizer
=====================================
Group Relative Policy Optimization for training a text humanizer.
Reward signal comes from Copyleaks AI content detector.

Algorithm (from AuthorMist / DeepSeek-R1-Zero paper):
  For each training step:
    1. Sample B prompts (AI texts) from the dataset.
    2. For each prompt, generate G completions from the current policy π_θ.
    3. Score all B×G completions with the reward function.
    4. Compute per-group normalized advantages:
         A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
    5. Compute GRPO loss with PPO-style clipping + KL penalty:
         L = -E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)] + β × KL(π_θ ∥ π_ref)
    6. Backprop and update.
    7. Log to WandB, save checkpoints.

Key design decisions:
  - LoRA (r=16) on Qwen2.5-3B-Instruct to fit in A100 80GB alongside browser workers.
  - Reference model = frozen base LoRA-disabled model (no extra GPU memory needed
    with PEFT's disable_adapter context manager).
  - Copyleaks scoring is async; training loop awaits reward batches before update.
  - Gradient checkpointing to reduce activation memory.
"""

import asyncio
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .config import Config
from .data_collector import InfiniteDataLoader, load_all_datasets, TrainingSample
from .reward_copyleaks import CopyleaksRewardPool, ScoreResult
from .reward_gptzero import GPTZeroRewardPool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    """Load Qwen2.5-3B-Instruct with LoRA."""
    model_name = cfg.model.model_name
    dtype = getattr(torch, cfg.model.torch_dtype)  # e.g. torch.bfloat16

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",   # left-pad for causal LM batch generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {model_name} ({cfg.model.torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if cfg.model.use_flash_attention else "eager",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    if cfg.model.use_lora:
        logger.info(
            f"Applying LoRA: r={cfg.model.lora_r}, alpha={cfg.model.lora_alpha}, "
            f"targets={cfg.model.lora_target_modules}"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=cfg.model.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    return model, tokenizer


def build_prompt(cfg: Config, ai_text: str, tokenizer) -> str:
    """
    Format the prompt using Qwen's chat template.
    Returns the raw string (not tokenized) with the assistant header appended
    so the model continues from there.
    """
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": cfg.user_prompt_template.format(text=ai_text)},
    ]
    # apply_chat_template with add_generation_prompt=True appends the
    # <|im_start|>assistant\n header, priming generation.
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_str


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[List[str]], List[torch.Tensor], List[torch.Tensor]]:
    """
    For each prompt, generate `group_size` completions.

    Returns:
        texts:       List[List[str]]      — decoded completions [B][G]
        prompt_lens: List[int]            — number of prompt tokens per prompt
        input_ids_list: List[Tensor]      — full sequence input ids [B][G]
    """
    device = next(model.parameters()).device

    all_texts: List[List[str]] = []
    all_input_ids: List[List[torch.Tensor]] = []
    all_prompt_lens: List[int] = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        ).to(device)

        prompt_len = enc["input_ids"].shape[1]
        all_prompt_lens.append(prompt_len)

        # Repeat prompt G times for batched generation
        input_ids = enc["input_ids"].repeat(group_size, 1)
        attention_mask = enc["attention_mask"].repeat(group_size, 1)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens (after the prompt)
        group_texts = []
        group_ids = []
        for i in range(group_size):
            completion_ids = outputs[i, prompt_len:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            group_texts.append(text)
            group_ids.append(outputs[i].detach().cpu())  # detach from inference graph

        all_texts.append(group_texts)
        all_input_ids.append(group_ids)

    return all_texts, all_prompt_lens, all_input_ids


# ─────────────────────────────────────────────────────────────────────────────
# Log probability computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sequence_log_probs(
    model,
    input_ids: torch.Tensor,      # (1, seq_len)
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute the mean log-probability of the completion tokens.
    Prompt tokens are masked out (we only care about completion likelihood).

    Returns scalar tensor.
    """
    # Clone to detach from any inference-mode graph (tensors produced inside
    # torch.no_grad() / generate() cannot be saved for backward as-is).
    input_ids = input_ids.clone()
    input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]           # (1, seq_len-1, vocab)
    shift_labels = input_ids[:, 1:]            # (1, seq_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (1, seq_len-1, vocab)
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (1, seq_len-1)

    # Mask out prompt tokens; only average over completion
    completion_start = prompt_len - 1  # offset by 1 due to shift
    completion_log_probs = token_log_probs[:, completion_start:]  # (1, completion_len)

    return completion_log_probs.mean()


def compute_batch_log_probs(
    model,
    all_input_ids: List[List[torch.Tensor]],  # [B][G]
    all_prompt_lens: List[int],               # [B]
) -> torch.Tensor:
    """
    Compute log probs for all B×G sequences.
    Returns Tensor of shape (B*G,).
    Processes one sequence at a time to avoid OOM (sequences have variable length).
    """
    log_probs = []
    for b_idx, (group_ids, prompt_len) in enumerate(
        zip(all_input_ids, all_prompt_lens)
    ):
        for ids in group_ids:
            device = next(model.parameters()).device
            ids = ids.to(device)
            lp = compute_sequence_log_probs(model, ids, prompt_len)
            log_probs.append(lp)
    return torch.stack(log_probs)  # (B*G,)


# ─────────────────────────────────────────────────────────────────────────────
# GRPO loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    log_probs: torch.Tensor,        # (B*G,) current policy, requires_grad
    old_log_probs: torch.Tensor,    # (B*G,) from rollout (detached)
    ref_log_probs: torch.Tensor,    # (B*G,) reference model (detached)
    rewards: torch.Tensor,          # (B*G,) from reward function
    group_size: int,
    clip_epsilon: float,
    kl_beta: float,
) -> Tuple[torch.Tensor, dict]:
    """
    GRPO objective:
      L = -E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)] + β*KL

    where r(θ) = exp(log π_θ(y) - log π_old(y))
    and A_i = (reward_i - mean_group) / std_group
    """
    B_G = log_probs.shape[0]
    B = B_G // group_size

    # ── Per-group advantage normalization ─────────────────────────────────
    rewards_grouped = rewards.view(B, group_size)          # (B, G)
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)     # (B, 1)
    std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
    advantages_grouped = (rewards_grouped - mean_r) / std_r  # (B, G)
    advantages = advantages_grouped.view(B_G)              # (B*G,)

    # ── Importance sampling ratio ──────────────────────────────────────────
    ratio = torch.exp(log_probs - old_log_probs)

    # ── PPO-style clipped objective ────────────────────────────────────────
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # ── KL divergence penalty (forward KL: π_ref ∥ π_θ) ──────────────────
    # Approximate: KL ≈ log π_old / π_θ (using old as proxy for ref when
    # the gap is small, or use actual ref_log_probs)
    kl = (ref_log_probs - log_probs).mean()   # penalize drift from reference
    kl_loss = kl_beta * kl

    total_loss = policy_loss + kl_loss

    metrics = {
        "loss/policy": policy_loss.item(),
        "loss/kl": kl.item(),
        "loss/total": total_loss.item(),
        "reward/mean": rewards.mean().item(),
        "reward/std": rewards.std().item(),
        "reward/max": rewards.max().item(),
        "advantage/mean": advantages.mean().item(),
        "ratio/mean": ratio.mean().item(),
        "ratio/clip_frac": ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon))
                           .float().mean().item(),
    }

    return total_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Reference model log probs (no grad, LoRA disabled)
# ─────────────────────────────────────────────────────────────────────────────

def get_ref_log_probs(
    model,  # PeftModel
    all_input_ids: List[List[torch.Tensor]],
    all_prompt_lens: List[int],
) -> torch.Tensor:
    """
    Compute log probs under the reference (pre-LoRA) policy.
    Uses PEFT's disable_adapter() to temporarily freeze LoRA weights,
    so no second model copy is needed.
    """
    from peft import PeftModel as _PeftModel
    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            return compute_batch_log_probs(model, all_input_ids, all_prompt_lens)
    else:
        # No LoRA: reference == current policy (no KL signal)
        return compute_batch_log_probs(model, all_input_ids, all_prompt_lens).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.training.output_dir, exist_ok=True)

        # WandB
        self.use_wandb = bool(cfg.training.wandb_project)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.training.wandb_project,
                    name=cfg.training.wandb_run_name,
                    config={
                        "model": cfg.model.model_name,
                        "lora_r": cfg.model.lora_r,
                        "group_size": cfg.grpo.group_size,
                        "clip_epsilon": cfg.grpo.clip_epsilon,
                        "kl_beta": cfg.grpo.kl_beta,
                        "lr": cfg.training.learning_rate,
                        "batch_size": cfg.training.batch_size,
                    },
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging.")
                self.use_wandb = False

    def _log(self, metrics: dict, step: int):
        msg = f"Step {step:05d} | " + " | ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
        )
        logger.info(msg)
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def _save_checkpoint(self, model, tokenizer, step: int):
        ckpt_dir = os.path.join(self.cfg.training.output_dir, f"step_{step:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def train(self):
        cfg = self.cfg

        # ── Load data ─────────────────────────────────────────────────────
        logger.info("Loading datasets...")
        samples = load_all_datasets(
            cfg.data.datasets,
            cache_dir=cfg.data.cache_dir,
            min_chars=cfg.data.min_chars,
            max_chars=cfg.data.max_chars,
            seed=cfg.data.seed,
        )
        if not samples:
            raise RuntimeError("No training samples loaded. Check dataset config.")
        dataloader = InfiniteDataLoader(samples, seed=cfg.data.seed)

        # ── Load model ────────────────────────────────────────────────────
        model, tokenizer = load_model_and_tokenizer(cfg)

        # Optionally resume — reload base model without LoRA, then load checkpoint
        if cfg.training.resume_from:
            logger.info(f"Resuming from {cfg.training.resume_from}")
            dtype = getattr(torch, cfg.model.torch_dtype)
            base = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if cfg.model.use_flash_attention else "eager",
            )
            base.config.use_cache = False
            model = PeftModel.from_pretrained(base, cfg.training.resume_from, is_trainable=True)
            model.gradient_checkpointing_enable()
            model.print_trainable_parameters()

        # ── Optimizer and scheduler ───────────────────────────────────────
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=cfg.training.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_steps - cfg.training.warmup_steps,
            eta_min=cfg.training.learning_rate * 0.1,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.training.warmup_steps],
        )

        # ── Reward pool ───────────────────────────────────────────────────
        async def run_training():
            async with CopyleaksRewardPool(
                mock_mode=cfg.copyleaks.mock_mode,
                num_workers=cfg.copyleaks.num_workers,
            ) as reward_pool:
                await _training_loop(
                    cfg, model, tokenizer, dataloader, optimizer, scheduler,
                    reward_pool, self._log, self._save_checkpoint,
                )

        asyncio.run(run_training())

        logger.info("Training complete.")
        if self.use_wandb:
            self.wandb.finish()


async def _training_loop(
    cfg: Config,
    model,
    tokenizer,
    dataloader: InfiniteDataLoader,
    optimizer,
    scheduler,
    reward_pool: CopyleaksRewardPool,
    log_fn,
    save_fn,
):
    """Inner async training loop (needs to run inside asyncio for reward pool)."""
    device = next(model.parameters()).device
    step = 0
    t_start = time.time()

    logger.info(
        f"Starting GRPO training: {cfg.training.num_steps} steps, "
        f"batch_size={cfg.training.batch_size}, group_size={cfg.grpo.group_size}"
    )

    for step in range(1, cfg.training.num_steps + 1):
        step_start = time.time()

        # ── 1. Sample prompts ────────────────────────────────────────────
        batch: List[TrainingSample] = dataloader.next_batch(cfg.training.batch_size)
        ai_texts = [s.text for s in batch]
        prompts = [build_prompt(cfg, t, tokenizer) for t in ai_texts]

        # ── 2. Rollout: generate G completions per prompt ────────────────
        model.eval()
        with torch.no_grad():
            completions_by_prompt, prompt_lens, input_ids_by_prompt = \
                generate_completions(
                    model, tokenizer, prompts,
                    group_size=cfg.grpo.group_size,
                    max_new_tokens=cfg.grpo.max_new_tokens,
                    temperature=cfg.grpo.temperature,
                    top_p=cfg.grpo.top_p,
                )

        # ── 3. Compute old log probs (for importance sampling) ────────────
        old_log_probs = compute_batch_log_probs(
            model, input_ids_by_prompt, prompt_lens
        ).detach()  # (B*G,)

        # ── 4. Reference log probs (KL baseline) ─────────────────────────
        ref_log_probs = get_ref_log_probs(
            model, input_ids_by_prompt, prompt_lens
        ).detach()  # (B*G,)

        # ── 5. Score completions with Copyleaks ───────────────────────────
        # Flatten B×G completions for batch scoring
        flat_completions: List[str] = []
        for group in completions_by_prompt:
            flat_completions.extend(group)

        # Apply post-processing before scoring (same pipeline as inference).
        # Import from app.postprocess, NOT app.main — importing main.py would
        # trigger DIPPER model loading and fill the disk.
        try:
            from app.postprocess import post_process
            scored_texts = [post_process(t) for t in flat_completions]
        except Exception:
            scored_texts = flat_completions  # fallback: score raw completions

        score_results: List[ScoreResult] = await reward_pool.score_batch(scored_texts)
        rewards_list = [r.reward for r in score_results]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # ── 6. GRPO policy update ─────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        # Recompute log probs with grad enabled
        log_probs = compute_batch_log_probs(
            model, input_ids_by_prompt, prompt_lens
        )  # (B*G,), requires_grad

        total_loss, metrics = compute_grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            group_size=cfg.grpo.group_size,
            clip_epsilon=cfg.grpo.clip_epsilon,
            kl_beta=cfg.grpo.kl_beta,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            cfg.training.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        # ── 7. Logging ────────────────────────────────────────────────────
        step_time = time.time() - step_start
        metrics["lr"] = scheduler.get_last_lr()[0]
        metrics["step_time_s"] = step_time
        metrics["elapsed_h"] = (time.time() - t_start) / 3600

        if step % cfg.training.log_every == 0:
            # Log a sample completion for qualitative inspection
            best_idx = rewards.argmax().item()
            best_text = flat_completions[best_idx][:200]
            logger.info(f"Best completion (reward={rewards[best_idx]:.3f}): {best_text!r}")
            log_fn(metrics, step)

        # ── 8. Checkpoint + GPTZero eval ──────────────────────────────────
        if step % cfg.training.save_every == 0:
            save_fn(model, tokenizer, step)
            gptzero_key = os.getenv("GPTZERO_API_KEY", "")
            if gptzero_key and best_text:
                try:
                    async with GPTZeroRewardPool(num_workers=1, api_key=gptzero_key) as gz:
                        gz_score = await gz._workers[0].score(best_text)
                    log_fn({"gptzero/ai_prob": gz_score.ai_probability,
                            "gptzero/reward": gz_score.reward}, step)
                    logger.info(f"[GPTZero eval] step={step} AI={gz_score.ai_probability:.2%}")
                except Exception as e:
                    logger.warning(f"GPTZero eval failed: {e}")

    # Final checkpoint
    save_fn(model, tokenizer, step)
    logger.info(f"Training finished in {(time.time() - t_start)/3600:.2f} hours.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("train.log"),
        ],
    )

    cfg = Config()

    # Quick CLI overrides
    if "--mock" in sys.argv:
        cfg.copyleaks.mock_mode = True
        logger.info("Mock reward mode enabled (Copyleaks will not be called).")

    if "--steps" in sys.argv:
        idx = sys.argv.index("--steps")
        cfg.training.num_steps = int(sys.argv[idx + 1])

    trainer = GRPOTrainer(cfg)
    trainer.train()
