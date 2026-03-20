"""
Training Configuration for GRPO Humanizer
==========================================
All hyperparameters in one place. Override via env vars for RunPod.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Base model to fine-tune
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    # Load in bfloat16 on A100
    torch_dtype: str = "bfloat16"
    # Use flash attention 2 if available
    use_flash_attention: bool = True

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Apply LoRA to all linear projections in Qwen
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class GRPOConfig:
    # Number of completions to sample per prompt (group size G)
    group_size: int = 4
    # PPO clipping epsilon
    clip_epsilon: float = 0.2
    # KL penalty coefficient (beta)
    kl_beta: float = 0.04
    # Number of policy update steps per rollout batch
    ppo_epochs: int = 1
    # Max new tokens for generation
    max_new_tokens: int = 512
    # Generation temperature
    temperature: float = 0.9
    # Top-p sampling
    top_p: float = 0.9


@dataclass
class TrainingConfig:
    # Output directory for checkpoints
    output_dir: str = os.getenv("OUTPUT_DIR", "./checkpoints")
    # WandB project name (set WANDB_PROJECT env var, or "" to disable)
    wandb_project: str = os.getenv("WANDB_PROJECT", "humanizer-grpo")
    wandb_run_name: str = os.getenv("WANDB_RUN_NAME", "qwen2.5-3b-grpo")

    # Training steps (not epochs — GRPO is online RL)
    num_steps: int = int(os.getenv("NUM_STEPS", "2000"))
    # Prompts per gradient step
    batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    # Effective batch = batch_size * group_size completions per step

    # Optimizer
    learning_rate: float = float(os.getenv("LR", "5e-6"))
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LR schedule: "cosine" or "constant"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 50

    # Save checkpoint every N steps
    save_every: int = int(os.getenv("SAVE_EVERY", "100"))
    # Log every N steps
    log_every: int = 10

    # Resume from checkpoint (path to adapter dir)
    resume_from: Optional[str] = os.getenv("RESUME_FROM", None)


@dataclass
class DataConfig:
    # HuggingFace datasets to pull AI text from
    datasets: list = field(default_factory=lambda: [
        "Hello-SimpleAI/HC3",           # ChatGPT answers
    ])
    hc3_split: str = "train"
    # Max characters per training sample (Copyleaks free tier limit ~3000 chars)
    max_chars: int = 2500
    # Min characters (too short = uninformative reward)
    min_chars: int = 200
    # Local cache dir for datasets
    cache_dir: str = os.getenv("DATA_CACHE", "./data_cache")
    # Pre-shuffle seed
    seed: int = 42


@dataclass
class CopyleaksConfig:
    # Use mock rewards (True = random scores, for offline dev/testing)
    mock_mode: bool = os.getenv("MOCK_REWARDS", "false").lower() == "true"
    # Seconds between Copyleaks requests (respect rate limit)
    rate_limit_seconds: float = float(os.getenv("RATE_LIMIT", "4.0"))
    # Number of parallel Playwright browser instances
    num_workers: int = int(os.getenv("CL_WORKERS", "2"))
    # Max retries per text
    max_retries: int = 3
    # Request timeout in seconds
    timeout_seconds: float = 45.0
    # Path to proxy list file (one proxy per line, e.g. http://user:pass@host:port)
    # Leave empty to use no proxy (direct IP). Get proxies from Webshare/ProxyMesh etc.
    proxy_file: str = os.getenv("PROXY_FILE", "")
    # Headless browser (no longer used, kept for compatibility)
    headless: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    copyleaks: CopyleaksConfig = field(default_factory=CopyleaksConfig)

    # System prompt injected before every humanize request
    system_prompt: str = (
        "You are a text rewriter. Rewrite the given text so it sounds like a real "
        "person wrote it — natural, conversational, and varied. Avoid formal or "
        "robotic phrasing. Preserve all facts and meaning exactly. "
        "Output ONLY the rewritten text. No intro, no explanation, no meta-commentary."
    )

    # User prompt template — {text} is replaced with the AI text
    user_prompt_template: str = (
        "Rewrite this text so it sounds like a real human wrote it. "
        "Output the rewrite only — no intro, no 'Here is', no explanation.\n\n"
        "{text}"
    )
