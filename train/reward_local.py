"""
Local AI-Detection Reward Function
====================================
Uses a frozen DeBERTa-v3-Large classifier (desklib/ai-text-detector-v1.01)
to score text. Runs on the same GPU as training — no network, no captcha.

RAID benchmark #1 (Jan 2025) — trained on GPT-4, Claude, Gemini, Llama, Mistral etc.

Reward = P(human)  (1.0 = fully human, 0.0 = fully AI)
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# RAID benchmark #1 (Jan 2025) — DeBERTa-v3-Large trained on diverse modern LLMs.
# Checkpoint uses "model.*" key prefix and num_labels=1 (sigmoid) — loaded manually below.
MODEL_ID = "desklib/ai-text-detector-v1.01"
MIN_CHARS = 100
MAX_CHARS = 10000


@dataclass
class ScoreResult:
    text_preview: str
    ai_probability: float
    reward: float
    raw: Dict[str, Any]
    error: Optional[str] = None


def _prepare_text(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        last_space = text.rfind(" ")
        text = text[:last_space] if last_space > 0 else text
    return text


class LocalDetectorWorker:
    """
    Wraps a frozen DeBERTa-v3-Large AI-detector (desklib/ai-text-detector-v1.01).
    Loads once on start(); scores are synchronous CPU/GPU inference.
    """

    def __init__(self, worker_id: int = 0, device: str = "auto"):
        self.worker_id = worker_id
        self.device = device
        self._pipe = None

    def start_sync(self):
        import torch
        import torch.nn as nn
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, AutoConfig, DebertaV2Model, DebertaV2PreTrainedModel

        device_idx = 0 if (self.device == "auto" and torch.cuda.is_available()) else -1
        logger.info(f"[Detector] Loading {MODEL_ID} on device={device_idx}")

        # desklib checkpoint: encoder stored under "model.*", classifier at top level,
        # NO pooler — classifies directly from [CLS] hidden state via Linear(1024, 1).
        class _DesklibDetector(DebertaV2PreTrainedModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.deberta = DebertaV2Model(cfg)
                self.classifier = nn.Linear(cfg.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                out = self.deberta(input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]  # [CLS] token, shape (batch, 1024)
                return self.classifier(cls)            # (batch, 1)

        config = AutoConfig.from_pretrained(MODEL_ID)
        model = _DesklibDetector(config)

        # Remap "model.*" → "deberta.*"; classifier.* stays as-is
        ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
        raw_sd = safetensors.torch.load_file(ckpt_path)
        remapped = {
            ("deberta." + k[len("model."):] if k.startswith("model.") else k): v
            for k, v in raw_sd.items()
        }
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning(f"[Detector] Missing keys: {missing[:5]}")
        if unexpected:
            logger.warning(f"[Detector] Unexpected keys: {unexpected[:5]}")

        if device_idx >= 0:
            model = model.cuda(device_idx)
        model.eval()

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._device_idx = device_idx
        self._torch = torch
        logger.info(f"[Detector] Model loaded")

    def _infer(self, text: str) -> float:
        """Synchronous inference. Returns sigmoid score (higher = more AI)."""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        if self._device_idx >= 0:
            inputs = {k: v.cuda(self._device_idx) for k, v in inputs.items()}
        with self._torch.no_grad():
            logit = self._model(**inputs)          # (1, 1)
        return float(self._torch.sigmoid(logit[0, 0]).cpu())

    async def start(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.start_sync)

    async def stop(self):
        self._model = None

    async def score(self, text: str) -> ScoreResult:
        text = _prepare_text(text)
        preview = text[:80].replace("\n", " ")

        if len(text) < MIN_CHARS:
            return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                               raw={"error": "too_short"}, error="too_short")

        loop = asyncio.get_event_loop()
        try:
            # sigmoid output: higher = more AI-like
            ai_prob = await loop.run_in_executor(None, self._infer, text)
            ai_prob = max(0.0, min(1.0, ai_prob))
            reward = 1.0 - ai_prob
            logger.info(f"[Detector] AI={ai_prob:.2%} reward={reward:.3f} | {preview!r}")
            return ScoreResult(text_preview=preview, ai_probability=ai_prob,
                               reward=reward, raw={"sigmoid": ai_prob})
        except Exception as e:
            logger.error(f"[Detector] Inference failed: {e}")
            return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                               raw={"error": str(e)}, error=str(e))


class MockRewardWorker:
    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id

    async def start(self): pass
    async def stop(self): pass

    async def score(self, text: str) -> ScoreResult:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        ai_prob = max(0.0, min(1.0, random.betavariate(5, 2)))
        return ScoreResult(text_preview=text[:80], ai_probability=ai_prob,
                           reward=1.0 - ai_prob, raw={"mock": True})


class LocalRewardPool:
    """Single detector shared across all scoring calls (no pool needed — it's local)."""

    def __init__(self, mock_mode: bool = False, device: str = "auto", **kwargs):
        self.mock_mode = mock_mode
        if mock_mode:
            self._worker = MockRewardWorker(0)
        else:
            self._worker = LocalDetectorWorker(device=device)

    async def __aenter__(self):
        await self._worker.start()
        return self

    async def __aexit__(self, *args):
        await self._worker.stop()

    async def score_batch(self, texts: List[str]) -> List[ScoreResult]:
        return [await self._worker.score(t) for t in texts]


def score_texts_sync(texts, mock_mode=False, **kwargs):
    async def _run():
        async with LocalRewardPool(mock_mode=mock_mode, **kwargs) as pool:
            return await pool.score_batch(texts)
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    texts = [
        "Artificial intelligence has significantly transformed various industries "
        "and continues to reshape how organizations operate and deliver value to "
        "their customers around the world today in many ways.",
        "i think ai is pretty cool honestly it changed a lot of stuff we do every "
        "day like my phone autocorrects way better than it used to and stuff like that.",
    ]
    mock = "--mock" in sys.argv
    results = score_texts_sync(texts, mock_mode=mock)
    for r in results:
        print(f"AI={r.ai_probability:.2%} reward={r.reward:.3f} | {r.text_preview!r}")
