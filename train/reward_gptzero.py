"""
GPTZero Reward Function
========================
Uses the GPTZero API to score text for AI detection.
GPTZero is one of the most widely-used commercial detectors and
generalizes well as a training signal.

Set env var: GPTZERO_API_KEY=your_key_here
Get a key at: https://gptzero.me/api

Reward = 1 - completely_generated_prob  (1.0 = human, 0.0 = AI)
"""

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)

GPTZERO_URL = "https://api.gptzero.me/v2/predict/text"
MIN_CHARS = 150
MAX_CHARS = 50000
# GPTZero free tier: 150 req/day. Paid tiers have higher limits.
DEFAULT_RATE_LIMIT = 0.5  # seconds between requests (2 req/sec max on paid)


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


class GPTZeroWorker:
    def __init__(
        self,
        worker_id: int = 0,
        api_key: Optional[str] = None,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.worker_id = worker_id
        self.api_key = api_key or os.getenv("GPTZERO_API_KEY", "")
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_at = 0.0

        if not self.api_key:
            raise ValueError("GPTZERO_API_KEY not set. Export it or pass api_key=")

    async def start(self):
        self._client = httpx.AsyncClient(
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        logger.info(f"[GPTZero Worker {self.worker_id}] Ready")

    async def stop(self):
        if self._client:
            await self._client.aclose()

    async def _wait_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_at
        wait = self.rate_limit - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    async def score(self, text: str) -> ScoreResult:
        text = _prepare_text(text)
        preview = text[:80].replace("\n", " ")

        if len(text) < MIN_CHARS:
            return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                               raw={"error": "too_short"}, error="too_short")

        for attempt in range(self.max_retries):
            try:
                await self._wait_rate_limit()
                resp = await self._client.post(
                    GPTZERO_URL,
                    json={"document": text},
                )

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"[GPTZero] Rate limited, sleeping {wait}s")
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                data = resp.json()
                doc = data.get("documents", [{}])[0]
                ai_prob = float(doc.get("completely_generated_prob", 0.5))
                ai_prob = max(0.0, min(1.0, ai_prob))
                reward = 1.0 - ai_prob

                logger.info(
                    f"[GPTZero Worker {self.worker_id}] AI={ai_prob:.2%} "
                    f"reward={reward:.3f} | {preview!r}"
                )
                return ScoreResult(text_preview=preview, ai_probability=ai_prob,
                                   reward=reward, raw=doc)

            except Exception as e:
                logger.warning(f"[GPTZero Worker {self.worker_id}] Attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5.0 * (attempt + 1))

        return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                           raw={"error": "all_retries_failed"}, error="all_retries_failed")


class MockRewardWorker:
    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id

    async def start(self): pass
    async def stop(self): pass

    async def score(self, text: str) -> ScoreResult:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        ai_prob = max(0.0, min(1.0, random.betavariate(5, 2)))
        return ScoreResult(text_preview=text[:80], ai_probability=ai_prob,
                           reward=1.0 - ai_prob, raw={"mock": True})


class GPTZeroRewardPool:
    """
    Pool of GPTZero workers for parallel scoring.
    Each worker has its own httpx client and rate limiter.
    """

    def __init__(
        self,
        num_workers: int = 2,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: float = 30.0,
        max_retries: int = 3,
        mock_mode: bool = False,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.mock_mode = mock_mode
        if mock_mode:
            self._workers = [MockRewardWorker(i) for i in range(num_workers)]
        else:
            self._workers = [
                GPTZeroWorker(
                    worker_id=i,
                    api_key=api_key,
                    rate_limit=rate_limit,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                for i in range(num_workers)
            ]

    async def __aenter__(self):
        await asyncio.gather(*[w.start() for w in self._workers])
        return self

    async def __aexit__(self, *args):
        await asyncio.gather(*[w.stop() for w in self._workers])

    async def score_batch(self, texts: List[str]) -> List[ScoreResult]:
        assignments = [[] for _ in self._workers]
        for i, text in enumerate(texts):
            assignments[i % len(self._workers)].append(text)

        async def worker_task(worker, worker_texts):
            return [await worker.score(t) for t in worker_texts]

        worker_results = await asyncio.gather(*[
            worker_task(w, assignments[i]) for i, w in enumerate(self._workers)
        ])

        ordered = [None] * len(texts)
        counters = [0] * len(self._workers)
        for i in range(len(texts)):
            w_idx = i % len(self._workers)
            ordered[i] = worker_results[w_idx][counters[w_idx]]
            counters[w_idx] += 1
        return ordered


def score_texts_sync(texts, mock_mode=False, **kwargs):
    async def _run():
        async with GPTZeroRewardPool(mock_mode=mock_mode, **kwargs) as pool:
            return await pool.score_batch(texts)
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
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
