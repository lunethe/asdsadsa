"""
Copyleaks Reward Function
==========================
Calls Copyleaks' anonymous AI-scan API directly — no browser needed.

Discovered endpoint:
  POST https://app.copyleaks.com/api/v2/dashboard/anonymous/ai-scan/submit/text
  Body: {"text": "<string, min 150 chars>"}
  Response: {"summary": {"ai": 0.95, "human": 0.05}, ...}

Reward = summary.human  (1.0 = fully human, 0.0 = fully AI)
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

API_URL = "https://app.copyleaks.com/api/v2/dashboard/anonymous/ai-scan/submit/text"
HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://app.copyleaks.com/v1/scan/ai/embedded",
    "Origin": "https://app.copyleaks.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

MIN_CHARS = 150   # Copyleaks rejects shorter texts
MAX_CHARS = 25000  # Free tier limit shown in UI


@dataclass
class ScoreResult:
    text_preview: str       # First 80 chars for logging
    ai_probability: float   # 0.0 = human, 1.0 = AI
    reward: float           # = 1.0 - ai_probability (maximize this)
    raw: Dict[str, Any]
    error: Optional[str] = None


def _prepare_text(text: str) -> str:
    """Ensure text meets API requirements."""
    text = text.strip()
    if len(text) > MAX_CHARS:
        # Cut at last word boundary
        text = text[:MAX_CHARS]
        text = text[: text.rfind(" ")] if " " in text else text
    return text


class CopyleaksAPIWorker:
    """
    Scores texts by calling the Copyleaks anonymous API directly.
    Uses httpx for async HTTP — no browser, no disk, fast.
    """

    def __init__(
        self,
        worker_id: int,
        rate_limit: float = 4.0,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.worker_id = worker_id
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._last_request_at = 0.0

    async def start(self):
        try:
            import httpx
        except ImportError:
            raise ImportError("Run: pip install httpx")
        self._client = httpx.AsyncClient(
            headers=HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
        )
        logger.info(f"[Worker {self.worker_id}] HTTP client ready")

    async def stop(self):
        if self._client:
            await self._client.aclose()

    async def _wait_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_at
        wait = self.rate_limit - elapsed
        if wait > 0:
            await asyncio.sleep(wait + random.uniform(0.2, 1.0))
        self._last_request_at = time.monotonic()

    async def score(self, text: str) -> ScoreResult:
        text = _prepare_text(text)
        preview = text[:80].replace("\n", " ")

        # Pad short texts to meet minimum length
        if len(text) < MIN_CHARS:
            logger.warning(
                f"[Worker {self.worker_id}] Text too short ({len(text)} chars), "
                f"returning neutral reward."
            )
            return ScoreResult(
                text_preview=preview,
                ai_probability=0.5,
                reward=0.5,
                raw={"error": "too_short"},
                error="too_short",
            )

        for attempt in range(self.max_retries):
            try:
                await self._wait_rate_limit()
                result = await self._call_api(text, preview)
                return result
            except Exception as e:
                logger.warning(
                    f"[Worker {self.worker_id}] Attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5.0 * (attempt + 1))

        logger.error(f"[Worker {self.worker_id}] All retries failed: {preview!r}")
        return ScoreResult(
            text_preview=preview,
            ai_probability=0.5,
            reward=0.5,
            raw={"error": "all_retries_failed"},
            error="all_retries_failed",
        )

    async def _call_api(self, text: str, preview: str) -> ScoreResult:
        import json as _json

        payload = _json.dumps({"text": text})
        response = await self._client.post(API_URL, content=payload)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "10"))
            logger.warning(
                f"[Worker {self.worker_id}] Rate limited, sleeping {retry_after}s"
            )
            await asyncio.sleep(retry_after)
            raise RuntimeError("rate_limited")

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        summary = data.get("summary", {})

        ai_prob = float(summary.get("ai", 0.5))
        ai_prob = max(0.0, min(1.0, ai_prob))
        reward = 1.0 - ai_prob

        logger.info(
            f"[Worker {self.worker_id}] AI={ai_prob:.2%} → reward={reward:.3f} "
            f"| {preview!r}"
        )
        return ScoreResult(
            text_preview=preview,
            ai_probability=ai_prob,
            reward=reward,
            raw=data,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock worker (for offline testing)
# ─────────────────────────────────────────────────────────────────────────────

class MockRewardWorker:
    """Random scores skewed toward AI-detected. Use MOCK_REWARDS=true."""

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id

    async def start(self):
        pass

    async def stop(self):
        pass

    async def score(self, text: str) -> ScoreResult:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        ai_prob = random.betavariate(5, 2)  # mean ~0.71
        ai_prob = max(0.0, min(1.0, ai_prob))
        return ScoreResult(
            text_preview=text[:80],
            ai_probability=ai_prob,
            reward=1.0 - ai_prob,
            raw={"mock": True},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pool: N workers scoring concurrently
# ─────────────────────────────────────────────────────────────────────────────

class CopyleaksRewardPool:
    """
    Manages N async HTTP workers for parallel scoring.

    Usage:
        async with CopyleaksRewardPool(num_workers=3) as pool:
            results = await pool.score_batch(texts)
    """

    def __init__(
        self,
        num_workers: int = 3,
        rate_limit: float = 4.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        mock_mode: bool = False,
        # Kept for API compatibility with old code, no longer used:
        headless: bool = True,
    ):
        self.mock_mode = mock_mode
        if mock_mode:
            self._workers = [MockRewardWorker(i) for i in range(num_workers)]
        else:
            self._workers = [
                CopyleaksAPIWorker(
                    worker_id=i,
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
        """
        Score all texts using the worker pool (round-robin assignment).
        Each worker processes its share sequentially to respect rate limits.
        """
        assignments: List[List[str]] = [[] for _ in self._workers]
        for i, text in enumerate(texts):
            assignments[i % len(self._workers)].append(text)

        async def worker_task(worker, worker_texts):
            results = []
            for t in worker_texts:
                r = await worker.score(t)
                results.append(r)
            return results

        worker_results = await asyncio.gather(*[
            worker_task(w, assignments[i])
            for i, w in enumerate(self._workers)
        ])

        # Reassemble in original order
        ordered = [None] * len(texts)
        counters = [0] * len(self._workers)
        for i in range(len(texts)):
            w_idx = i % len(self._workers)
            ordered[i] = worker_results[w_idx][counters[w_idx]]
            counters[w_idx] += 1

        return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper
# ─────────────────────────────────────────────────────────────────────────────

def score_texts_sync(
    texts: List[str],
    num_workers: int = 3,
    mock_mode: bool = False,
    **kwargs,
) -> List[ScoreResult]:
    async def _run():
        async with CopyleaksRewardPool(
            num_workers=num_workers,
            mock_mode=mock_mode,
            **kwargs,
        ) as pool:
            return await pool.score_batch(texts)
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    texts = [
        "Artificial intelligence has significantly transformed various industries "
        "and continues to reshape how organizations operate and deliver value to "
        "their customers and stakeholders around the world today.",
        "I think AI is pretty cool. It changed a lot of stuff we do every day, "
        "honestly. Like, my phone now autocorrects way better than it used to, "
        "and that alone saves me from so many typos.",
    ]

    mock = "--mock" in sys.argv
    print(f"Scoring {len(texts)} texts ({'mock' if mock else 'live Copyleaks API'})...")
    results = score_texts_sync(texts, mock_mode=mock)
    for r in results:
        print(f"  AI={r.ai_probability:.2%}  Reward={r.reward:.3f}  | {r.text_preview!r}")
