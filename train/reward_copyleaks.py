"""
Copyleaks Reward Function (CapSolver edition)
==============================================
Solves the Cloudflare Turnstile on Copyleaks using CapSolver, then POSTs
directly to the anonymous AI-scan API.  No browser / Playwright needed.

Set env var: CAPSOLVER_API_KEY=your_key_here
Get a key at: https://capsolver.com

Reward = summary.human  (1.0 = fully human, 0.0 = fully AI)
"""

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Suppress noisy httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

import httpx

logger = logging.getLogger(__name__)

CAPSOLVER_URL      = "https://api.capsolver.com"
COPYLEAKS_SCAN_URL = "https://app.copyleaks.com/api/v2/dashboard/anonymous/ai-scan/submit/text"
TURNSTILE_SITEKEY  = "0x4AAAAAAADZUXiboAFN3tU8"
TURNSTILE_PAGE_URL = "https://app.copyleaks.com/v1/scan/ai/embedded"

MIN_CHARS = 350
MAX_CHARS = 25000
DEFAULT_RATE_LIMIT = 1.0   # seconds between requests (proxy rotation handles rate limits)


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


async def _solve_turnstile(client: httpx.AsyncClient, api_key: str) -> str:
    """
    Use CapSolver to solve the Copyleaks Turnstile widget.
    Returns the captchaResponse token string.
    """
    # 1. Create task
    create_resp = await client.post(
        f"{CAPSOLVER_URL}/createTask",
        json={
            "clientKey": api_key,
            "task": {
                "type": "AntiTurnstileTaskProxyLess",
                "websiteURL": TURNSTILE_PAGE_URL,
                "websiteKey": TURNSTILE_SITEKEY,
            },
        },
        timeout=30.0,
    )
    create_resp.raise_for_status()
    create_data = create_resp.json()
    if create_data.get("errorId", 0) != 0:
        raise RuntimeError(f"CapSolver createTask error: {create_data}")
    task_id = create_data["taskId"]

    # 2. Poll for result
    for _ in range(30):
        await asyncio.sleep(3.0)
        result_resp = await client.post(
            f"{CAPSOLVER_URL}/getTaskResult",
            json={"clientKey": api_key, "taskId": task_id},
            timeout=30.0,
        )
        result_resp.raise_for_status()
        result_data = result_resp.json()
        if result_data.get("errorId", 0) != 0:
            raise RuntimeError(f"CapSolver getTaskResult error: {result_data}")
        if result_data.get("status") == "ready":
            token = result_data["solution"]["token"]
            logger.debug(f"[CapSolver] Turnstile token obtained")
            return token

    raise RuntimeError("CapSolver timed out waiting for Turnstile solution")


def _load_proxies(proxy_file: str) -> List[str]:
    if not proxy_file or not os.path.exists(proxy_file):
        return []
    proxies = []
    with open(proxy_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("http"):
                line = f"http://{line}"
            proxies.append(line)
    return proxies


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": "https://app.copyleaks.com",
    "Referer": "https://app.copyleaks.com/v1/scan/ai/embedded",
}


class CopyleaksWorker:
    """
    Scores texts via Copyleaks anonymous AI detector.
    On 429, immediately rotates to the next proxy and retries — no sleeping.
    """

    def __init__(
        self,
        worker_id: int = 0,
        api_key: Optional[str] = None,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        max_retries: int = 3,
        timeout: float = 60.0,
        proxies: Optional[List[str]] = None,
    ):
        self.worker_id = worker_id
        self.api_key = api_key or os.getenv("CAPSOLVER_API_KEY", "")
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.proxies = proxies or []
        self._proxy_idx = worker_id % len(self.proxies) if self.proxies else 0
        self._client: Optional[httpx.AsyncClient] = None
        self._capsolver_client: Optional[httpx.AsyncClient] = None
        self._last_request_at = 0.0

        if not self.api_key:
            raise ValueError("CAPSOLVER_API_KEY not set. Export it or pass api_key=")

    def _current_proxy(self) -> Optional[str]:
        return self.proxies[self._proxy_idx] if self.proxies else None

    def _rotate_proxy(self):
        if self.proxies:
            self._proxy_idx = (self._proxy_idx + 1) % len(self.proxies)
            logger.info(f"[Worker {self.worker_id}] Rotated to proxy {self._current_proxy()}")

    async def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=_HEADERS,
            proxies=self._current_proxy(),
            timeout=self.timeout,
        )

    async def start(self):
        self._client = await self._make_client()
        # CapSolver calls go direct (no proxy needed)
        self._capsolver_client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"[Copyleaks Worker {self.worker_id}] Ready"
                    + (f" via {self._current_proxy()}" if self.proxies else ""))

    async def stop(self):
        if self._client:
            await self._client.aclose()
        if self._capsolver_client:
            await self._capsolver_client.aclose()

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

                token = await _solve_turnstile(self._capsolver_client, self.api_key)

                resp = await self._client.post(
                    COPYLEAKS_SCAN_URL,
                    json={"text": text, "captchaResponse": token, "acdiToken": None},
                )

                if resp.status_code == 429 or (resp.status_code in (403, 503) and
                        any(w in resp.text for w in ("cloudflare", "banned", "blocked", "ray id"))):
                    logger.warning(f"[Worker {self.worker_id}] {resp.status_code} — rotating proxy")
                    self._rotate_proxy()
                    await self._client.aclose()
                    self._client = await self._make_client()
                    continue

                if resp.status_code == 400 and "too-short" in resp.text:
                    return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                                       raw={"error": "too_short"}, error="too_short")

                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                data = resp.json()
                summary = data.get("summary", {})
                ai_prob = float(summary.get("ai", 0.5))
                ai_prob = max(0.0, min(1.0, ai_prob))
                reward = 1.0 - ai_prob

                logger.info(f"[Worker {self.worker_id}] AI={ai_prob:.2%} reward={reward:.3f} | {preview!r}")
                return ScoreResult(text_preview=preview, ai_probability=ai_prob,
                                   reward=reward, raw=data)

            except Exception as e:
                logger.warning(f"[Worker {self.worker_id}] Attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2.0)

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


class CopyleaksRewardPool:
    """
    Pool of workers. Each worker gets the full proxy list and rotates
    through them on 429 — no sleeping, just switch proxy and retry.
    """

    def __init__(
        self,
        num_workers: int = 8,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: float = 60.0,
        max_retries: int = 3,
        mock_mode: bool = False,
        api_key: Optional[str] = None,
        proxy_file: Optional[str] = None,
        **kwargs,
    ):
        self.mock_mode = mock_mode
        if mock_mode:
            self._workers = [MockRewardWorker(i) for i in range(num_workers)]
        else:
            proxies = _load_proxies(proxy_file or os.getenv("PROXY_FILE", ""))
            if proxies:
                logger.info(f"Loaded {len(proxies)} proxies, {num_workers} workers")
            else:
                logger.warning("No proxies loaded — rate limits will hit fast")
            self._workers = [
                CopyleaksWorker(
                    worker_id=i,
                    api_key=api_key,
                    rate_limit=rate_limit,
                    timeout=timeout,
                    max_retries=max_retries,
                    proxies=proxies,
                )
                for i in range(num_workers)
            ]

    async def __aenter__(self):
        await asyncio.gather(*[w.start() for w in self._workers])
        return self

    async def __aexit__(self, *args):
        await asyncio.gather(*[w.stop() for w in self._workers])

    async def _restart_worker(self, idx: int):
        """Restart a dead worker in-place."""
        try:
            await self._workers[idx].stop()
        except Exception:
            pass
        w = self._workers[idx]
        fresh = CopyleaksWorker(
            worker_id=w.worker_id,
            api_key=w.api_key,
            rate_limit=w.rate_limit,
            max_retries=w.max_retries,
            timeout=w.timeout,
            proxies=w.proxies,
        )
        await fresh.start()
        self._workers[idx] = fresh
        logger.info(f"[Pool] Worker {idx} restarted")

    async def score_batch(self, texts: List[str]) -> List[ScoreResult]:
        assignments = [[] for _ in self._workers]
        for i, text in enumerate(texts):
            assignments[i % len(self._workers)].append(text)

        async def worker_task(idx, worker_texts):
            results = []
            for t in worker_texts:
                r = await self._workers[idx].score(t)
                if r.error == "all_retries_failed":
                    logger.warning(f"[Pool] Worker {idx} failed, restarting...")
                    await self._restart_worker(idx)
                    r = await self._workers[idx].score(t)
                results.append(r)
            return results

        worker_results = await asyncio.gather(*[
            worker_task(i, assignments[i]) for i in range(len(self._workers))
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
        async with CopyleaksRewardPool(mock_mode=mock_mode, **kwargs) as pool:
            return await pool.score_batch(texts)
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    texts = [
        "Artificial intelligence has significantly transformed various industries "
        "and continues to reshape how organizations operate and deliver value to "
        "their customers around the world today in many ways.",
        "honestly i just dont get why my cat knocks stuff off the table. like she "
        "looks me dead in the eye and just pushes my cup off. why",
    ]
    mock = "--mock" in sys.argv
    results = score_texts_sync(texts, mock_mode=mock)
    for r in results:
        print(f"AI={r.ai_probability:.2%} reward={r.reward:.3f} | {r.text_preview!r}")
