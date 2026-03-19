"""
Copyleaks Reward Function
==========================
Calls the Copyleaks AI-scan API from within a Playwright browser context.
Direct HTTP calls from datacenter IPs (RunPod) get 403'd — using the browser
as a proxy routes requests through a residential-looking browser session.

Strategy:
  1. Open one persistent browser page at app.copyleaks.com/v1/scan/ai/embedded
  2. For each text, call fetch() via page.evaluate() (same-origin, browser cookies)
  3. Parse summary.ai from the JSON response

Endpoint (discovered by intercepting browser network traffic):
  POST /api/v2/dashboard/anonymous/ai-scan/submit/text
  Body: {"text": "<string, min 150 chars>"}
  Response: {"summary": {"ai": 0.95, "human": 0.05}, ...}

Reward = summary.human  (1.0 = fully human, 0.0 = fully AI)
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

EMBEDDED_URL = "https://app.copyleaks.com/v1/scan/ai/embedded"
API_PATH = "/api/v2/dashboard/anonymous/ai-scan/submit/text"
MIN_CHARS = 150
MAX_CHARS = 25000


@dataclass
class ScoreResult:
    text_preview: str
    ai_probability: float   # 0.0 = human, 1.0 = AI
    reward: float           # 1.0 - ai_probability
    raw: Dict[str, Any]
    error: Optional[str] = None


def _prepare_text(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        last_space = text.rfind(" ")
        text = text[:last_space] if last_space > 0 else text
    return text


class CopyleaksWorker:
    """
    One Playwright browser page kept open at the Copyleaks embedded detector.
    Scores texts by calling fetch() from within the page (bypasses IP blocks).
    """

    def __init__(
        self,
        worker_id: int,
        rate_limit: float = 4.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        headless: bool = True,
        proxy: Optional[str] = None,
    ):
        self.worker_id = worker_id
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.headless = headless
        self.proxy = proxy
        self._browser = None
        self._page = None
        self._playwright = None
        self._last_request_at = 0.0
        self._acdi_token = None

    async def start(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("Run: pip install playwright && playwright install chromium")

        self._playwright = await async_playwright().start()

        launch_kwargs = dict(
            headless=self.headless,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-blink-features=AutomationControlled"],
        )
        if self.proxy:
            launch_kwargs["proxy"] = {"server": self.proxy}

        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        self._page = await context.new_page()

        # Navigate once to establish session/cookies and fetch acdiToken
        await self._page.goto(EMBEDDED_URL, wait_until="domcontentloaded",
                              timeout=int(self.timeout * 1000))

        # Wait for Angular to initialize and store the acdiToken in localStorage
        await asyncio.sleep(3.0)
        self._acdi_token = await self._page.evaluate(
            "const raw = localStorage.getItem('AI_ACDI'); raw ? atob(raw) : null"
        )
        if self._acdi_token:
            logger.info(f"[Worker {self.worker_id}] acdiToken acquired (len={len(self._acdi_token)})")
        else:
            logger.warning(f"[Worker {self.worker_id}] acdiToken not found in localStorage")
        logger.info(f"[Worker {self.worker_id}] Browser ready at {EMBEDDED_URL}")

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _wait_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_at
        wait = self.rate_limit - elapsed
        if wait > 0:
            await asyncio.sleep(wait + random.uniform(0.2, 1.0))
        self._last_request_at = time.monotonic()

    async def score(self, text: str) -> ScoreResult:
        text = _prepare_text(text)
        preview = text[:80].replace("\n", " ")

        if len(text) < MIN_CHARS:
            logger.warning(f"[Worker {self.worker_id}] Text too short ({len(text)} chars)")
            return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                               raw={"error": "too_short"}, error="too_short")

        for attempt in range(self.max_retries):
            try:
                await self._wait_rate_limit()
                result = await self._fetch_via_browser(text, preview)
                return result
            except Exception as e:
                logger.warning(
                    f"[Worker {self.worker_id}] Attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    # Reload the page in case the session expired
                    try:
                        await self._page.reload(wait_until="domcontentloaded",
                                                timeout=int(self.timeout * 1000))
                        await self._refresh_acdi_token()
                    except Exception:
                        pass
                    await asyncio.sleep(5.0 * (attempt + 1))

        logger.error(f"[Worker {self.worker_id}] All retries failed: {preview!r}")
        return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                           raw={"error": "all_retries_failed"}, error="all_retries_failed")

    async def _refresh_acdi_token(self):
        """Re-read acdiToken from localStorage (after page reload)."""
        await asyncio.sleep(3.0)
        self._acdi_token = await self._page.evaluate(
            "const raw = localStorage.getItem('AI_ACDI'); raw ? atob(raw) : null"
        )

    async def _fetch_via_browser(self, text: str, preview: str) -> ScoreResult:
        """Call the Copyleaks API via fetch() executed inside the browser page."""
        body_obj = {"text": text}
        if self._acdi_token:
            body_obj["acdiToken"] = self._acdi_token
        payload = json.dumps(body_obj)

        # Execute fetch from within the page — same origin, browser cookies, real IP
        result = await self._page.evaluate(
            """async ([apiPath, payload]) => {
                const resp = await fetch(apiPath, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/plain, */*',
                    },
                    body: payload,
                });
                const text = await resp.text();
                return { status: resp.status, body: text };
            }""",
            [API_PATH, payload],
        )

        status = result["status"]
        body = result["body"]

        if status == 429:
            raise RuntimeError("rate_limited (429)")
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {body[:200]}")

        data = json.loads(body)
        summary = data.get("summary", {})
        ai_prob = float(summary.get("ai", 0.5))
        ai_prob = max(0.0, min(1.0, ai_prob))
        reward = 1.0 - ai_prob

        logger.info(
            f"[Worker {self.worker_id}] AI={ai_prob:.2%} → reward={reward:.3f} "
            f"| {preview!r}"
        )
        return ScoreResult(text_preview=preview, ai_probability=ai_prob,
                           reward=reward, raw=data)


# ─────────────────────────────────────────────────────────────────────────────
# Mock worker
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Pool
# ─────────────────────────────────────────────────────────────────────────────

def _load_proxies(proxy_file: str) -> list:
    import os
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


class CopyleaksRewardPool:
    """
    Manages N browser workers. Each keeps a persistent page open and
    calls the Copyleaks API via in-browser fetch().
    """

    def __init__(
        self,
        num_workers: int = 2,
        rate_limit: float = 4.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        mock_mode: bool = False,
        headless: bool = True,
        proxy_file: str = None,
    ):
        self.mock_mode = mock_mode
        if mock_mode:
            self._workers = [MockRewardWorker(i) for i in range(num_workers)]
        else:
            proxies = _load_proxies(proxy_file) if proxy_file else []
            if proxies:
                logger.info(f"Loaded {len(proxies)} proxies for {num_workers} workers")
            self._workers = [
                CopyleaksWorker(
                    worker_id=i,
                    rate_limit=rate_limit,
                    timeout=timeout,
                    max_retries=max_retries,
                    headless=headless,
                    proxy=proxies[i % len(proxies)] if proxies else None,
                )
                for i in range(num_workers)
            ]

    async def __aenter__(self):
        await asyncio.gather(*[w.start() for w in self._workers])
        return self

    async def __aexit__(self, *args):
        await asyncio.gather(*[w.stop() for w in self._workers])

    async def score_batch(self, texts: List[str]) -> List["ScoreResult"]:
        assignments = [[] for _ in self._workers]
        for i, text in enumerate(texts):
            assignments[i % len(self._workers)].append(text)

        async def worker_task(worker, worker_texts):
            results = []
            for t in worker_texts:
                results.append(await worker.score(t))
            return results

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


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper + CLI test
# ─────────────────────────────────────────────────────────────────────────────

def score_texts_sync(texts, num_workers=2, mock_mode=False, **kwargs):
    async def _run():
        async with CopyleaksRewardPool(num_workers=num_workers,
                                       mock_mode=mock_mode, **kwargs) as pool:
            return await pool.score_batch(texts)
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    texts = [
        "Artificial intelligence has significantly transformed various industries "
        "and continues to reshape how organizations operate and deliver value to "
        "their customers around the world today in many ways.",
        "I think AI is pretty cool honestly. It changed a lot of stuff we do "
        "every single day. Like my phone autocorrects way better than before.",
    ]
    mock = "--mock" in sys.argv
    results = score_texts_sync(texts, mock_mode=mock)
    for r in results:
        print(f"AI={r.ai_probability:.2%} reward={r.reward:.3f} | {r.text_preview!r}")
