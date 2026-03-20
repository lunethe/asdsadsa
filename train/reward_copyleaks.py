"""
Copyleaks Reward Function
==========================
Drives the Copyleaks embedded AI detector via Playwright UI automation.

Root cause of previous 403s:
  The API requires a Cloudflare Turnstile captchaResponse in the POST body.
  Direct fetch() calls lack this token. Only the Angular app, after the
  Turnstile widget runs, injects it. Solution: drive the UI (click Scan),
  intercept the response via page.on('response').

Strategy:
  1. Keep one Playwright page open at app.copyleaks.com/v1/scan/ai/embedded
  2. For each text: fill the editor → click Scan → intercept API response
  3. The Turnstile widget runs automatically and injects captchaResponse
  4. Parse summary.ai from the intercepted JSON response

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
API_URL = "https://app.copyleaks.com/api/v2/dashboard/anonymous/ai-scan/submit/text"
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
    One Playwright browser page at the Copyleaks embedded detector.
    Scores texts by driving the Angular UI (fill + click Scan) so that
    Cloudflare Turnstile auto-completes and injects captchaResponse.
    """

    def __init__(
        self,
        worker_id: int,
        rate_limit: float = 4.0,
        max_retries: int = 3,
        timeout: float = 60.0,
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
        self._on_results_page = False

    async def start(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("Run: pip install playwright && playwright install chromium")

        self._playwright = await async_playwright().start()

        launch_kwargs = dict(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
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

        await self._page.goto(EMBEDDED_URL, wait_until="networkidle",
                              timeout=int(self.timeout * 1000))
        # Wait for Angular + Turnstile to initialise
        await asyncio.sleep(3.0)
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

    async def _reset_to_input(self):
        """Navigate back to the input form (click Try Again if on results page)."""
        try:
            try_again = await self._page.query_selector(".try-again-btn")
            if try_again:
                await try_again.click()
                await asyncio.sleep(0.8)
                self._on_results_page = False
        except Exception:
            pass

    async def _fill_editor(self, text: str):
        """Type text into the Angular scan editor."""
        # Click a chip first — chips are always enabled and clicking one
        # focuses the editor component reliably.
        chip = await self._page.query_selector(
            "mat-chip-option, .mdc-evolution-chip__action"
        )
        if chip:
            await chip.click()
            await asyncio.sleep(0.5)

        # Select all content (chip text) and replace with our text
        await self._page.keyboard.press("Control+a")
        await asyncio.sleep(0.1)

        # Set clipboard content via JS, then paste — avoids slow key-by-key typing
        await self._page.evaluate(
            """(txt) => {
                const dt = new DataTransfer();
                dt.setData('text/plain', txt);
                document.dispatchEvent(new ClipboardEvent('paste', {
                    clipboardData: dt, bubbles: true, cancelable: true
                }));
            }""",
            text,
        )
        await asyncio.sleep(0.3)

        # Fallback: if clipboard paste didn't work, force-set via JS
        current_len = await self._page.evaluate(
            "document.querySelector('.scan-text-editor')?.textContent?.length || 0"
        )
        if current_len < 10:
            await self._page.evaluate(
                """(txt) => {
                    const el = document.querySelector('.scan-text-editor');
                    if (el) {
                        el.textContent = txt;
                        el.dispatchEvent(new InputEvent('input', {bubbles: true, data: txt}));
                    }
                }""",
                text,
            )
            await asyncio.sleep(0.3)

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
                result = await self._score_via_ui(text, preview)
                return result
            except Exception as e:
                logger.warning(
                    f"[Worker {self.worker_id}] Attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    try:
                        await self._page.goto(EMBEDDED_URL, wait_until="networkidle",
                                              timeout=int(self.timeout * 1000))
                        await asyncio.sleep(3.0)
                        self._on_results_page = False
                    except Exception:
                        pass
                    await asyncio.sleep(5.0 * (attempt + 1))

        logger.error(f"[Worker {self.worker_id}] All retries failed: {preview!r}")
        return ScoreResult(text_preview=preview, ai_probability=0.5, reward=0.5,
                           raw={"error": "all_retries_failed"}, error="all_retries_failed")

    async def _score_via_ui(self, text: str, preview: str) -> ScoreResult:
        """Drive the Angular UI: fill text → click Scan → intercept response."""
        response_holder: Dict[str, Any] = {}
        response_event = asyncio.Event()

        async def on_response(response):
            if API_URL in response.url and "status" not in response_holder:
                try:
                    body = await response.text()
                    response_holder["status"] = response.status
                    response_holder["body"] = body
                    response_event.set()
                except Exception:
                    pass

        self._page.on("response", on_response)
        try:
            # Return to input form if we're on the results page
            await self._reset_to_input()

            # Fill the text editor
            await self._fill_editor(text)

            # Find and click the Scan button
            # The "Scan" button is the only mdc-button--raised on the input page
            scan_btn = await self._page.query_selector("button.mdc-button--raised")
            if not scan_btn:
                raise RuntimeError("Scan button not found")

            await scan_btn.click()
            self._on_results_page = True

            # Wait for the API response
            try:
                await asyncio.wait_for(response_event.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(f"Timeout ({self.timeout}s) waiting for scan response")

        finally:
            self._page.remove_listener("response", on_response)

        status = response_holder["status"]
        body = response_holder["body"]

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
    drives the Copyleaks UI to score texts (Turnstile-safe).
    """

    def __init__(
        self,
        num_workers: int = 2,
        rate_limit: float = 4.0,
        timeout: float = 60.0,
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
