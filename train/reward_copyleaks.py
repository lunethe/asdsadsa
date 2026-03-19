"""
Copyleaks Reward Function
==========================
Scores text using Copyleaks' free AI content detector.
Returns a reward in [0, 1] where 1.0 = classified as fully human.

Strategy:
  1. Launch headless Chromium via Playwright
  2. Navigate to copyleaks.com/ai-content-detector
  3. Intercept the internal API XHR/fetch response to capture the score
     (faster and more reliable than DOM scraping)
  4. If interception misses, fall back to DOM parsing
  5. Rate-limit requests to avoid getting blocked
  6. Pool N browser contexts for parallel scoring

Usage:
    async with CopyleaksRewardPool(num_workers=2) as pool:
        rewards = await pool.score_batch(["text1", "text2", ...])
"""

import asyncio
import re
import random
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

COPYLEAKS_URL = "https://copyleaks.com/ai-content-detector"

# Copyleaks free tier truncates at ~3000 chars; we truncate slightly under
FREE_TIER_CHAR_LIMIT = 2800


@dataclass
class ScoreResult:
    text_preview: str       # First 80 chars for logging
    ai_probability: float   # 0.0 = human, 1.0 = AI
    reward: float           # = 1.0 - ai_probability (maximize this)
    raw: Dict[str, Any]     # Raw response data
    error: Optional[str] = None


def _truncate(text: str) -> str:
    """Truncate to free-tier limit, cutting at a word boundary."""
    if len(text) <= FREE_TIER_CHAR_LIMIT:
        return text
    truncated = text[:FREE_TIER_CHAR_LIMIT]
    # Cut at last space to avoid mid-word truncation
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


class CopyleaksWorker:
    """
    A single Playwright browser instance that scores texts sequentially.
    Multiple workers run in parallel for higher throughput.
    """

    def __init__(
        self,
        worker_id: int,
        rate_limit: float,
        timeout: float,
        headless: bool,
        max_retries: int,
    ):
        self.worker_id = worker_id
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.headless = headless
        self.max_retries = max_retries
        self._browser = None
        self._playwright = None
        self._last_request_at = 0.0

    async def start(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("Run: pip install playwright && playwright install chromium")

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        logger.info(f"[Worker {self.worker_id}] Browser started")

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _wait_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_at
        wait = self.rate_limit - elapsed
        if wait > 0:
            # Add jitter to avoid synchronized bursts across workers
            await asyncio.sleep(wait + random.uniform(0.3, 1.2))
        self._last_request_at = time.monotonic()

    async def score(self, text: str) -> ScoreResult:
        text = _truncate(text)
        preview = text[:80].replace("\n", " ")

        for attempt in range(self.max_retries):
            try:
                await self._wait_rate_limit()
                result = await self._do_score(text, preview)
                return result
            except Exception as e:
                logger.warning(
                    f"[Worker {self.worker_id}] Attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5.0 * (attempt + 1))

        logger.error(f"[Worker {self.worker_id}] All retries failed for: {preview!r}")
        return ScoreResult(
            text_preview=preview,
            ai_probability=0.5,
            reward=0.5,
            raw={"error": "all_retries_failed"},
            error="all_retries_failed",
        )

    async def _do_score(self, text: str, preview: str) -> ScoreResult:
        """Core scraping logic: open page, submit text, capture score."""
        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()
        captured: Dict[str, Any] = {}

        # ── Intercept the internal API response ──────────────────────────
        # Copyleaks' SPA makes a POST/GET to their internal API.
        # We listen for any JSON response containing "ai" or "score" fields.
        async def on_response(response):
            url = response.url
            # Match their AI-detector API calls (endpoint may vary with deploys)
            if response.status == 200 and any(k in url for k in [
                "ai-content", "ai_content", "writer-detector",
                "check-ai", "detect", "scan",
            ]):
                try:
                    data = await response.json()
                    _extract_score_from_json(data, captured)
                except Exception:
                    pass

        page.on("response", on_response)

        try:
            await page.goto(COPYLEAKS_URL, timeout=int(self.timeout * 1000),
                            wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle",
                                           timeout=int(self.timeout * 1000))

            # ── Find the text input ───────────────────────────────────────
            textarea = await _find_element(page, [
                "textarea",
                "[contenteditable='true']",
                "[data-testid='text-input']",
                "[placeholder*='paste']",
                "[placeholder*='text']",
                "[class*='TextArea']",
                "[class*='textarea']",
            ])
            if textarea is None:
                raise RuntimeError("Could not locate text input on Copyleaks page")

            await textarea.click()
            # Use fill() for textarea; for contenteditable use type()
            tag = await textarea.evaluate("el => el.tagName.toLowerCase()")
            if tag == "textarea":
                await textarea.fill(text)
            else:
                await textarea.evaluate(
                    "el => { el.innerText = ''; }",
                )
                await textarea.type(text, delay=2)

            # ── Click the submit / Check button ───────────────────────────
            submit = await _find_element(page, [
                "button[type='submit']",
                "[data-testid='check-button']",
                "[data-testid='detect-button']",
                "button:has-text('Check')",
                "button:has-text('Detect')",
                "button:has-text('Scan')",
                "button:has-text('Analyze')",
            ])
            if submit is None:
                raise RuntimeError("Could not locate submit button on Copyleaks page")

            await submit.click()

            # ── Wait up to timeout for the API response ───────────────────
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await page.wait_for_timeout(1000)
                if "ai_score" in captured:
                    break
                # Also try DOM fallback while waiting
                dom_score = await _parse_score_from_dom(page)
                if dom_score is not None:
                    captured["ai_score"] = dom_score
                    captured["method"] = "dom"
                    break

            if "ai_score" not in captured:
                raise RuntimeError(
                    "Timed out waiting for Copyleaks score. "
                    "Page content snippet: "
                    + (await page.inner_text("body"))[:300]
                )

            ai_prob = float(captured["ai_score"])
            ai_prob = max(0.0, min(1.0, ai_prob))
            reward = 1.0 - ai_prob

            logger.info(
                f"[Worker {self.worker_id}] Score: {ai_prob:.2%} AI → "
                f"reward={reward:.3f} | {preview!r}"
            )
            return ScoreResult(
                text_preview=preview,
                ai_probability=ai_prob,
                reward=reward,
                raw=dict(captured),
            )
        finally:
            await context.close()


def _extract_score_from_json(data: Any, out: Dict):
    """
    Recursively search a JSON blob for the AI probability field.
    Copyleaks has changed their response schema over time; we handle
    several known shapes.
    """
    if not isinstance(data, dict):
        if isinstance(data, list):
            for item in data:
                _extract_score_from_json(item, out)
        return

    # Known field names (as a fraction 0-1 or percentage 0-100)
    for key in ("ai", "aiScore", "ai_score", "aiProbability", "ai_probability",
                "score", "probability"):
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                # Normalize to 0-1
                normalized = val / 100.0 if val > 1.0 else float(val)
                out["ai_score"] = normalized
                out["method"] = f"api_field:{key}"
                out["raw_value"] = val
                return

    # Try nested "summary" or "result" objects
    for key in ("summary", "result", "data", "detection", "analysis"):
        if key in data:
            _extract_score_from_json(data[key], out)
            if "ai_score" in out:
                return

    # Recurse all values
    for v in data.values():
        if isinstance(v, (dict, list)):
            _extract_score_from_json(v, out)
            if "ai_score" in out:
                return


async def _parse_score_from_dom(page) -> Optional[float]:
    """
    Fallback: scrape the displayed percentage from the rendered DOM.
    Looks for things like "95% AI" or "5% Human".
    """
    try:
        body_text = await page.inner_text("body")
    except Exception:
        return None

    # Pattern: "97% AI content" or "3% Human"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*%\s*(?:AI|artificial)",
        r"AI\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:generated|written by AI)",
    ]
    for pat in patterns:
        m = re.search(pat, body_text, re.IGNORECASE)
        if m:
            return float(m.group(1)) / 100.0

    # Pattern for human percentage (invert)
    human_patterns = [
        r"(\d+(?:\.\d+)?)\s*%\s*Human",
        r"Human\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%",
    ]
    for pat in human_patterns:
        m = re.search(pat, body_text, re.IGNORECASE)
        if m:
            return 1.0 - float(m.group(1)) / 100.0

    # Try to find a large number followed by % near result keywords
    result_section = re.search(
        r"(?:result|score|detection).*?(\d{1,3})\s*%",
        body_text,
        re.IGNORECASE | re.DOTALL,
    )
    if result_section:
        val = float(result_section.group(1))
        # Ambiguous: could be AI% or Human% — assume AI%
        return val / 100.0

    return None


async def _find_element(page, selectors: List[str]):
    """Try selectors in order, return first match."""
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if await el.count() > 0:
                return el
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Mock reward (for offline dev / unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class MockRewardWorker:
    """Returns random scores. Use MOCK_REWARDS=true for local testing."""

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id

    async def start(self):
        pass

    async def stop(self):
        pass

    async def score(self, text: str) -> ScoreResult:
        await asyncio.sleep(random.uniform(0.05, 0.2))  # simulate latency
        # Skew toward AI-detected so there's gradient signal to learn from
        ai_prob = random.betavariate(5, 2)  # mean ~0.71 (hard to fool)
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
    Manages a pool of browser workers and distributes scoring work.

    Usage:
        async with CopyleaksRewardPool(num_workers=2) as pool:
            results = await pool.score_batch(texts)
    """

    def __init__(
        self,
        num_workers: int = 2,
        rate_limit: float = 4.0,
        timeout: float = 45.0,
        headless: bool = True,
        max_retries: int = 3,
        mock_mode: bool = False,
    ):
        self.mock_mode = mock_mode
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers = []

        if mock_mode:
            self._workers = [MockRewardWorker(i) for i in range(num_workers)]
        else:
            self._workers = [
                CopyleaksWorker(
                    worker_id=i,
                    rate_limit=rate_limit,
                    timeout=timeout,
                    headless=headless,
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
        Score all texts using the worker pool.
        Distributes texts round-robin across workers; each worker
        processes its share sequentially (respecting rate limits).
        """
        # Assign each text to a worker slot
        assignments: List[List[str]] = [[] for _ in self._workers]
        for i, text in enumerate(texts):
            assignments[i % len(self._workers)].append(text)

        async def worker_task(worker, texts_for_worker):
            results = []
            for t in texts_for_worker:
                r = await worker.score(t)
                results.append(r)
            return results

        # Run workers concurrently
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
# Convenience: synchronous wrapper for non-async callers
# ─────────────────────────────────────────────────────────────────────────────

def score_texts_sync(
    texts: List[str],
    num_workers: int = 2,
    mock_mode: bool = False,
    **kwargs,
) -> List[ScoreResult]:
    """Synchronous wrapper around CopyleaksRewardPool."""

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
        "Artificial intelligence has significantly transformed various industries.",
        "I think AI is pretty cool and has changed a lot of stuff we do.",
    ]

    mock = "--mock" in sys.argv
    print(f"Scoring {len(texts)} texts ({'mock' if mock else 'live Copyleaks'})...")
    results = score_texts_sync(texts, mock_mode=mock)
    for r in results:
        print(f"  AI: {r.ai_probability:.2%}  Reward: {r.reward:.3f}  "
              f"| {r.text_preview!r}")
