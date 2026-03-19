"""
AI Text Humanizer - FastAPI Server
===================================
Uses DIPPER (11B T5-XXL) for paraphrasing + programmatic post-processing.
Falls back to lightweight model if full DIPPER can't load.
"""

import os
import re
import time
import random
import logging
from typing import Optional

import torch
import nltk
from nltk.tokenize import sent_tokenize
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_MODE = os.getenv("HUMANIZER_MODEL", "full")  # "full" or "lightweight"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AI Text Humanizer", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
class DipperParaphraser:
    """DIPPER: 11B T5-XXL paraphraser trained on human literary translations."""

    def __init__(self, model_name: str, tokenizer_name: str, device: str):
        logger.info(f"Loading model: {model_name} on {device}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cuda" and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def paraphrase(
        self,
        text: str,
        lex_diversity: int = 60,
        order_diversity: int = 60,
        sent_interval: int = 3,
        prefix: str = "",
    ) -> str:
        assert lex_diversity in range(0, 101, 20)
        assert order_diversity in range(0, 101, 20)

        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity

        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            prompt = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                prompt += f" {prefix}"
            prompt += f" <sent> {window} </sent>"

            inputs = self.tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.75,
                    top_k=None,
                    repetition_penalty=1.1,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text += " " + decoded[0]

        return output_text.strip()


# Initialize model
# HUMANIZER_MODEL options: "full" (DIPPER 11B) | "lightweight" (DIPPER T5-L) | "qwen" (fine-tuned Qwen2.5-3B)
_qwen_humanizer = None

if MODEL_MODE == "qwen":
    logger.info("Using fine-tuned Qwen2.5-3B-Instruct model")
    from app.inference_qwen import QwenHumanizer
    _qwen_humanizer = QwenHumanizer(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        checkpoint_dir=os.getenv("QWEN_CHECKPOINT", ""),  # LoRA adapter dir
        device=DEVICE,
    )
    paraphraser = None  # not used in qwen mode
elif MODEL_MODE == "lightweight":
    logger.info("Using lightweight DIPPER model (T5-Large, ~770M params)")
    paraphraser = DipperParaphraser(
        model_name="SamSJackson/paraphrase-dipper-no-ctx",
        tokenizer_name="google/t5-efficient-large-nl32",
        device=DEVICE,
    )
else:
    logger.info("Using full DIPPER model (T5-XXL, ~11B params)")
    paraphraser = DipperParaphraser(
        model_name="kalpeshk2011/dipper-paraphraser-xxl",
        tokenizer_name="google/t5-v1_1-xxl",
        device=DEVICE,
    )


# ──────────────────────────────────────────────
# Programmatic Post-Processing
# ──────────────────────────────────────────────

CONTRACTIONS = [
    (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"), (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"), (r"\bwill not\b", "won't"), (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"), (r"\bcould not\b", "couldn't"), (r"\bhas not\b", "hasn't"),
    (r"\bhave not\b", "haven't"), (r"\bhad not\b", "hadn't"), (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"), (r"\bthey are\b", "they're"), (r"\bwe are\b", "we're"),
    (r"\byou are\b", "you're"), (r"\bI am\b", "I'm"), (r"\bthat is\b", "that's"),
    (r"\bit is\b", "it's"), (r"\bthere is\b", "there's"), (r"\blet us\b", "let's"),
    (r"\bI have\b", "I've"), (r"\bthey have\b", "they've"), (r"\bwe have\b", "we've"),
    (r"\bI would\b", "I'd"), (r"\bI will\b", "I'll"), (r"\bthey will\b", "they'll"),
]

AI_WORDS = [
    (r"\butilize[sd]?\b", "use"), (r"\bleverage[sd]?\b", "use"),
    (r"\bcrucial(ly)?\b", "important"), (r"\bcomprehensive\b", "full"),
    (r"\brobust\b", "strong"), (r"\benhance[sd]?\b", "improve"),
    (r"\bfacilitate[sd]?\b", "help"), (r"\boptimal\b", "best"),
    (r"\binnovative\b", "new"), (r"\blandscape\b", "scene"),
    (r"\btestament\b", "proof"), (r"\bpivotal\b", "key"),
    (r"\bnuanced\b", "detailed"), (r"\bdelve[sd]?\b", "dig into"),
    (r"\bfoster(?:s|ed|ing)?\b", "grow"), (r"\bembark(?:s|ed|ing)?\b", "start"),
    (r"\bmyriad\b", "tons of"), (r"\bplethora\b", "bunch of"),
    (r"\brealm\b", "area"), (r"\bunderscore[sd]?\b", "show"),
    (r"\bmultifaceted\b", "complex"), (r"\bstreamline[sd]?\b", "simplify"),
    (r"\btapestry\b", "mix"), (r"\bnavigat(?:e|es|ed|ing)\b", "handle"),
    (r"\belevate[sd]?\b", "boost"), (r"\bbeacon\b", "example"),
    (r"\bdemonstrate[sd]?\b", "show"), (r"\bnumerous\b", "a lot of"),
    (r"\bindividuals\b", "people"), (r"\bsignificantly\b", "a lot"),
    (r"\bmoreover\b", "plus"), (r"\bfurthermore\b", "and"),
    (r"\badditionally\b", "also"), (r"\bconsequently\b", "so"),
    (r"\bnevertheless\b", "still"), (r"\bnonetheless\b", "but"),
    (r"\bthus\b", "so"), (r"\bhence\b", "so"),
    (r"\bmeticulous(ly)?\b", "careful"), (r"\bseamless(ly)?\b", "smooth"),
    (r"\bcrafted\b", "made"), (r"\bcurated\b", "picked"),
    (r"\bcommence[sd]?\b", "start"), (r"\bsufficient\b", "enough"),
    (r"\bprior to\b", "before"), (r"\bin order to\b", "to"),
    (r"\bregarding\b", "about"), (r"\bmethodology\b", "method"),
    (r"\bjourney\b", "process"), (r"\bkaleidoscope\b", "mix"),
    (r"\bsymphony\b", "blend"), (r"\binterplay\b", "connection"),
]

AI_PHRASES = [
    (r"\bplays? a (?:significant |crucial |vital |important |key )?role\b", "matters"),
    (r"\bserves? as (?:a )?", "works like "),
    (r"\ba wide (?:range|variety|array) of\b", "lots of"),
    (r"\bin today'?s (?:\w+ ?){0,2}(?:world|age|era)\b", "these days"),
    (r"\bit'?s (?:important|worth|essential) to (?:note|remember|understand) that\b", ""),
    (r"\bat the end of the day\b", "really"),
    (r"\bwhen it comes to\b", "with"),
    (r"\bin terms of\b", "for"),
    (r"\bon the other hand\b", "but then"),
    (r"\bhas been (?:shown|proven|demonstrated) to\b", "seems to"),
    (r"\bstudies (?:have )?(?:shown|suggest|indicate)\b", "research says"),
    (r"\bmany people (?:believe|think|feel)\b", "a lot of people figure"),
    (r"\ba testament to\b", "proof of"),
    (r"\bshed[s]? light on\b", "help explain"),
    (r"\bthroughout history\b", "over the years"),
    (r"\bacross the globe\b", "everywhere"),
    (r"\bscientific breakthroughs?\b", "big discoveries in science"),
    (r"\bhealth benefits?\b", "good stuff for your body"),
    (r"\baid[s]? digestion\b", "help your stomach"),
    (r"\bsupport[s]? (?:cardiovascular|heart) health\b", "keep your heart working right"),
    (r"\bpacked with\b", "full of"),
    (r"\bloaded with\b", "got plenty of"),
    (r"\bcultural significance\b", "meaning in culture"),
    (r"\bfor centuries\b", "for a really long time"),
]

BANNED_STARTERS = [
    (r"^However,?\s", "But "), (r"^Therefore,?\s", "So "),
    (r"^Additionally,?\s", "Also "), (r"^Moreover,?\s", "And "),
    (r"^Furthermore,?\s", "Plus "), (r"^Consequently,?\s", "So "),
    (r"^Nevertheless,?\s", "Still "), (r"^Nonetheless,?\s", "But "),
    (r"^In essence,?\s", ""), (r"^Essentially,?\s", ""),
    (r"^Fundamentally,?\s", ""), (r"^Ultimately,?\s", ""),
    (r"^It'?s worth noting\s?(that)?\s?", ""),
    (r"^In conclusion,?\s", ""), (r"^Overall,?\s", ""),
    (r"^To summarize,?\s", ""), (r"^In addition,?\s", "Also "),
]


def post_process(text: str) -> str:
    """Aggressive programmatic post-processing to remove remaining AI patterns."""
    r = text

    # Kill dashes and semicolons
    r = re.sub(r"\s+[—–]\s+", ", ", r)
    r = re.sub(r";\s*(\w)", lambda m: f". {m.group(1).upper()}", r)
    r = re.sub(r":(\s+[a-z])", lambda m: f". {m.group(1).strip()[0].upper()}{m.group(1).strip()[1:]}", r)

    # Phrase-level kills (before word-level)
    for pat, rep in AI_PHRASES:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    # Force contractions
    for pat, rep in CONTRACTIONS:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    # Word-level kills
    for pat, rep in AI_WORDS:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    # Fix banned sentence starters
    sentences = sent_tokenize(r)
    fixed = []
    for s in sentences:
        for pat, rep in BANNED_STARTERS:
            m = re.match(pat, s, re.IGNORECASE)
            if m:
                s = re.sub(pat, rep, s, flags=re.IGNORECASE)
                if rep == "" and s:
                    s = s[0].upper() + s[1:]
                break
        # Kill "This demonstrates..." pattern
        s = re.sub(
            r"^This (demonstrates|highlights|suggests|indicates|illustrates|means|reveals)",
            lambda m: {"demonstrates": "That shows", "highlights": "You can see",
                       "suggests": "Seems like", "indicates": "Looks like",
                       "illustrates": "You can see", "means": "So basically",
                       "reveals": "Turns out"}.get(m.group(1).lower(), "So"),
            s, flags=re.IGNORECASE
        )
        fixed.append(s)

    # Burstiness injection: break runs of similar-length sentences
    output = []
    for i, s in enumerate(fixed):
        wl = len(s.split())
        p1 = len(output[-1].split()) if output else 0
        p2 = len(output[-2].split()) if len(output) > 1 else 0
        if len(output) >= 2 and abs(wl - p1) < 5 and abs(p1 - p2) < 5 and wl > 10:
            ci = s.find(", ", len(s) // 3)
            if ci > 0:
                output.append(s[:ci + 1].strip())
                rest = s[ci + 2:].strip()
                if rest:
                    output.append(rest[0].upper() + rest[1:])
                continue
        output.append(s)

    r = " ".join(output)

    # Capitalize after sentence enders
    r = re.sub(r"([.!?])\s+([a-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", r)

    # Cleanup
    r = re.sub(r"\s{2,}", " ", r)
    r = r.replace(",.", ".").replace("..", ".")

    # Zero-width character injection (~4% rate, inside words)
    words = r.split(" ")
    zwc = ["\u200B", "\u200C", "\u200D", "\uFEFF"]
    for i in range(len(words)):
        if random.random() < 0.04 and len(words[i]) > 4:
            pos = 1 + random.randint(0, len(words[i]) - 3)
            c = random.choice(zwc)
            words[i] = words[i][:pos] + c + words[i][pos:]
    r = " ".join(words)

    return r.strip()


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to humanize")
    lex_diversity: int = Field(60, ge=0, le=100, description="Lexical diversity (0-100, step 20)")
    order_diversity: int = Field(60, ge=0, le=100, description="Order diversity (0-100, step 20)")
    sent_interval: int = Field(3, ge=1, le=5, description="Sentences per chunk")
    post_process_enabled: bool = Field(True, description="Run AI-phrase post-processing")

class HumanizeResponse(BaseModel):
    original: str
    humanized: str
    original_words: int
    humanized_words: int
    model: str
    processing_time: float


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_MODE, "device": DEVICE}


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize(req: HumanizeRequest):
    t0 = time.time()

    # Snap diversity values to valid DIPPER increments (multiples of 20)
    lex = min(100, max(0, round(req.lex_diversity / 20) * 20))
    order = min(100, max(0, round(req.order_diversity / 20) * 20))

    try:
        # Step 1: Paraphrase (DIPPER or Qwen)
        if MODEL_MODE == "qwen" and _qwen_humanizer is not None:
            result = _qwen_humanizer.humanize(req.text)
        else:
            result = paraphraser.paraphrase(
                text=req.text,
                lex_diversity=lex,
                order_diversity=order,
                sent_interval=req.sent_interval,
            )

        # Step 2: Post-processing
        if req.post_process_enabled:
            result = post_process(result)

        return HumanizeResponse(
            original=req.text,
            humanized=result,
            original_words=len(req.text.split()),
            humanized_words=len(result.split()),
            model=MODEL_MODE,
            processing_time=round(time.time() - t0, 2),
        )

    except Exception as e:
        logger.error(f"Humanization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
