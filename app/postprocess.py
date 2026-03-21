"""
Post-processing for humanized text.
Extracted here so training code can import it without triggering model loading.
"""

import re

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

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

    r = re.sub(r"\s+[—–]\s+", ", ", r)
    r = re.sub(r";\s*(\w)", lambda m: f". {m.group(1).upper()}", r)
    r = re.sub(r":(\s+[a-z])", lambda m: f". {m.group(1).strip()[0].upper()}{m.group(1).strip()[1:]}", r)

    for pat, rep in AI_PHRASES:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    for pat, rep in CONTRACTIONS:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

    for pat, rep in AI_WORDS:
        r = re.sub(pat, rep, r, flags=re.IGNORECASE)

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
        s = re.sub(
            r"^This (demonstrates|highlights|suggests|indicates|illustrates|means|reveals)",
            lambda m: {"demonstrates": "That shows", "highlights": "You can see",
                       "suggests": "Seems like", "indicates": "Looks like",
                       "illustrates": "You can see", "means": "So basically",
                       "reveals": "Turns out"}.get(m.group(1).lower(), "So"),
            s, flags=re.IGNORECASE
        )
        fixed.append(s)

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
    r = re.sub(r"([.!?])\s+([a-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", r)
    r = re.sub(r"\s{2,}", " ", r)
    r = r.replace(",.", ".").replace("..", ".")

    # Strip any zero-width / invisible characters the model may have inserted
    for zwc in ("\u200B", "\u200C", "\u200D", "\uFEFF", "\u00AD"):
        r = r.replace(zwc, "")


    return r.strip()
