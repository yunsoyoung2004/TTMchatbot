import re
from collections import Counter
import difflib

STYLE_PATTERNS = [
    r"짜증", r"됐어", r"죽겠", r"어쩌", r"싫어", r"안해", r"그만", r"귀찮",
    r"몰라", r"하하+", r"뭐래", r"끄적", r"재밌", r"흥", r"젠장", r"[ㅋㅎ]{2,}", r"하\.\.\.", r"으으+"
]

def fraction_repeated_words(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(words)

def fraction_unique_words(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    unique = sum(1 for c in counts.values() if c == 1)
    return unique / len(words)

def fraction_style_shift(text: str) -> float:
    text = text.lower()
    matches = 0
    for pat in STYLE_PATTERNS:
        matches += len(re.findall(pat, text))
    words = re.findall(r'\b\w+\b', text)
    return matches / len(words) if words else 0.0

def fraction_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_tokens = set(re.findall(r'\b\w+\b', a.lower()))
    b_tokens = set(re.findall(r'\b\w+\b', b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    seq_sim = difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()
    return (jaccard + seq_sim) / 2

def is_meaningless(text: str) -> bool:
    cleaned = re.sub(r"[가-힣a-zA-Z0-9]", "", text)
    return (
        re.fullmatch(r"(.)\1{4,}", text)
        or re.fullmatch(r"[ㅋㅎㅜㅠㅏ-ㅣㄱ-ㅎ]{4,}", text)
        or (len(text.strip()) <= 4 and len(cleaned) >= 3)
        or re.fullmatch(r"(하하|ㅎㅎ|ㅋㅋ)+", text.strip())
    )
