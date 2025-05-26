import re
from collections import Counter
from nltk import pos_tag, word_tokenize
import difflib
from .utils import get_sia
import nltk

# ✅ NLTK 리소스 다운로드
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

# ✅ 스타일 변화 탐지용 패턴
STYLE_SHIFT_PATTERNS = ["짜증", "됐어", "죽겠어", "어쩌라고", "몰라"]

def fraction_repeated_words(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for c in counts.values() if c >= 2) / len(words)

def fraction_style_shift(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    teen = sum(1 for w in words if w in {"ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ"})
    aggressive = sum(1 for pat in STYLE_SHIFT_PATTERNS if pat in text)
    return (teen + aggressive) / len(words)

def fraction_past_tense_verbs(text: str) -> float:
    try:
        tags = pos_tag(word_tokenize(text))
        verbs = [t for _, t in tags if t.startswith("VB")]
        past = [t for t in verbs if t in ("VBD", "VBN")]
        return len(past) / len(verbs) if verbs else 0.0
    except Exception:
        return 0.0  # 예외 발생 시 안전하게 처리

def fraction_unique_words(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for c in counts.values() if c == 1) / len(words)

def fraction_sentences_that_are_questions(text: str) -> float:
    sentences = re.split(r'[.!?]', text)
    valid = [s for s in sentences if s.strip()]
    if not valid:
        return 0.0
    return sum(1 for s in valid if '?' in s) / len(valid)

def fraction_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()
