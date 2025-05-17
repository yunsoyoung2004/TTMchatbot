import re
from collections import Counter
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# ✅ 감성 분석기 전역 객체 (재사용)
sia = SentimentIntensityAnalyzer()

def fraction_repeated_words(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    repeated = sum(1 for count in word_counts.values() if count >= 2)
    return repeated / len(words) if words else 0

def teenager_score(text: str) -> float:
    # ✅ 한국어 청소년 은어 및 신조어 포함 (키워드 + 패턴)
    slang_keywords = {
        "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㄱㄱ", "ㅇㅇ", "ㄴㄴ", "ㄷㄷ", "ㅈㅅ", "ㄹㅇ", "ㅇㅈ",
        "헐", "대박", "쩐다", "오진다", "실화냐", "존맛", "개이득", "미쳤다", "무야호",
        "ㅂㅂ", "ㅎㅇ", "ㅗ", "ㅜ", "ㅓ", "ㅏ", "ㄹㅊ", "ㄹㅋ", "ㄱㅅ", "ㄴㅇㅅㅌ", "ㅅㅂ"
    }
    slang_patterns = [
        r"ㅋ{2,}", r"ㅎ{2,}", r"ㅠ{2,}", r"ㅜ{2,}", r"ㄷ{2,}", r"ㄱ{2,}", r"ㅅㅂ", r"존맛", r"개이득"
    ]
    words = re.findall(r'\b\w+\b', text.lower())
    total = len(words)
    slang_count = sum(1 for word in words if word in slang_keywords)
    for pattern in slang_patterns:
        slang_count += len(re.findall(pattern, text))
    return slang_count / total if total else 0

def fraction_past_tense_verbs(text: str) -> float:
    tags = pos_tag(word_tokenize(text))
    verbs = [tag for _, tag in tags if tag.startswith("VB")]
    past = [tag for tag in verbs if tag in ("VBD", "VBN")]
    return len(past) / len(verbs) if verbs else 1.0

def fraction_unique_words(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    unique = sum(1 for count in word_counts.values() if count == 1)
    return unique / len(words) if words else 0

def fraction_sentences_that_are_questions(text: str) -> float:
    sentences = re.split(r'[.!?]', text)
    questions = [s for s in sentences if '?' in s.strip()]
    return len(questions) / len(sentences) if sentences else 0
