from nltk.sentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

def get_sentiment(text: str, sentiment="compound"):
    return _sia.polarity_scores(text)[sentiment]

def contains_keywords(text: str, keywords: list[str]):
    return any(k in text for k in keywords)

PROBE_BANK = {
    "empathy": (
        "공감 상담자", 
        "아무도 날 이해 못 해요...",
        lambda x: get_sentiment(x) > 0.25 or contains_keywords(x, ["이해", "공감", "힘드셨겠", "괜찮"])
    ),
    "mi": (
        "MI 상담자", 
        "바꾸고 싶기도 해요",
        lambda x: contains_keywords(x, ["변화", "시도", "의지", "노력"])
    ),
    "cbt1": (
        "CBT1 상담자", 
        "그때 떠오른 생각은?", 
        lambda x: contains_keywords(x, ["생각", "감정", "느낌"]) or x.endswith("?")
    ),
    "cbt2": (
        "CBT2 상담자", 
        "그 생각에 근거는 있었나요?", 
        lambda x: contains_keywords(x, ["왜곡", "근거", "다르게", "대안"]) or "?" in x
    ),
    "cbt3": (
        "CBT3 상담자", 
        "이번 주 실천 계획은 무엇인가요?", 
        lambda x: contains_keywords(x, ["실천", "계획", "준비", "방해"]) or x.endswith("?")
    )
}
