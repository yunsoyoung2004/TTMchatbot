import re
from nltk.sentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

def get_sentiment(text: str, sentiment: str = "compound") -> float:
    """
    감정 점수를 반환 (VADER 기반)
    """
    return _sia.polarity_scores(text)[sentiment]

def contains_keywords(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)

empathy_probe = (
    "당신은 따뜻하고 공감적인 상담자입니다. 사용자의 감정을 존중하고, 그들이 겪는 어려움을 진심으로 이해하려고 노력합니다.",
    "솔직히 말해서 아무도 저를 이해하지 못해요...",
    lambda x: get_sentiment(x) > 0.25 or contains_keywords(x, ["이해", "공감", "마음", "그럴 수", "힘드셨겠", "괜찮"])
)

mi_probe = (
    "당신은 변화 동기를 함께 탐색하는 MI 상담자입니다. 양가감정을 수용하며, 변화에 대한 내적 동기를 찾아갑니다.",
    "그대로 두고 싶은 마음도 있지만, 바꾸고 싶은 생각도 있어요.",
    lambda x: contains_keywords(x, ["변화", "시도", "의지", "노력", "시작"]) or get_sentiment(x) > 0.1
)

cbt1_probe = (
    "당신은 소크라테스식 CBT 상담자로, 사용자가 자신의 생각과 감정을 탐색하도록 유도합니다.",
    "그때 가장 먼저 떠오른 생각은 무엇이었나요?",
    lambda x: contains_keywords(x, ["생각", "감정", "느낌"]) or x.endswith("?")
)

cbt2_probe = (
    "당신은 자동사고의 왜곡을 탐색하고, 대안적인 사고를 찾도록 도와주는 CBT 상담자입니다.",
    "혹시 그 생각에 어떤 근거가 있었을까요?",
    lambda x: contains_keywords(x, ["왜곡", "근거", "다르게", "다른 해석", "대안"]) or "?" in x
)

cbt3_probe = (
    "당신은 행동 실천과 방해요소 대처 전략을 도와주는 CBT 후반부 상담자입니다.",
    "이번 주에 어떤 실천을 시도해볼 수 있을까요?",
    lambda x: contains_keywords(x, ["실천", "계획", "시도", "준비", "고위험", "방해"]) or x.endswith("?")
)

PROBE_BANK = {
    "empathy": empathy_probe,
    "mi": mi_probe,
    "cbt1": cbt1_probe,
    "cbt2": cbt2_probe,
    "cbt3": cbt3_probe,
}
