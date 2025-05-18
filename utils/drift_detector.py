from utils.drift_features import (
    fraction_repeated_words,
    teenager_score,
    fraction_past_tense_verbs,
    fraction_unique_words,
    fraction_sentences_that_are_questions,
)
from utils.drift_config import FEATURE_WEIGHTS, DRIFT_THRESHOLD
from drift.probe_bank import PROBE_BANK
from utils.logger import logger  # ✅ 전역 로거 사용

# ✅ 페르소나 드리프트 감지 (점수 기반 + 프롬프트 기반)
def detect_persona_drift(stage: str, reply_text: str) -> bool:
    """
    특정 단계(stage)에서 주어진 응답(reply_text)이 페르소나 드리프트인지 여부 판단
    점수 기반 + 프롬프트 기반 모두 통합 적용
    """
    # 🎯 특성값 계산
    features = {
        "repetition": fraction_repeated_words(reply_text),
        "teen_slang": teenager_score(reply_text),
        "past_tense": fraction_past_tense_verbs(reply_text),
        "uniqueness": fraction_unique_words(reply_text),
        "question_ratio": fraction_sentences_that_are_questions(reply_text),
    }

    # 🧮 최종 점수 계산
    score = sum(features[key] * FEATURE_WEIGHTS.get(key, 0) for key in features)

    # ✳️ 점수 기반 드리프트
    score_drift = score > DRIFT_THRESHOLD

    # 🧪 프롬프트 기반 드리프트
    prompt_drift = False
    if stage in PROBE_BANK:
        _, _, judge_func = PROBE_BANK[stage]
        prompt_drift = not judge_func(reply_text)

    # 📋 로깅
    logger.info(
        f"[DRIFT DETECTOR] Stage: {stage}, Score: {score:.3f}, "
        f"Drift: score={score_drift}, probe={prompt_drift}, Features: {features}"
    )

    return score_drift or prompt_drift
