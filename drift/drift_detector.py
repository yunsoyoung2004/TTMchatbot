from utils.drift_features import (
    fraction_repeated_words,
    teenager_score,
    fraction_past_tense_verbs,
    fraction_unique_words,
    fraction_sentences_that_are_questions,
)
from .probe_bank import PROBE_BANK
from utils.drift_config import FEATURE_WEIGHTS, DRIFT_THRESHOLD
from utils.logger import logger


def detect_persona_drift(stage: str, reply_text: str) -> bool:
    """
    특정 단계(stage)에서 주어진 응답(reply_text)이 페르소나 드리프트인지 여부 판단.
    점수 기반 + 프롬프트 기반(probe) 모두 적용.
    """
    # ✅ 특징 기반 점수 계산
    features = {
        "repetition": fraction_repeated_words(reply_text),
        "teen_slang": teenager_score(reply_text),
        "past_tense": fraction_past_tense_verbs(reply_text),
        "uniqueness": fraction_unique_words(reply_text),
        "question_ratio": fraction_sentences_that_are_questions(reply_text),
    }

    # ✅ 점수 기반 드리프트 판별
    score = sum(features[key] * FEATURE_WEIGHTS.get(key, 0.0) for key in features)
    score_drift = score > DRIFT_THRESHOLD

    # ✅ 프롬프트 기반 드리프트 판별
    prompt_drift = False
    if stage in PROBE_BANK:
        _, _, judge_func = PROBE_BANK[stage]
        prompt_drift = not judge_func(reply_text)

    # ✅ 로깅
    logger.info(
        f"[DRIFT DETECTOR] Stage={stage} | Score={score:.3f} | "
        f"ScoreDrift={score_drift} | PromptDrift={prompt_drift} | Features={features}"
    )

    return score_drift or prompt_drift


def probe_only_drift(stage: str, reply_text: str) -> bool:
    """
    프롬프트 기반 판단만 사용하는 드리프트 판별 함수 (점수 기반 무시)
    """
    if stage not in PROBE_BANK:
        return False

    _, _, judge_func = PROBE_BANK[stage]
    drifted = not judge_func(reply_text)

    logger.info(
        f"[PROBE ONLY DRIFT] Stage={stage} | Drift={drifted} | Reply=\"{reply_text}\""
    )

    return drifted
