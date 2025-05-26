from drift.drift_features import *
from drift.probe_bank import PROBE_BANK
from drift.drift_config import *
from shared.logger import logger
import nltk

def get_drift_analysis(stage: str, reply: str, previous_reply: str = None, previous_stage: str = None) -> dict:
    features = {
        "lexical_redundancy": fraction_repeated_words(reply) + (1.0 - fraction_unique_words(reply)),
        "style_shit": fraction_style_shift(reply),
        "past_tense": fraction_past_tense_verbs(reply),
        "question_ratio": fraction_sentences_that_are_questions(reply),
        "semantic_repetition": fraction_similarity(reply, previous_reply) if previous_reply else 0.0
    }

    # ✅ FEATURE_WEIGHTS 누락 키 점검
    missing_weights = [k for k in features if k not in FEATURE_WEIGHTS]
    if missing_weights:
        logger.warning(f"⚠️ FEATURE_WEIGHTS에 누락된 키: {missing_weights}")

    score = sum(features[k] * FEATURE_WEIGHTS.get(k, 0) for k in features)

    # ✅ soft drift 허용
    if previous_stage and previous_stage in ALLOWED_STAGE_TRANSITIONS:
        if stage in ALLOWED_STAGE_TRANSITIONS[previous_stage] and DRIFT_THRESHOLD < score < DRIFT_THRESHOLD + 0.1:
            score_drift = False
        else:
            score_drift = score > DRIFT_THRESHOLD
    else:
        score_drift = score > DRIFT_THRESHOLD

    # ✅ stage 유효성 검사 및 judge_func 처리
    if stage not in PROBE_BANK:
        logger.warning(f"⚠️ 정의되지 않은 스테이지 '{stage}' → 프롬프트 기반 드리프트 감지 생략")
        prompt_drift = False
    else:
        _, _, judge_func = PROBE_BANK[stage]
        prompt_drift = not judge_func(reply)

    reasons = []
    if score_drift:
        reasons.append("score")
    if prompt_drift:
        reasons.append("prompt")

    logger.info(
        f"[DRIFT DETECT] Stage: {stage}, Score: {score:.3f}, Drift: score={score_drift}, prompt={prompt_drift}, Features: {features}"
    )

    return {
        "drift": score_drift or prompt_drift,
        "reasons": reasons,
        "score": score,
        "features": features,
    }

def detect_persona_drift(stage: str, reply: str, previous_reply: str = None, previous_stage: str = None) -> bool:
    return get_drift_analysis(stage, reply, previous_reply, previous_stage)["drift"]

def run_detect(state) -> bool:
    try:
        previous_reply = state.history[-2] if len(state.history) >= 2 else None

        # ✅ drift_trace 안전성 보장
        if state.drift_trace and isinstance(state.drift_trace[-1], (list, tuple)) and len(state.drift_trace[-1]) == 2:
            previous_stage = state.drift_trace[-1][0]
        else:
            previous_stage = None

        if not state.response:
            logger.warning("⚠️ 응답이 비어있음 → 드리프트 감지 생략")
            return False

        drifted = detect_persona_drift(state.stage, state.response, previous_reply, previous_stage)
        state.drift_trace.append((state.stage, drifted))
        state.drift_trace = state.drift_trace[-5:]  # 최근 5개만 유지

        # ✅ boolean만 합산하여 drift 여부 판단
        return sum(d for _, d in state.drift_trace if isinstance(d, bool)) >= 3

    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return True
