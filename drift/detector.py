import re
from drift.drift_features import *
from drift.drift_config import *
from shared.logger import logger

def get_drift_analysis(stage: str, reply: str, previous_reply: str = None, previous_stage: str = None) -> dict:
    meaningless = is_meaningless(reply)
    semantic_similarity = fraction_similarity(reply, previous_reply) if previous_reply else 0.0
    semantic_score = 1.0 - semantic_similarity

    lexical = 1.0 if meaningless else (fraction_repeated_words(reply) + (1.0 - fraction_unique_words(reply)))
    style = 0.5 if meaningless else fraction_style_shift(reply)
    semantic = semantic_score

    features = {
        "lexical_redundancy": lexical,
        "style_shit": style,
        "semantic_repetition": semantic
    }

    score = (
        lexical * FEATURE_WEIGHTS.get("lexical_redundancy", 0.0)
        + style * FEATURE_WEIGHTS.get("style_shit", 0.0)
        + semantic * FEATURE_WEIGHTS.get("semantic_repetition", 0.0)
    )

    drifted = score > DRIFT_THRESHOLD or meaningless
    reasons = []
    if score > DRIFT_THRESHOLD:
        reasons.append(f"score>{DRIFT_THRESHOLD:.2f}")
    if meaningless:
        reasons.append("meaningless_input")

    logger.info(f"{'🟥 DRIFT 발생' if drifted else '🟩 DRIFT 없음'} | Stage={stage} | Score={score:.3f}")
    logger.info(f" ↳ Features: Lexical={lexical:.3f}, Style={style:.3f}, Semantic={semantic:.3f}")
    logger.info(f" ↳ Reasons: {', '.join(reasons) if reasons else '없음'}")

    return {
        "drift": drifted,
        "reasons": reasons,
        "score": score,
        "features": features
    }

def pure_run_detect(state) -> dict:
    previous_reply = state.history[-2] if len(state.history) >= 2 else None
    previous_stage = (
        state.drift_trace[-1][0]
        if state.drift_trace and isinstance(state.drift_trace[-1], (list, tuple)) and len(state.drift_trace[-1]) == 2
        else None
    )
    return get_drift_analysis(state.stage, state.response, previous_reply, previous_stage)

def run_detect(state) -> dict:
    try:
        if not state.response:
            logger.warning("⚠️ 응답이 비어있음 → 드리프트 감지 생략")
            return {
                "next_stage": state.stage,
                "turn": state.turn,
                "response": "",
                "history": state.history,
                "preset_questions": state.preset_questions,
                "drift_trace": state.drift_trace,
                "user_profile": state.user_profile or {},
                "reset_triggered": False,
                "intro_shown": state.intro_shown,
                "reasons": [],
            }

        analysis = pure_run_detect(state)
        score = analysis["score"]
        drifted = analysis["drift"]
        reasons = analysis.get("reasons", [])

        state.drift_trace.append((state.stage, drifted))
        state.drift_trace = state.drift_trace[-3:]
        drift_count = sum(d for _, d in state.drift_trace)

        logger.info(f"📊 최근 3턴 드리프트 상태: {[d for _, d in state.drift_trace]} (총 {drift_count}회)")

        reset_triggered = False
        if drift_count >= 3:
            logger.info("🚨 최근 3턴 중 3회 드리프트 감지 → MI 단계로 전환")
            state.stage = "mi"
            state.turn = 0
            state.drift_trace.clear()
            state.history.clear()
            state.response = (
                "지금 상담을 다시 시작해볼게요. "
                "천천히 괜찮으시다면 지금 어떤 점이 가장 고민되시는지 들려주세요."
            )
            reset_triggered = True
        elif drifted:
            logger.info("⚠️ 드리프트 감지됨 → 안내 문구 추가")
            state.response = str(state.response).strip() + "\n\n천천히 침착하게 다시 생각해서 대답해볼까요?"

        return {
            "next_stage": state.stage,
            "turn": state.turn,
            "response": str(state.response),
            "history": state.history,
            "preset_questions": getattr(state, "preset_questions", []),
            "drift_trace": state.drift_trace,
            "user_profile": getattr(state, "user_profile", {}),
            "reset_triggered": reset_triggered,
            "intro_shown": getattr(state, "intro_shown", False),
            "reasons": reasons,
            "score": score,
        }

    except Exception as e:
        logger.error(f"❌ Drift detection failed: {e}")
        return {
            "next_stage": "mi",
            "turn": 0,
            "response": "오류가 발생해 상담을 다시 시작합니다. 다시 어떤 점이 힘드신지 말씀해 주세요.",
            "history": [],
            "preset_questions": [],
            "drift_trace": [],
            "user_profile": {},
            "reset_triggered": True,
            "intro_shown": False,
            "reasons": ["exception"],
            "score": 0.0,
        }
