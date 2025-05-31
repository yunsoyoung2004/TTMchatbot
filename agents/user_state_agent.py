# agents/user_state_agent.py

from typing import Literal, List, Tuple
from pydantic import BaseModel
from drift.detector import get_drift_analysis  # ✅ 직접 감지 함수 사용

# ✅ 상태 모델 정의
class AgentState(BaseModel):
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3"]
    question: str
    response: str
    history: List[str]
    turn: int
    drift_trace: List[Tuple[str, bool]] = []

# ✅ run_detect 재정의 (진짜 분석기 사용)
def run_detect(state: AgentState) -> dict:
    previous = state.history[-2] if len(state.history) >= 2 else ""
    return get_drift_analysis(
        stage=state.stage,
        reply=state.response,
        previous_reply=previous,
        previous_stage=None
    )

# ✅ 점수 기반 평가 함수 (프롬프트 제거됨)
def evaluate_user_state_score_only(state: AgentState) -> Tuple[str, bool]:
    result = run_detect(state)
    score = result.get("score", 0.0)
    reasons = result.get("reasons", [])
    rollback = result.get("drift", False)
    summary = f"[DRIFT SCORE] {score:.2f} | 이유: {', '.join(reasons) if reasons else '없음'}"
    return summary, rollback

# ✅ 사용자 상태 평가 실행 함수
async def run_user_state_agent(state: AgentState, model_path: str = "", mode: str = "drift_profile"):
    if mode == "plain":
        return {}

    result = run_detect(state)

    if mode == "drift_only":
        return {"enhanced": result}

    if result.get("drift", False):
        summary, rollback = evaluate_user_state_score_only(state)
        print(f"[DRIFT-EVAL] {summary} → MI 전환 필요? {rollback}")
        return {
            "need_rollback": rollback,
            "summary": summary
        }

    return {}
