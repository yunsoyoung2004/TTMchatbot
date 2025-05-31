from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Literal

class AgentState(BaseModel):
    # ✅ 세션 및 단계 정보
    session_id: str
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3"]
    turn: int

    # ✅ 대화 내용
    question: Optional[str] = None
    response: Optional[str] = None
    history: List[str] = Field(default_factory=list)

    # ✅ 대화 상태 관련 플래그
    intro_shown: bool = False
    retry_count: int = 0
    pending_response: Optional[str] = None
    awaiting_s_turn_decision: bool = False
    awaiting_preparation_decision: bool = False

    # ✅ 드리프트 상태
    drift_trace: List[Tuple[str, bool]] = Field(default_factory=list)  # 예: [("cbt1", True)]
    reset_triggered: bool = False  # MI 리셋 여부

    # ✅ 사용자 정보 (선택 사항)
    user_type: Optional[str] = None
    last_active_time: Optional[str] = None  # ISO8601 형식 권장

