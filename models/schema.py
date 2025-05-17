# 📁 shared/state.py
from typing import List, Optional, Literal
from pydantic import BaseModel

# ✅ TTM 단계 기반 에이전트 상태 모델
class AgentState(BaseModel):
    session_id: str                                      # 세션 식별자
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3", "action"]
    question: str                                         # 사용자의 현재 질문
    response: str                                         # 모델의 직전 응답
    history: List[str]                                   # 대화 히스토리 (交互 저장)
    turn: int                                             # 현재 턴 번호
    intro_shown: bool                                     # 도입 메시지 출력 여부
    pending_response: Optional[str] = None                # 지연 응답 캐시 (필요시)
    awaiting_s_turn_decision: Optional[bool] = False      # CBT1 의사결정 여부
    awaiting_preparation_decision: Optional[bool] = False # CBT3 실천계획 여부
    retry_count: int = 0                                  # 반복 실패 횟수

# ✅ 요청 본문 모델
class ChatRequest(BaseModel):
    state: AgentState
