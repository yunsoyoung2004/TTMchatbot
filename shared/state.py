from typing import TypedDict, Literal, List

class AgentState(TypedDict):
    stage: Literal[
        "empathy", "mi", "s_turn",
        "cbt1", "cbt2", "cbt3",   # ✅ 단계별 CBT 구분
        "action", "ppi", "end"
    ]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    awaiting_preparation_decision: bool
    awaiting_end_decision: bool
