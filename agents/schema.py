# 📁 agents/schema.py

from pydantic import BaseModel
from typing import Literal

class AgentState(BaseModel):
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3", "end"]
    question: str
    response: str = ""  # stream_empathy_reply 내부에서 사용됨
