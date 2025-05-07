from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional
import json

# ✅ 상태 정의
class AgentState(BaseModel):
    stage: Literal["empathy", "mi", "s_turn", "cbt", "ppi", "action", "end"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    pending_response: Optional[str] = None
    awaiting_s_turn_decision: Optional[bool] = False
    awaiting_preparation_decision: Optional[bool] = False
    retry_count: int = 0

class ChatRequest(BaseModel):
    state: AgentState

# ✅ 스트리밍 함수만 import
from agents.empathy_agent import stream_empathy_reply
from agents.mi_agent import stream_mi_reply
from agents.s_turn_agent import stream_s_turn_reply
from agents.cbt_agent import stream_cbt_reply
from agents.action_agent import stream_ppi_reply

# ✅ FastAPI 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return JSONResponse({"message": "✅ TTM 멀티에이전트 챗봇 서버 실행 중"})

# ✅ 스트리밍 기반 단일 엔드포인트
@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    state = AgentState(**data.get("state", {}))

    async def async_gen():
        if state.stage == "empathy":
            async for chunk in stream_empathy_reply(state.question.strip()):
                yield chunk  # ✅ 이미 bytes인 경우 그대로 yield
            yield b"\n---END_STAGE---\n" + json.dumps({"next_stage": "mi"}).encode("utf-8")

        elif state.stage == "mi":
            async for chunk in stream_mi_reply(state):
                yield chunk

        elif state.stage == "s_turn":
            async for chunk in stream_s_turn_reply(state):
                yield chunk

        elif state.stage == "cbt":
            async for chunk in stream_cbt_reply(state):
                yield chunk

        elif state.stage in ["ppi", "action"]:
            async for chunk in stream_ppi_reply(state):
                yield chunk

        else:
            yield "⚠️ 현재 단계에서 스트리밍 응답이 지원되지 않습니다.\n".encode("utf-8")
            yield b"\n---END_STAGE---\n" + json.dumps({"next_stage": state.stage}).encode("utf-8")

    return StreamingResponse(async_gen(), media_type="text/plain")
