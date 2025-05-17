from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional
import json, os, asyncio, nltk

# ✅ 드리프트 감지
from utils.drift_detector import detect_persona_drift
from utils.logger import logger

# ✅ 에이전트 스트림 응답 함수
from agents import (
    stream_empathy_reply, stream_mi_reply,
    stream_cbt1_reply, stream_cbt2_reply, stream_cbt3_reply
)

# ✅ 모델 경로
MODEL_PATHS = {}

# ✅ FastAPI 인스턴스
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ NLTK 자원 및 더미 루프
@app.on_event("startup")
async def startup_event():
    for resource in ["punkt", "averaged_perceptron_tagger", "vader_lexicon"]:
        try: nltk.data.find(resource)
        except LookupError: nltk.download(resource)
    asyncio.create_task(dummy_loop())

async def dummy_loop():
    while True:
        await asyncio.sleep(3600)

# ✅ 사용자 상태 모델
class AgentState(BaseModel):
    session_id: str
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    pending_response: Optional[str] = None
    awaiting_s_turn_decision: Optional[bool] = False
    awaiting_preparation_decision: Optional[bool] = False
    retry_count: int = 0

# ✅ 스트리밍 엔드포인트
@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    state = AgentState(**data.get("state", {}))
    reply_chunks = []

    async def collect(agent_func, model_key):
        nonlocal reply_chunks
        async for chunk in agent_func(state, MODEL_PATHS[model_key]):
            decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            reply_chunks.append(decoded)
            yield chunk

    if not MODEL_PATHS:
        yield "⚠️ 모델이 준비되지 않았습니다.\n".encode("utf-8")
        return

    agent_map = {
        "empathy": stream_empathy_reply,
        "mi": stream_mi_reply,
        "cbt1": stream_cbt1_reply,
        "cbt2": stream_cbt2_reply,
        "cbt3": stream_cbt3_reply,
    }

    agent_func = agent_map.get(state.stage)
    if not agent_func:
        yield f"⚠️ {state.stage} 단계는 지원되지 않습니다.\n".encode("utf-8")
        return

    async for chunk in collect(agent_func, state.stage):
        yield chunk

    full_reply = "".join(reply_chunks).strip()

    # ✅ 드리프트 감지
    if detect_persona_drift(state.stage, full_reply):
        logger.warning(f"[DRIFT] {state.stage} → MI 단계로 전환됨")
        yield "\n[시스템] 페르소나 드리프트 감지됨 → MI 단계로 전환합니다.\n".encode("utf-8")
        state.stage = "mi"
        state.turn = 0

    # ✅ 상태 반환
    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": state.stage,
        "turn": state.turn + 1,
        "response": full_reply,
        "history": state.history + [state.question, full_reply]
    }, ensure_ascii=False).encode("utf-8")
