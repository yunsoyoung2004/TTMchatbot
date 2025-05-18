from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional
import os, asyncio, json, traceback
from huggingface_hub import hf_hub_download

from utils.drift_detector import detect_persona_drift
from agents import (
    stream_empathy_reply, stream_mi_reply,
    stream_cbt1_reply, stream_cbt2_reply, stream_cbt3_reply
)

print("🌀 main.py 실행됨 - FastAPI 앱 초기화", flush=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_CONFIG = {
    "empathy": ("youngbongbong/empathymodel", "merged-empathy-8.0B-chat-Q4_K_M.gguf"),
    "mi": ("youngbongbong/mimodel", "merged-mi-chat-q4_k_m.gguf"),
    "cbt1": ("youngbongbong/cbt1model", "merged-first-8.0B-chat-Q4_K_M.gguf"),
    "cbt2": ("youngbongbong/cbt2model", "merged-mid-8.0B-chat-Q4_K_M.gguf"),
    "cbt3": ("youngbongbong/cbt3model", "merged-cbt3-8.0B-chat-Q4_K_M.gguf"),
}
MODEL_PATHS = {}

@app.on_event("startup")
async def startup_event():
    print("🚀 [STARTUP] 앱 시작됨 - 모델 다운로드 시작", flush=True)
    await download_all_models()
    asyncio.create_task(dummy_loop())

async def download_all_models():
    token = os.getenv("HUGGINGFACE_TOKEN")
    for stage, (repo, filename) in MODEL_CONFIG.items():
        try:
            print(f"📦 [{stage}] 모델 다운로드 시작...", flush=True)
            model_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                token=token,
                cache_dir="/root/.cache/huggingface",
                force_download=False
            )
            MODEL_PATHS[stage] = model_path
            print(f"✅ [{stage}] 다운로드 완료 → {model_path}", flush=True)
        except Exception as e:
            print(f"❌ [{stage}] 다운로드 실패", flush=True)
            traceback.print_exc()

async def dummy_loop():
    while True:
        await asyncio.sleep(3600)

@app.get("/")
async def root():
    return {"status": "ready" if all(MODEL_PATHS.values()) else "initializing"}

@app.get("/status")
async def status():
    return {
        "ready": all(MODEL_PATHS.values()),
        "models": {
            stage: os.path.exists(path) if path else False
            for stage, path in MODEL_PATHS.items()
        }
    }

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

@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    state = AgentState(**data.get("state", {}))
    reply_chunks = []

    async def event_stream():
        model_path = MODEL_PATHS.get(state.stage)
        if not model_path or not os.path.exists(model_path):
            yield f"⚠️ 모델 파일 누락: {state.stage}\n".encode("utf-8")
            return

        agent_func = {
            "empathy": stream_empathy_reply,
            "mi": stream_mi_reply,
            "cbt1": stream_cbt1_reply,
            "cbt2": stream_cbt2_reply,
            "cbt3": stream_cbt3_reply,
        }.get(state.stage)

        if not agent_func:
            yield f"⚠️ 지원되지 않는 단계: {state.stage}\n".encode("utf-8")
            return

        try:
            async for chunk in agent_func(state, model_path):
                reply_chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
                yield chunk
        except Exception as e:
            yield f"⚠️ 오류 발생: {e}\n".encode("utf-8")
            return

        full_reply = "".join(reply_chunks).strip()

        try:
            if detect_persona_drift(state.stage, full_reply):
                yield "\n[시스템] 페르소나 드리프트 감지됨. MI 단계로 이동합니다.\n".encode("utf-8")
                state.stage = "mi"
                state.turn = 0
        except Exception as e:
            print(f"[드리프트 예외 무시] {e}", flush=True)

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn + 1,
            "response": full_reply,
            "history": state.history + [state.question, full_reply]
        }, ensure_ascii=False).encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/plain")
