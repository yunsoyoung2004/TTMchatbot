from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional
import json, os, asyncio
from huggingface_hub import hf_hub_download

# ✅ 모델 경로 저장소
MODEL_PATHS = {}

# ✅ FastAPI 인스턴스 생성
app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 모델 다운로드 로직 (startup.py 대체)
@app.on_event("startup")
async def download_all_models():
    print("📦 모델 다운로드 시작")
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❗ Hugging Face 토큰이 없습니다.")
        return

    REPOS = {
        "mi":   ("youngbongbong/mimodel", "merged-mi-chat-q4_k_m.gguf"),
        "cbt":  ("youngbongbong/cbtmodel", "merged-cbt-chat-q4_k_m.gguf"),
        "ppi":  ("youngbongbong/ppimodel", "merged-ppi-prep-chat-q4_k_m.gguf"),
        "base": ("youngbongbong/mimodel", "merged-mi-chat-q4_k_m.gguf"),
    }

    for name, (repo, file) in REPOS.items():
        path = hf_hub_download(repo_id=repo, filename=file, token=token)
        MODEL_PATHS[name] = path
        print(f"✅ {name.upper()} 모델 경로 등록: {path}")

    asyncio.create_task(dummy_loop())  # keep-alive loop도 이곳에 포함

async def dummy_loop():
    while True:
        await asyncio.sleep(3600)

# ✅ 상태 모델
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

# ✅ 에이전트 import
from agents.empathy_agent import stream_empathy_reply
from agents.mi_agent import stream_mi_reply
from agents.s_turn_agent import stream_s_turn_reply
from agents.cbt_agent import stream_cbt_reply
from agents.action_agent import stream_ppi_reply

# ✅ 모델 준비 상태 확인
@app.get("/status")
def check_model_status():
    return {"ready": bool(MODEL_PATHS)}

@app.get("/")
def root():
    return JSONResponse({"message": "✅ TTM 멀티에이전트 챗봇 서버 실행 중"})

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    state = AgentState(**data.get("state", {}))

    async def async_gen():
        if not MODEL_PATHS:
            yield "⚠️ 모델이 아직 준비되지 않았습니다.\n".encode("utf-8")
            return

        if state.stage == "empathy":
            async for chunk in stream_empathy_reply(state.question.strip(), MODEL_PATHS["base"]):
                yield chunk
            yield b"\n---END_STAGE---\n" + json.dumps({"next_stage": "mi"}).encode("utf-8")

        elif state.stage == "mi":
            async for chunk in stream_mi_reply(state, MODEL_PATHS["mi"]):
                yield chunk

        elif state.stage == "s_turn":
            async for chunk in stream_s_turn_reply(state, MODEL_PATHS["base"]):
                yield chunk

        elif state.stage == "cbt":
            async for chunk in stream_cbt_reply(state, MODEL_PATHS["cbt"]):
                yield chunk

        elif state.stage in ["ppi", "action"]:
            async for chunk in stream_ppi_reply(state, MODEL_PATHS["ppi"]):
                yield chunk

        else:
            yield "⚠️ 현재 단계에서 스트리밍 응답이 지원되지 않습니다.\n".encode("utf-8")
            yield b"\n---END_STAGE---\n" + json.dumps({"next_stage": state.stage}).encode("utf-8")

    return StreamingResponse(async_gen(), media_type="text/plain")

# ✅ 로컬 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
