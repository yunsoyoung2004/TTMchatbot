from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional
import json, os, asyncio
from huggingface_hub import hf_hub_download

# ✅ FastAPI 인스턴스는 단 1회 선언
app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 상태 모델 정의
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

# ✅ 에이전트 불러오기
from agents.empathy_agent import stream_empathy_reply
from agents.mi_agent import stream_mi_reply
from agents.s_turn_agent import stream_s_turn_reply
from agents.cbt_agent import stream_cbt_reply
from agents.action_agent import stream_ppi_reply

# ✅ 모델 준비 상태
model_ready = False
model_paths = {}

# ✅ 모델 다운로드
@app.on_event("startup")
async def load_all_models():
    global model_ready, model_paths
    try:
        print("⏳ Hugging Face에서 모델 다운로드 중...")
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        models_to_download = {
            "base": {
                "repo_id": "youngbongbong/mimodel",
                "filename": "merged-mi-chat-q4_k_m.gguf",
            },
            "cbt": {
                "repo_id": "youngbongbong/cbtmodel",
                "filename": "merged-cbt-chat-q4_k_m.gguf",
            },
            "mi": {
                "repo_id": "youngbongbong/ppimodel",
                "filename": "merged-ppi-prep-chat-q4_k_m.gguf",
            },
            "ppi": {
                "repo_id": "youngbongbong/ppimodel",
                "filename": "merged-ppi-prep-chat-q4_k_m.gguf",
            }
        }

        for key, meta in models_to_download.items():
            print(f"📥 {key} 모델 다운로드 중...")
            path = hf_hub_download(
                repo_id=meta["repo_id"],
                filename=meta["filename"],
                token=hf_token
            )
            model_paths[key] = path
            print(f"✅ {key} 모델 경로: {path}")

        model_ready = True
        print("🎉 모든 모델 다운로드 및 경로 등록 완료")

    except Exception as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        model_ready = False

# ✅ 상태 확인용 라우터
@app.get("/status")
def check_model_status():
    return {"ready": model_ready}

# ✅ 루트 확인용 GET
@app.get("/")
def root():
    return JSONResponse({"message": "✅ TTM 멀티에이전트 챗봇 서버 실행 중"})

# ✅ 루트 HEAD (Render 헬스체크용)
@app.head("/")
def root_head():
    return Response(status_code=200)

# ✅ 스트리밍 응답 엔드포인트
@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    state = AgentState(**data.get("state", {}))

    async def async_gen():
        if not model_ready:
            yield "⚠️ 모델이 아직 준비되지 않았습니다.\n".encode("utf-8")
            return

        if state.stage == "empathy":
            async for chunk in stream_empathy_reply(state.question.strip(), model_paths["base"]):
                yield chunk
            yield b"\n---END_STAGE---\n" + json.dumps({"next_stage": "mi"}).encode("utf-8")

        elif state.stage == "mi":
            async for chunk in stream_mi_reply(state, model_paths["mi"]):
                yield chunk

        elif state.stage == "s_turn":
            async for chunk in stream_s_turn_reply(state, model_paths["base"]):
                yield chunk

        elif state.stage == "cbt":
            async for chunk in stream_cbt_reply(state, model_paths["cbt"]):
                yield chunk

        elif state.stage in ["ppi", "action"]:
            async for chunk in stream_ppi_reply(state, model_paths["ppi"]):
                yield chunk

        else:
            yield "⚠️ 현재 단계에서 스트리밍 응답이 지원되지 않습니다.\n".encode("utf-8")
            yield b"\n---END_STAGE---\n" + json.dumps({"next_stage": state.stage}).encode("utf-8")

    return StreamingResponse(async_gen(), media_type="text/plain")

# ✅ keep-alive dummy loop
@app.on_event("startup")
async def keep_alive():
    asyncio.create_task(dummy_loop())

async def dummy_loop():
    while True:
        await asyncio.sleep(3600)

# ✅ 로컬 실행용 (Render는 무시함)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
