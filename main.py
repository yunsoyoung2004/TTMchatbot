from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Tuple
from huggingface_hub import snapshot_download
from functools import partial
import json, os, asyncio, time, re, logging
import tqdm.std
import threading

# ✅ tqdm 병렬 패치
if not hasattr(tqdm.std.tqdm, "_lock"):
    tqdm.std.tqdm._lock = threading.RLock()
if not hasattr(tqdm.std.tqdm, "_instances"):
    tqdm.std.tqdm._instances = set()

# ✅ 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ttmchatbot")

# ✅ 에이전트 임포트
from agents.empathy_agent import stream_empathy_reply
from agents.mi_agent import stream_mi_reply
from agents.cbt1_agent import stream_cbt1_reply
from agents.cbt2_agent import stream_cbt2_reply
from agents.cbt3_agent import stream_cbt3_reply
from agents.user_state_agent import run_user_state_agent, run_detect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentState(BaseModel):
    session_id: str
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3", "end"]
    question: Optional[str] = None
    response: Optional[str] = None
    history: List[str] = Field(default_factory=list)
    turn: int = 0
    preset_questions: List[str] = Field(default_factory=list)
    drift_trace: List[Tuple[str, bool]] = Field(default_factory=list)
    reset_triggered: bool = False
    intro_shown: bool = False
    retry_count: int = 0
    pending_response: Optional[str] = None
    awaiting_s_turn_decision: bool = False
    awaiting_preparation_decision: bool = False
    user_profile: Optional[dict] = None
    user_type: Optional[str] = None
    last_active_time: Optional[str] = None

@app.on_event("startup")
async def startup_tasks():
    global model_ready, model_paths
    try:
        logger.info("🚀 모델 다운로드 시작")
        loop = asyncio.get_event_loop()
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.error("❌ 환경 변수 HUGGINGFACE_TOKEN이 설정되지 않았습니다")
            model_ready = False
            return

        async def dl(repo_id: str, local_dir: str):
            logger.info(f"📥 {repo_id} 다운로드 시작 → {local_dir}")
            path = await loop.run_in_executor(None, partial(
                snapshot_download,
                repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                revision="main",
                token=token,
            ))
            logger.info(f"✅ {repo_id} 다운로드 완료 → {path}")
            return repo_id, path

        results = await asyncio.gather(
            dl("youngbongbong/empathymodel", "/models/empathy"),
            dl("youngbongbong/mimodel", "/models/mi"),
            dl("hieupt/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF", "/models/detect"),
            dl("youngbongbong/cbt1model", "/models/cbt1"),
            dl("youngbongbong/cbt2model", "/models/cbt2"),
            dl("youngbongbong/cbt3model", "/models/cbt3"),
            return_exceptions=True
        )

        paths = {}
        for result in results:
            if isinstance(result, Exception):
                model_ready = False
                return
            repo_id, path = result
            if "empathymodel" in repo_id:
                paths["empathy"] = os.path.join(path, "merged-empathy-8.0B-chat-Q4_K_M.gguf")
            elif "mimodel" in repo_id:
                paths["mi"] = os.path.join(path, "merged-mi-chat-q4_k_m.gguf")
            elif "cbt1model" in repo_id:
                paths["cbt1"] = os.path.join(path, "merged-first-8.0B-chat-Q4_K_M.gguf")
            elif "cbt2model" in repo_id:
                paths["cbt2"] = os.path.join(path, "merged-mid-8.0B-chat-Q4_K_M.gguf")
            elif "cbt3model" in repo_id:
                paths["cbt3"] = os.path.join(path, "merged-cbt3-8.0B-chat-Q4_K_M.gguf")
            elif "TinyLlama" in repo_id:
                paths["detect"] = os.path.join(path, "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf")

        model_paths = paths
        model_ready = all(os.path.exists(p) for p in model_paths.values())

        from eval.eval_drift import evaluate_drift_detection
        result = await evaluate_drift_detection()  # ✅ 결과 저장

        if result:
            logger.info("🏁 Drift Detection 최종 평가 결과 요약:")
            for k, v in result.items():
                logger.info(f"{k}: {v}")

    except Exception:
        logger.exception("❌ startup_tasks() 전체 실패")
        model_ready = False

    asyncio.create_task(dummy_loop())


@app.get("/")
def root():
    return JSONResponse({"message": "✅ TTM 멀티에이전트 챗봇 서버 실행 중"})

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/status")
def check_model_status():
    return {"ready": model_ready}

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode())
        incoming_state = data.get("state", {})
        incoming_state.setdefault("preset_questions", [])
        incoming_state.setdefault("drift_trace", [])
        state = AgentState(**incoming_state)
    except Exception:
        return StreamingResponse(iter([
            R"\n⚠️ 입력 상태를 파싱하는 중 오류가 발생했습니다.\n",
            b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": "empathy",
                "response": "입력 상태가 잘못되었습니다. 다시 시도해 주세요.",
                "turn": 0,
                "history": [],
                "preset_questions": [],
                "drift_trace": [],
                "user_profile": {},
                "reset_triggered": False,
                "intro_shown": False
            }, ensure_ascii=False).encode("utf-8")
        ]), media_type="text/plain")

    async def async_gen():
        if not model_ready:
            yield R"⚠️ 모델이 아직 준비되지 않았습니다.\n"
            return

        drift_result = run_detect(state)

        # ✅ 오직 reset_triggered 기준만으로 리셋 응답 출력
        if drift_result.get("reset_triggered"):
            yield drift_result["response"].encode("utf-8")
            yield b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": drift_result.get("next_stage", state.stage),
                "response": drift_result.get("response", ""),
                "turn": drift_result.get("turn", 0),
                "history": drift_result.get("history", []),
                "preset_questions": drift_result.get("preset_questions", []),
                "drift_trace": drift_result.get("drift_trace", []),
                "user_profile": drift_result.get("user_profile", {}),
                "reset_triggered": False,  # ✅ 다음 턴으로 넘기지 않음
                "intro_shown": drift_result.get("intro_shown", False)
            }, ensure_ascii=False).encode("utf-8")
            return

        async def collect_stream(generator):
            async for chunk in generator:
                yield chunk

        agent_streams = {
            "empathy": lambda: stream_empathy_reply((state.question or "").strip(), model_paths["empathy"], state.turn, state),
            "mi": lambda: stream_mi_reply(state, model_paths["mi"]),
            "cbt1": lambda: stream_cbt1_reply(state, model_paths["cbt1"]),
            "cbt2": lambda: stream_cbt2_reply(state, model_paths["cbt2"]),
            "cbt3": lambda: stream_cbt3_reply(state, model_paths["cbt3"]),
        }

        if state.stage not in agent_streams:
            yield R"모든 세션이 종료되었습니다. 감사합니다.\n"
            return

        agent_gen = collect_stream(agent_streams[state.stage]())
        async for chunk in agent_gen:
            yield chunk

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "response": state.response or "",
            "turn": state.turn,
            "history": state.history,
            "preset_questions": state.preset_questions,
            "drift_trace": state.drift_trace,
            "user_profile": state.user_profile or {},
            "reset_triggered": False,
            "intro_shown": state.intro_shown
        }, ensure_ascii=False).encode("utf-8")

    return StreamingResponse(async_gen(), media_type="text/plain")

async def dummy_loop():
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=True)

