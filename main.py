from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Tuple
from huggingface_hub import snapshot_download
from functools import partial
import json, os, asyncio, time, re, logging

# ✅ tqdm 락 오류 패치 (Cloud Run에서 병렬 다운로드 안전하게 실행되도록)
import tqdm.std
import threading

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
from agents.user_state_agent import run_user_state_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentState(BaseModel):
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: Optional[int] = 0
    preset_questions: List[str] = Field(default_factory=list)
    drift_trace: List[Tuple[str, bool]] = Field(default_factory=list)
    user_profile: Optional[dict] = None

model_ready = False
model_paths = {}

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
            try:
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
            except Exception as e:
                logger.exception(f"❌ {repo_id} 다운로드 실패")
                raise e

        # ✅ 병렬 다운로드
        results = await asyncio.gather(
            dl("youngbongbong/empathymodel", "/models/empathy"),
            dl("youngbongbong/mimodel", "/models/mi"),
            dl("hieupt/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF", "/models/detect"),
            dl("youngbongbong/cbt1model", "/models/cbt1"),
            dl("youngbongbong/cbt2model", "/models/cbt2"),
            dl("youngbongbong/cbt3model", "/models/cbt3"),
            return_exceptions=True
        )

        # ✅ 다운로드 실패 검사 및 경로 정리
        paths = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("❌ 하나 이상의 모델 다운로드 중단됨 → model_ready = False")
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

        # ✅ 모델 경로 확인
        model_paths = paths
        for name, path in model_paths.items():
            if not os.path.exists(path):
                logger.error(f"❌ 모델 경로 없음: {name} → {path}")

        model_ready = all(os.path.exists(p) for p in model_paths.values())
        logger.info("✅ 모델 준비 완료" if model_ready else "❌ 모델 경로 오류 발생")

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
        logger.info(f"📦 RAW REQUEST BODY: {body.decode(errors='ignore')}")
        data = json.loads(body.decode())
        mode = data.get("mode", "drift_profile")
        incoming_state = data.get("state", {})
        incoming_state.setdefault("preset_questions", [])
        incoming_state.setdefault("drift_trace", [])
        state = AgentState(**incoming_state)
        logger.info(f"🟢 STAGE={state.stage.upper()} | TURN={state.turn} | Q='{state.question.strip()}'")
    except Exception:
        logger.exception("❌ 입력 파싱 오류")
        return StreamingResponse(iter([
            r"\n⚠️ 입력 상태를 파싱하는 중 오류가 발생했습니다.\n",
            b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": "empathy",
                "response": "입력 상태가 잘못되었습니다. 다시 시도해 주세요.",
                "turn": 0,
                "history": [],
                "preset_questions": [],
                "drift_trace": [],
                "user_profile": {}
            }, ensure_ascii=False).encode("utf-8")
        ]), media_type="text/plain")

    async def async_gen():
        if not model_ready:
            logger.warning("⚠️ 모델이 아직 준비되지 않았습니다.")
            yield r"⚠️ 모델이 아직 준비되지 않았습니다.\n"
            return

        logger.info(f"🧭 현재 단계: {state.stage.upper()} | 턴: {state.turn}")
        logger.info(f"📨 사용자 질문: '{state.question.strip()}'")

        full_text = ""
        start_time = time.time()

        async def collect_stream(generator):
            nonlocal full_text
            async for chunk in generator:
                try:
                    full_text += chunk.decode("utf-8")
                except Exception as e:
                    logger.warning(f"⚠️ 디코딩 오류: {e}")
                    continue
                yield chunk

        agent_streams = {
            "empathy": lambda: stream_empathy_reply(state.question.strip(), model_paths["empathy"], state.turn, state),
            "mi": lambda: stream_mi_reply(state, model_paths["mi"]),
            "cbt1": lambda: stream_cbt1_reply(state, model_paths["cbt1"]),
            "cbt2": lambda: stream_cbt2_reply(state, model_paths["cbt2"]),
            "cbt3": lambda: stream_cbt3_reply(state, model_paths["cbt3"]),
        }

        if state.stage not in agent_streams:
            yield r"모든 세션이 종료되었습니다. 감사합니다.\n"
            return

        agent_gen = collect_stream(agent_streams[state.stage]())
        user_state_future = (
            asyncio.ensure_future(run_user_state_agent(state, model_path=model_paths["detect"], mode=mode))
            if mode != "plain" and model_paths.get("detect")
            else None
        )

        try:
            async for chunk in agent_gen:
                yield chunk
        except Exception:
            logger.exception("❌ 스트리밍 오류 발생")
            yield f"⚠️ 답변 생성 오류".encode("utf-8")
            return

        logger.info(f"⏱️ 응답 시간: {time.time() - start_time:.2f}초")

        user_state_result = await user_state_future if user_state_future else {}

        match = re.search(r'---END_STAGE---\n({.*})', full_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                next_stage = result.get("next_stage", state.stage)
                state.turn = result.get("turn", 0)
                state.history = result.get("history", [])
                state.response = result.get("response", "")
            except Exception:
                logger.warning("⚠️ 전이 파싱 실패")
                next_stage = state.stage
        else:
            logger.warning("⚠️ END_STAGE 태그 없음")
            next_stage = state.stage

        if user_state_result:
            if user_state_result.get("need_rollback"):
                next_stage = "mi"
                state.user_profile = None
            elif user_state_result.get("profile"):
                state.user_profile = user_state_result["profile"]

        result = {
            "next_stage": next_stage,
            "response": state.response.strip() or "답변 없음",
            "turn": state.turn,
            "history": state.history,
            "preset_questions": state.preset_questions,
            "drift_trace": state.drift_trace,
            "user_profile": state.user_profile or {}
        }
        logger.info(f"🔁 다음 단계: {next_stage.upper()} | 턴: {state.turn}")
        yield b"\n---END_STAGE---\n" + json.dumps(result, ensure_ascii=False).encode("utf-8")

    return StreamingResponse(async_gen(), media_type="text/plain")

async def dummy_loop():
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=True)
