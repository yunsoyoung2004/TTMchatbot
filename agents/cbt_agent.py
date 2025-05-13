from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing

# ✅ 전역 모델 인스턴스 캐시
LLM_CBT_INSTANCE = {}

# ✅ 모델 로딩 함수 (model_path 기반)
def load_cbt_model(model_path: str) -> Llama:
    global LLM_CBT_INSTANCE
    if model_path not in LLM_CBT_INSTANCE:
        print("🚀 CBT Llama 모델 최초 로딩 중...")
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_threads=NUM_THREADS,
            n_ctx=384,
            n_batch=8,
            max_tokens=48,
            temperature=0.5,
            top_p=0.85,
            repeat_penalty=1.05,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["User:", "Assistant:"]
        )
        print(f"✅ CBT 모델 로딩 완료: {model_path}")
    return LLM_CBT_INSTANCE[model_path]

# ✅ 상태 모델
class AgentState(BaseModel):
    stage: Literal["cbt", "action"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

# ✅ CBT 스트리밍 응답 함수
async def stream_cbt_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "👋 안녕하세요. 저는 사고를 재구성하는 CBT 상담자입니다. "
            "지금 떠오르는 생각이나 그 배경을 편하게 이야기해 주세요."
        )
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt",
            "turn": 1,
            "response": intro,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [intro]
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input:
        fallback = "생각나는 부분을 편하게 말씀해 주세요."
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": fallback
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm_cbt = load_cbt_model(model_path)

        messages = [
            {"role": "system", "content": (
                "너는 CBT 상담자야. 사용자의 비합리적인 사고를 따뜻하게 재구성하는 1~2문장 질문을 해. "
                "반드시 한 문단으로 끝내고, 모든 문장은 존댓말로 마무리해."
            )}
        ]
        if len(state.history) >= 2:
            messages.append({"role": "user", "content": state.history[-2]})
            messages.append({"role": "assistant", "content": state.history[-1]})
        messages.append({"role": "user", "content": user_input})

        buffer = ""
        first_token_sent = False

        for chunk in llm_cbt.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                buffer += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

    except Exception as e:
        error_msg = f"⚠️ CBT 응답 오류: {e}"
        yield error_msg.encode("utf-8")
        return

    reply = buffer.strip()
    if not reply.endswith(("다.", "요.", "죠?", "나요?", "까요?", "습니까?")):
        reply += " 이 부분에 대해 어떻게 생각하시나요?"

    if state.history and reply == state.history[-1].strip():
        reply = "조금 다른 관점에서 다시 생각해볼 수 있을까요?"

    next_turn = state.turn + 1
    next_stage = "action" if next_turn >= 8 else "cbt"
    if next_stage == "action":
        reply += "\n\n📘 사고를 잘 정리해 주셨어요. 이제 실천 계획을 함께 세워볼까요?"

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": next_stage,
        "turn": 0 if next_stage == "action" else next_turn,
        "response": reply,
        "question": "",
        "intro_shown": True,
        "awaiting_s_turn_decision": False,
        "history": state.history + [user_input, reply]
    }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt_reply"]
