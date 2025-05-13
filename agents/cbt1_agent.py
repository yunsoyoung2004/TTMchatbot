from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing

# 회피 모델 캐시
LLM_CBT_INSTANCE = {}

def load_cbt_model(model_path: str) -> Llama:
    global LLM_CBT_INSTANCE
    if model_path not in LLM_CBT_INSTANCE:
        print("🚀 CBT1 Llama 모델 로딩 중...")
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_threads=NUM_THREADS,
            n_ctx=512,
            n_batch=8,
            max_tokens=96,
            temperature=0.5,
            top_p=0.85,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["User:", "Assistant:"]
        )
        print(f"✅ CBT1 모델 로딩 완료: {model_path}")
    return LLM_CBT_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["cbt1", "cbt2", "cbt3", "action"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

async def stream_cbt1_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "안녕하세요. 저는 최근 사용 경험과 그에 대한 감정을 함께 살펴보는 상담자입니다."
            "편하게 최근 있었던 일이나 기분을 나눠주세요."
        )
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt1",
            "turn": 1,
            "response": intro,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [intro]
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input:
        fallback = "생각나는 부분부터 편하게 말씀해 주세요."
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": fallback
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt_model(model_path)
        messages = [
            {"role": "system", "content": (
                """너는 약물중독 CBT1 초기 세션 전문 상담자야. 다음의 목적을 기억하고 대화를 이끌어:
1) 최근 사용/갈망 여부 탐색, 2) 고위험 상황 탐색, 3) 사전 감정/사고 흐름 점검.
공감하며 존댓말로 묻고, 다음 질문 유형을 적절히 섞어 사용해:
- Q1. '그 생각의 근거는 무엇이었나요?'
- Q4. '그 생각을 계속 믿으면 어떤 결과가 생길까요?'
- 탐색형: '그 일 이후 무슨 일이 있었나요?', '그때 어떤 감정이 가장 먼저 들었나요?'
모든 질문은 1문단 내외로 구성하고, 존댓말로 마무리해줘."""
            )}
        ]
        if len(state.history) >= 2:
            messages.append({"role": "user", "content": state.history[-2]})
            messages.append({"role": "assistant", "content": state.history[-1]})
        messages.append({"role": "user", "content": user_input})

        buffer = ""
        first_token_sent = False

        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                buffer += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

    except Exception as e:
        yield f"⚠️ CBT1 응답 오류: {e}".encode("utf-8")
        return

    reply = buffer.strip()
    if not reply.endswith(("다.", "요.", "죠?", "나요?", "까요?", "습니까?")):
        reply += " 어떤 점이 가장 기억에 남으셨나요?"

    if state.history and reply == state.history[-1].strip():
        reply = "조금 다른 관점에서 다시 떠올려볼 수 있을까요?"

    next_turn = state.turn + 1
    next_stage = "cbt2" if next_turn >= 6 else "cbt1"
    if next_stage == "cbt2":
        reply += "\n\n📘 감사합니다. 이제 다음 단계로 넘어가 보겠습니다."

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": next_stage,
        "turn": 0 if next_stage != "cbt1" else next_turn,
        "response": reply,
        "question": "",
        "intro_shown": True,
        "awaiting_s_turn_decision": False,
        "history": state.history + [user_input, reply]
    }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt1_reply"]
