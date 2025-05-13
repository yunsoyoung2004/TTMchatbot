from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing

LLM_CBT_INSTANCE = {}

def load_cbt_model(model_path: str) -> Llama:
    global LLM_CBT_INSTANCE
    if model_path not in LLM_CBT_INSTANCE:
        print("🚀 CBT2 Llama 모델 로딩 중...")
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
        print(f"✅ CBT2 모델 로딩 완료: {model_path}")
    return LLM_CBT_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["cbt2", "cbt3", "action"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

async def stream_cbt2_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "이번 단계에서는 갈망 대처나 자기통제 같은 기술들을 연습해볼 거예요."
            "최근 있었던 상황 중 하나를 떠올리며 함께 적용해볼까요?"
        )
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt2",
            "turn": 1,
            "response": intro,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [intro]
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input:
        fallback = "최근 갈망을 느꼈던 상황이 있다면 이야기해보셔도 좋아요."
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
                """너는 약물중독 CBT 중반부 기술훈련 세션을 맡은 상담자야.
다음의 흐름에 따라 대화를 구성해:
1) 오늘 다룰 기술 주제 연결 (예: 갈망 대처)
2) 최근 사건과 기술 연결
3) 역할극이나 구체적 상황 설정
4) 기술 이해도 점검
다정하고 격려하는 톤으로, 역할극 대화도 포함시켜줘.
예: '같이 연습해볼까요?', '그 기술을 어떻게 적용하면 좋을까요?'
반드시 존댓말로 1문단 이내로 마무리해줘."""
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
        yield f"⚠️ CBT2 응답 오류: {e}".encode("utf-8")
        return

    reply = buffer.strip()
    if not reply.endswith(("다.", "요.", "죠?", "나요?", "까요?", "습니까?")):
        reply += " 혹시 지금 떠오르는 상황이 있으신가요?"

    if state.history and reply == state.history[-1].strip():
        reply = "그 상황을 조금 다르게 설정해서 다시 연습해볼까요?"

    next_turn = state.turn + 1
    next_stage = "cbt3" if next_turn >= 6 else "cbt2"
    if next_stage == "cbt3":
        reply += "\n\n📘 훈련이 잘 되셨어요. 다음 단계로 넘어가 보겠습니다."

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": next_stage,
        "turn": 0 if next_stage != "cbt2" else next_turn,
        "response": reply,
        "question": "",
        "intro_shown": True,
        "awaiting_s_turn_decision": False,
        "history": state.history + [user_input, reply]
    }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt2_reply"]
