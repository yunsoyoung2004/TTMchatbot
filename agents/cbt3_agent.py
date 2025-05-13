from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing

LLM_CBT_INSTANCE = {}

def load_cbt_model(model_path: str) -> Llama:
    global LLM_CBT_INSTANCE
    if model_path not in LLM_CBT_INSTANCE:
        print("🚀 CBT3 Llama 모델 로딩 중...")
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
        print(f"✅ CBT3 모델 로딩 완료: {model_path}")
    return LLM_CBT_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["cbt3", "action"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "이제 마지막 단계입니다. 이번 주에 실천할 수 있는 과제를 함께 정하고,"
            "예상되는 방해요소나 고위험 상황에 대한 대처 계획도 세워볼 거예요."
        )
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt3",
            "turn": 1,
            "response": intro,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [intro]
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input:
        fallback = "이번 주에 어떤 행동을 실천해볼 수 있을까요?"
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
                """너는 약물중독 CBT 후반부 상담자야. 지금은 과제 설정과 실행계획을 세우는 시간이고, 다음 흐름을 따라 대화를 구성해:
1) 맞춤형 과제 목표 설정
2) 구체적 실천 계획(언제/어디서/어떻게)
3) 예상 방해요인 및 해결책 토의
4) 고위험 상황에 대한 시뮬레이션 및 대처전략
말 끝을 존댓말로 하고, 실제 일상에 적용 가능한 실천 질문을 포함해줘.
예: '언제 이걸 실천해보면 좋을까요?', '방해가 된다면 어떤 대안을 써볼 수 있을까요?'"""
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
        yield f"⚠️ CBT3 응답 오류: {e}".encode("utf-8")
        return

    reply = buffer.strip()
    if not reply.endswith(("다.", "요.", "죠?", "나요?", "까요?", "습니까?")):
        reply += " 이 계획이 현실적으로 가능할까요?"

    if state.history and reply == state.history[-1].strip():
        reply = "같은 주제로 조금 더 구체적으로 계획해볼까요?"

    next_turn = state.turn + 1
    next_stage = "action" if next_turn >= 6 else "cbt3"
    if next_stage == "action":
        reply += "\n\n📘 잘하셨어요. 이제 실천을 위한 다음 단계로 넘어가요."

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": next_stage,
        "turn": 0 if next_stage != "cbt3" else next_turn,
        "response": reply,
        "question": "",
        "intro_shown": True,
        "awaiting_s_turn_decision": False,
        "history": state.history + [user_input, reply]
    }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt3_reply"]
