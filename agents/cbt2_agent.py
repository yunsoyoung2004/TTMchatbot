from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing

# ✅ 페르소나 드리프트 감지
from utils.drift_detector import detect_persona_drift

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
    stage: Literal["cbt2", "cbt3", "mi", "action"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

def get_cbt2_system_prompt() -> str:
    return (
        "당신은 소크라테스형 CBT 중반부 상담자입니다. "
        "사용자가 자동사고와 인지 왜곡을 스스로 인식하고 정리할 수 있도록 질문 위주로 대화를 이끌어 주세요.\n"
        "1) 자동사고 도전 및 재구성\n"
        "2) 인지 왜곡 탐색 및 근거 도전\n"
        "3) 대안적 사고 탐색 (검증된 믿음, 다른 해석, 효과 평가)\n"
        "4) 새로운 사고 적용 연습\n\n"
        "말투는 정중하되 질문은 명확하고 사고를 자극하도록 구성해 주세요."
    )

async def stream_cbt2_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    # ✅ 첫 턴 도입 멘트
    if state.turn == 0 and not state.intro_shown:
        intro = (
            "이제부터는 우리가 자동사고와 인지 왜곡을 함께 탐색해볼 거예요. "
            "최근에 떠올랐던 생각 중 반복되거나 강하게 느껴졌던 생각이 있다면 알려주시겠어요?"
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

    # ✅ 빈 질문 대응
    if not user_input:
        fallback = "최근 떠올랐던 반복적인 생각이나 걱정이 있었다면 말씀해 주세요."
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": fallback
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt_model(model_path)

        messages = [{"role": "system", "content": get_cbt2_system_prompt()}]
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
        reply += " 어떻게 생각하세요?"

    if state.history and reply == state.history[-1].strip():
        reply = "이번에는 조금 다른 관점에서 다시 질문해볼게요."

    # ✅ 드리프트 감지
    drifted = detect_persona_drift("cbt2", reply)
    if drifted:
        reply += "\n\n⚠️ 시스템 알림: 교사 역할이 불안정해졌습니다. MI 단계로 돌아가겠습니다."
        next_stage = "mi"
        turn = 0
    else:
        next_turn = state.turn + 1
        next_stage = "cbt3" if next_turn >= 10 else "cbt2"
        turn = 0 if next_stage != "cbt2" else next_turn
        if next_stage == "cbt3":
            reply += "\n\n📘 아주 잘 하셨어요. 이제 마지막 단계에서 실천 계획을 세워보겠습니다."

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": next_stage,
        "turn": turn,
        "response": reply,
        "question": "",
        "intro_shown": True,
        "awaiting_s_turn_decision": False,
        "history": state.history + [user_input, reply]
    }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt2_reply"]
