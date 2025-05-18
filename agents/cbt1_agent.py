from llama_cpp import Llama  
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing

# ✅ 페르소나 드리프트 감지
from utils.drift_detector import detect_persona_drift

LLM_CBT_INSTANCE = {}

def load_cbt_model(model_path: str) -> Llama:
    if model_path not in LLM_CBT_INSTANCE:
        print("🚀 CBT1 Llama 모델 로딩 중...")
        LLM_CBT_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_threads=max(1, multiprocessing.cpu_count() - 1),
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
    stage: Literal["cbt1", "cbt2", "cbt3", "mi"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

def get_turn_prompt(turn: int) -> str:
    prompts = [
        "최근 떠오른 감정이나 생각 중 가장 강렬했던 건 무엇이었나요?",
        "그 생각이 들게 된 이유나 근거는 무엇이라고 생각하시나요?",
        "그 생각이나 감정이 행동으로 이어진 적이 있었나요?",
        "그런 생각이 계속된다면 어떤 결과가 생길지 어떻게 예상하시나요?",
        "혹시 그 생각을 다른 방식으로 해석해 볼 수도 있을까요?"
    ]
    base = (
        "당신은 소크라테스형 CBT 상담자로, 사용자 스스로 감정과 사고를 탐색하도록 도와줍니다. "
        "정중하고 탐색적인 질문을 통해 사용자의 자동사고를 구조적으로 분석해 주세요.\n"
    )
    return base + prompts[min(turn, 4)]

async def stream_cbt_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    # ✅ 첫 인트로 안내
    if state.turn == 0 and not state.intro_shown:
        intro = (
            "안녕하세요. 지금부터는 최근의 감정, 생각, 행동 흐름을 함께 점검해보겠습니다. "
            "편하게 시작해볼까요? 최근 어떤 감정이나 생각이 가장 먼저 떠오르셨나요?"
        )
        updated_history = state.history + [intro]
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt1",
            "turn": 1,
            "response": intro,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")
        return

    # ✅ 질문 입력이 없을 경우 fallback
    if not user_input:
        fallback = [
            "최근 어떤 감정이나 생각이 가장 먼저 떠오르셨나요?",
            "그 생각이 왜 그렇게 들었는지, 이유가 무엇일까요?",
            "그런 감정이 행동에 어떤 영향을 미쳤을까요?",
            "그 생각을 계속 믿는다면 어떤 결과가 생길까요?",
            "그 생각을 다른 방식으로 해석할 수 있을까요?"
        ]
        fallback_msg = fallback[min(state.turn - 1, 4)]
        yield fallback_msg.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": fallback_msg,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [fallback_msg]
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt_model(model_path)

        messages = [{"role": "system", "content": get_turn_prompt(state.turn)}]
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

        reply = buffer.strip()

        # ✅ 응답 검증 및 정리
        if not reply.endswith(("다.", "요.", "죠?", "나요?", "까요?", "습니까?")):
            reply += " 이 부분에 대해 어떻게 생각하세요?"

        if state.history and reply == state.history[-1].strip():
            reply = "조금 다른 방식으로 다시 질문드려볼게요."

        # ✅ 드리프트 감지
        drifted = detect_persona_drift("cbt1", reply)
        if drifted:
            reply += "\n\n⚠️ 시스템 알림: CBT1 페르소나 일관성이 흐려졌습니다. MI 단계로 전환하겠습니다."
            next_stage = "mi"
            turn = 0
        else:
            next_turn = state.turn + 1
            next_stage = "cbt2" if next_turn >= 5 else "cbt1"
            turn = 0 if next_stage != "cbt1" else next_turn
            if next_stage == "cbt2":
                reply += "\n\n📘 좋습니다. 이제 다음 단계에서 인지 기술을 함께 연습해보겠습니다."

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": turn,
            "response": reply,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [user_input, reply]
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        err = f"⚠️ CBT1 응답 오류: {e}"
        yield err.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "turn": 0,
            "response": err,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history
        }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt_reply"]
