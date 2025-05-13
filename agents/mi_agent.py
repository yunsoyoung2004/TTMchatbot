import os, json, re, difflib, multiprocessing
from typing import AsyncGenerator, Literal, List, Optional
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ 전역 모델 캐시
LLM_MI_INSTANCE = {}

# ✅ 모델 로더 (model_path 기반)
def load_mi_model(model_path: str) -> Llama:
    global LLM_MI_INSTANCE
    if model_path not in LLM_MI_INSTANCE:
        print("🚀 MI 모델 최초 로딩 중...")
        LLM_MI_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=256,
            n_threads=max(1, multiprocessing.cpu_count() - 1),
            n_batch=4,
            max_tokens=64,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            repeat_penalty=1.1,
            frequency_penalty=0.7,
            presence_penalty=0.5,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>", "\n\n"]
        )
        print("✅ MI 모델 로딩 완료")
    return LLM_MI_INSTANCE[model_path]

# ✅ 상태 정의
class AgentState(BaseModel):
    stage: Literal["mi", "s_turn"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    pending_response: Optional[str] = None

# ✅ 시스템 프롬프트
def get_system_prompt(turn: int) -> str:
    base = (
        "당신은 공감적이고 따뜻한 MI 상담자입니다. "
        "사용자의 감정을 존중하며, 부드러운 존댓말로 1~2문장 이내로 응답해 주세요. "
        "상대를 자극하거나 단정하지 말고, 변화 가능성에 대한 열린 질문으로 이끌어 주세요. "
        "죽음/극단적 표현은 사용하지 마세요. 반드시 '?'로 끝나는 존댓말 질문 포함."
    )
    return base + (
        " 공감 1문장 + 상황 유도 질문 1문장으로 구성해 주세요." if turn < 2
        else " 공감 1문장 + 변화 탐색 질문 1문장으로 구성해 주세요."
    )

# ✅ 말투 보정
def normalize_politeness(text: str) -> str:
    if not re.search(r"(요|죠|가요|네요)[.?!]?$", text.strip()):
        return text.strip() + " 괜찮으셨을까요?"
    return text.strip()

# ✅ 중복 응답 필터
def is_redundant_response(new_text: str, history: List[str], threshold: float = 0.92) -> bool:
    past = history[-1::-2]
    return any(
        difflib.SequenceMatcher(None, new_text.strip(), h.strip()).ratio() > threshold
        for h in past
    )

# ✅ 위험 표현 감지
def is_risky(text: str) -> bool:
    return any(p in text for p in ["죽고", "자살", "극단적", "생을", "없어진", "끝내"])

# ✅ 변화 질문 여부 확인
def is_valid_change_question(text: str) -> bool:
    keywords = ["바꾸", "변화", "시작", "시도", "벗어나", "노력", "가능성", "도전"]
    return any(k in text for k in keywords) and text.strip().endswith("?")

# ✅ fallback 질문
def get_fallback_question(turn: int) -> str:
    fallback = [
        "최근 가장 바꾸고 싶은 부분은 무엇인가요?",
        "지금 이 순간에 시도해보고 싶은 변화가 있을까요?",
        "스스로 느끼는 가장 큰 변화 욕구는 무엇인가요?",
        "그 감정은 어떤 변화를 원하나요?",
        "지금 가능한 작은 변화는 어떤 걸까요?"
    ]
    return fallback[turn % len(fallback)]

# ✅ MI 응답 생성 함수
async def stream_mi_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "👋 안녕하세요. 저는 감정을 함께 탐색하는 MI 상담자입니다.\n"
            "지금 어떤 상황에서 이런 생각이 드셨는지부터 편하게 이야기해 주세요."
        )
        updated_history = state.history + [intro]
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "turn": 0,
            "response": intro,
            "intro_shown": True,
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input:
        msg = "편하게 이야기 이어가 주셔도 괜찮습니다."
        yield msg.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": msg
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_mi_model(model_path)

        messages = [{"role": "system", "content": get_system_prompt(state.turn)}]
        if len(state.history) >= 4:
            messages.append({"role": "user", "content": state.history[-2]})
            messages.append({"role": "assistant", "content": state.history[-1]})
        messages.append({"role": "user", "content": user_input})

        full_response = ""
        first_token_sent = False

        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip()

        if is_risky(reply) or len(reply) < 15 or is_redundant_response(reply, state.history):
            reply = get_fallback_question(state.turn)

        reply = normalize_politeness(reply)

        if state.turn >= 2 and not is_valid_change_question(reply):
            reply = get_fallback_question(state.turn)

        updated_history = state.history + [user_input, reply]
        next_stage = "s_turn" if state.turn + 1 >= 8 else "mi"

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": state.turn + 1,
            "response": reply,
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        err = f"⚠️ 응답 오류가 발생했어요: {e}"
        yield err.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": err
        }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_mi_reply"]
