# MI Agent with 5-Turn Structure + Drift Detection
import os, json, re, difflib, multiprocessing  
from typing import AsyncGenerator, Literal, List, Optional
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ 드리프트 감지 유틸 추가
from utils.drift_detector import detect_persona_drift

LLM_MI_INSTANCE = {}

def load_mi_model(model_path: str) -> Llama:
    if model_path not in LLM_MI_INSTANCE:
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
    return LLM_MI_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["mi", "cbt1"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    pending_response: Optional[str] = None

def get_system_prompt(turn: int) -> str:
    prompts = [
        "지금 상황에서 바꾸고 싶은 점이나 어려움은 어떤 게 있을까요?",
        "한편으로는 그대로 두고 싶은 마음도 있으셨을까요?",
        "이런 양가감정은 어떤 감정에서 비롯된 걸까요?",
        "그럼에도 불구하고 시도해보고 싶은 변화는 있으신가요?",
        "그 변화가 실제로 일어났다면 어떤 느낌일까요?"
    ]
    base = (
        "당신은 따뜻하고 공감적인 MI 상담자입니다. "
        "사용자의 감정을 존중하며, 2문장 이내로 존댓말로 응답하세요. "
        "반드시 마지막 문장은 질문이어야 하며, 극단적 표현은 피하세요. "
    )
    return base + prompts[min(turn, 4)]

def normalize_politeness(text: str) -> str:
    if not re.search(r"(요|죠|가요|네요)[.?!]?$", text.strip()):
        return text.strip() + " 괜찮으셨을까요?"
    return text.strip()

def is_redundant_response(new_text: str, history: List[str], threshold: float = 0.92) -> bool:
    past = history[-1::-2]
    return any(
        difflib.SequenceMatcher(None, new_text.strip(), h.strip()).ratio() > threshold
        for h in past
    )

def is_risky(text: str) -> bool:
    return any(p in text for p in ["죽고", "자살", "극단적", "생을", "없어진", "끝내"])

def get_fallback_question(turn: int) -> str:
    return [
        "최근 가장 바꾸고 싶은 점은 어떤 건가요?",
        "한편으로는 그대로 두고 싶은 마음도 있으셨나요?",
        "그런 감정은 어디서 시작되었을까요?",
        "지금 떠오르는 가장 작은 변화는 무엇일까요?",
        "그 변화가 생겼다면 어떤 기분이 드셨을까요?"
    ][turn % 5]

async def stream_mi_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "👋 안녕하세요. 저는 변화 동기를 함께 탐색하는 MI 상담자입니다.\n"
            "최근 어떤 점을 바꾸고 싶다고 느끼셨는지, 편하게 이야기해 주세요."
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
        msg = "편하게 이어서 말씀해 주셔도 괜찮습니다."
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
            messages.extend([
                {"role": "user", "content": state.history[-2]},
                {"role": "assistant", "content": state.history[-1]}
            ])
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

        # 예외 처리
        if is_risky(reply) or len(reply) < 15 or is_redundant_response(reply, state.history):
            reply = get_fallback_question(state.turn)

        reply = normalize_politeness(reply)

        # ✅ 드리프트 감지
        drifted = detect_persona_drift("mi", reply)
        if drifted:
            reply += "\n\n⚠️ 시스템 경고: MI 상담자의 말투가 일관되지 않았습니다. 다시 감정 탐색 단계로 돌아갑니다."
            next_stage = "mi"
            turn = 0
        else:
            next_stage = "cbt1" if state.turn + 1 >= 5 else "mi"
            turn = 0 if next_stage == "cbt1" else state.turn + 1

        updated_history = state.history + [user_input, reply]

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": turn,
            "response": reply,
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        err = f"⚠️ 오류가 발생했어요: {e}"
        yield err.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": err
        }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_mi_reply"]
