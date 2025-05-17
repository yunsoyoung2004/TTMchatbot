import os
import re
import json
from typing import AsyncGenerator
from llama_cpp import Llama

# ✅ 드리프트 감지 유틸
from utils.drift_detector import detect_persona_drift

LLM_INSTANCE = {}
TURN_COUNTER = {}

def get_turn_count(session_id: str) -> int:
    return TURN_COUNTER.get(session_id, 0)

def increment_turn_count(session_id: str) -> int:
    TURN_COUNTER[session_id] = TURN_COUNTER.get(session_id, 0) + 1
    return TURN_COUNTER[session_id]

def reset_turn_count(session_id: str):
    TURN_COUNTER.pop(session_id, None)

def load_llama_model(model_path: str) -> Llama:
    if model_path not in LLM_INSTANCE:
        LLM_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=256,
            n_threads=os.cpu_count(),
            n_batch=4,
            max_tokens=48,
            temperature=0.5,
            top_p=0.9,
            repeat_penalty=1.3,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>"]
        )
    return LLM_INSTANCE[model_path]

def deduplicate_streaming_text(text: str) -> str:
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    seen, result = set(), []
    for s in sentences:
        simplified = re.sub(r'\s+', '', s)
        if simplified and simplified not in seen:
            seen.add(simplified)
            result.append(s)
    return ' '.join(result)

def polish_sentence(text: str) -> str:
    if not re.search(r"(요|죠|네요|가요)[.?!]?$", text.strip()):
        text = text.strip() + " 괜찮으셨을까요?"
    if len(text.strip()) < 20:
        text += " 천천히 더 이야기해 주셔도 괜찮습니다."
    return text

EMPATHY_TEMPLATES = {
    0: "사용자에게 다정하게 안부 인사를 건네고 최근 기분을 묻습니다.",
    1: "사용자의 말에서 감정을 포착하고 명확히 언어화하며 공감합니다.",
    2: "사용자의 정서를 받아들이고 감정을 반영하며 지지해 주세요.",
    3: "사용자와의 신뢰를 쌓으며 관계 형성을 도와주는 따뜻한 응답을 생성합니다.",
    4: "지금까지 나눈 이야기를 정리하고 다음 단계로 자연스럽게 넘어갈 수 있게 도와줍니다."
}

async def stream_empathy_reply(user_input: str, model_path: str, session_id: str) -> AsyncGenerator[bytes, None]:
    user_input = user_input.strip()
    if len(user_input) < 3:
        fallback = "조금만 더 이야기해 주실 수 있을까요?"
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + b'{"next_stage": "mi"}'
        return

    try:
        llm = load_llama_model(model_path)
        turn = increment_turn_count(session_id)

        if turn >= 5:
            yield b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": "mi",
                "response": "지금까지 이야기 나눠 주셔서 감사합니다."
            }, ensure_ascii=False).encode("utf-8")
            reset_turn_count(session_id)
            return

        print(f"🔁 공감 턴 {turn + 1} / 5")

        stage_prompt = EMPATHY_TEMPLATES.get(turn, "공감하는 대화를 이어가 주세요.")
        messages = [
            {"role": "system", "content": f"너는 따뜻한 공감 상담자야. {stage_prompt}"},
            {"role": "user", "content": user_input}
        ]

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

        cleaned = deduplicate_streaming_text(full_response.strip())
        polished = polish_sentence(cleaned)

        # ✅ 드리프트 감지
        if detect_persona_drift("empathy", polished):
            warning = "⚠️ 시스템 감지: 상담자의 공감 말투가 흐트러졌습니다. MI 단계로 전환합니다."
            yield b"\n" + warning.encode("utf-8")
            yield b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": "mi",
                "response": polished + "\n\n" + warning
            }, ensure_ascii=False).encode("utf-8")
            reset_turn_count(session_id)
            return

        if turn == 4:
            yield b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": "mi",
                "response": polished
            }, ensure_ascii=False).encode("utf-8")
            reset_turn_count(session_id)
        else:
            yield b"\n---CONTINUE_STAGE---\n" + json.dumps({
                "next_stage": "empathy",
                "response": polished
            }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        error_msg = f"\n⚠️ 오류가 발생했어요: {e}"
        yield error_msg.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": "죄송합니다. 다시 말씀해 주실 수 있을까요?"
        }, ensure_ascii=False).encode("utf-8")
