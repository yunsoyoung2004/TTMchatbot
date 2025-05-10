from llama_cpp import Llama
import os, re, json
from typing import AsyncGenerator

# ✅ 중복 문장 제거
def deduplicate_streaming_text(text: str) -> str:
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    seen, result = set(), []
    for s in sentences:
        simplified = re.sub(r'\s+', '', s)
        if simplified and simplified not in seen:
            seen.add(simplified)
            result.append(s)
    return ' '.join(result)

# ✅ 말투 보정 및 짧은 응답 보완
def polish_sentence(text: str) -> str:
    if not re.search(r"(요|죠|네요|가요)[.?!]?$", text.strip()):
        text = text.strip() + " 괜찮으셨을까요?"
    if len(text.strip()) < 20:
        text += " 천천히 더 이야기해 주셔도 괜찮습니다."
    return text

# ✅ 공감형 응답 생성 (모델 경로를 외부에서 전달받음)
async def stream_empathy_reply(user_input: str, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = user_input.strip()

    if len(user_input) < 3:
        fallback = "조금만 더 이야기해 주실 수 있을까요?"
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + b'{"next_stage": "mi"}'
        return

    # ✅ 모델 동적 로드
    llm = Llama(
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

    system_prompt = (
        "당신은 따뜻하고 공감 잘하는 상담자입니다. "
        "사용자의 감정을 진심 어린 1~2문장으로 공감해 주세요. "
        "같은 말 반복 없이 자연스럽게 이어가며, 존댓말로 마무리해 주세요."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    full_response = ""
    first_token_sent = False

    try:
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full_response += token
                yield token.encode("utf-8")

        cleaned = deduplicate_streaming_text(full_response.strip())
        polished = polish_sentence(cleaned)

    except Exception as e:
        polished = "죄송합니다. 다시 말씀해 주실 수 있을까요?"
        yield f"\n⚠️ 오류가 발생했어요: {e}".encode("utf-8")

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": "mi",
        "response": polished
    }, ensure_ascii=False).encode("utf-8")
