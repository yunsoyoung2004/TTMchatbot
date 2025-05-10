from llama_cpp import Llama
import os, re, json
from typing import AsyncGenerator

LLM_INSTANCE = None

def load_llama_model(model_path: str):
    global LLM_INSTANCE
    if LLM_INSTANCE is None:
        print("🚀 Llama 모델 최초 로딩 중...")
        LLM_INSTANCE = Llama(
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
        print("✅ Llama 모델 로딩 완료")
    return LLM_INSTANCE

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

async def stream_empathy_reply(user_input: str, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = user_input.strip()
    print(f"🟡 사용자 입력 수신: '{user_input}'")

    if len(user_input) < 3:
        fallback = "조금만 더 이야기해 주실 수 있을까요?"
        print("⚠️ 입력이 너무 짧아 fallback 응답 전송")
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + b'{"next_stage": "mi"}'
        return

    try:
        print("🔄 모델 로딩 시작")
        llm = load_llama_model(model_path)
        print("✅ 모델 로딩 완료")

        system_prompt = (
            "당신은 따뜻하고 공감 잘하는 상담자입니다. "
            "사용자의 감정을 진심 어린 1~2문장으로 공감해 주세요. "
            "같은 말 반복 없이 자연스럽게 이어가며, 존댓말로 마무리해 주세요."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        print("🧠 메시지 구성 완료 → 모델 호출 시작")

        full_response = ""
        first_token_sent = False

        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            print(f"📤 토큰 수신 중: '{token}'")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        print(f"🧹 전체 응답 정제 전: '{full_response.strip()}'")
        cleaned = deduplicate_streaming_text(full_response.strip())
        polished = polish_sentence(cleaned)
        print(f"✨ 정제된 최종 응답: '{polished}'")

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": polished
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        error_msg = f"\n⚠️ 오류가 발생했어요: {e}"
        yield error_msg.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": "죄송합니다. 다시 말씀해 주실 수 있을까요?"
        }, ensure_ascii=False).encode("utf-8")

