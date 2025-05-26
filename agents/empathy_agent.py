import os, json
from typing import AsyncGenerator, Optional
from llama_cpp import Llama
from agents.schema import AgentState

LLM_INSTANCE = {}

# ✅ 모델 로딩
def load_llama_model(model_path: str, cache_key: str) -> Llama:
    global LLM_INSTANCE
    if cache_key not in LLM_INSTANCE:
        try:
            print(f"🚀 모델 로딩 시작: {cache_key}", flush=True)
            LLM_INSTANCE[cache_key] = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=os.cpu_count(),
                n_batch=4,
                max_tokens=64,
                temperature=0.6,
                top_p=0.9,
                repeat_penalty=1.1,
                n_gpu_layers=0,
                low_vram=True,
                use_mlock=False,
                verbose=False,
                chat_format="llama-3",
                stop=["<|im_end|>"]
            )
            print(f"✅ Llama 로딩 완료: {model_path}", flush=True)
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}", flush=True)
            raise RuntimeError("모델 로딩 중 문제가 발생했습니다.")
    return LLM_INSTANCE[cache_key]

# ✅ 시스템 프롬프트
def get_empathy_prompt() -> str:
    return (
        "당신은 따뜻하고 진심 어린 공감 상담자입니다.\n"
        "사용자는 이별, 상실, 고통, 외로움 같은 정서적인 문제를 이야기할 수 있으며,\n"
        "당신은 항상 존댓말로 부드럽고 진심 어린 말투로 응답해야 합니다.\n"
        "1~2문장으로 너무 길지 않게 말해주세요. 지나친 위로나 조언보다는, 감정에 귀 기울이는 반응이 중심이어야 합니다.\n"
        "절대 명령하거나 단정 짓지 말고, 사용자의 감정을 인정하고 함께 있어주는 따뜻한 친구처럼 이야기해주세요.\n"
        "모든 응답은 반드시 한국어로 출력하세요."
    )

# ✅ 공감 응답 생성기
async def stream_empathy_reply(
    question: str,
    model_path: str,
    turn: int = 0,
    state: Optional[AgentState] = None
) -> AsyncGenerator[bytes, None]:
    user_input = question.strip()
    print(f"🟡 사용자 입력 수신: '{user_input}' (턴 {turn})", flush=True)

    if turn == 0:
        greeting = "안녕하세요. 만나서 반가워요. 혹시 제가 뭐라고 불러드리면 좋을까요?"
        if state:
            state.response = greeting
        yield greeting.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "empathy",
            "response": greeting,
            "turn": 1,
            "intro_shown": True,
            "history": [user_input, greeting]
        }, ensure_ascii=False).encode("utf-8")
        return

    if len(user_input) < 3:
        fallback = "지금 어떤 마음이신지 조금 더 이야기해 주실 수 있으실까요?"
        if state:
            state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "empathy",
            "response": fallback,
            "turn": turn,
            "intro_shown": True,
            "history": [user_input, fallback]
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_llama_model(model_path, "empathy")
        messages = [
            {"role": "system", "content": get_empathy_prompt()},
            {"role": "user", "content": user_input}
        ]

        full_response = ""
        first_token_sent = False

        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip()
        if not reply or len(reply) < 2:
            reply = "괜찮아요. 지금 이 순간 어떤 마음이신지 천천히 들려주세요."

        if state:
            state.response = reply

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi" if turn >= 2 else "empathy",
            "response": reply,
            "turn": 0 if turn >= 2 else turn + 1,
            "intro_shown": True,
            "history": [user_input, reply]
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ stream_empathy_reply 예외 발생: {e}", flush=True)
        fallback = "죄송합니다. 잠시 오류가 있었어요. 다시 말씀해 주실 수 있을까요?"
        if state:
            state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "empathy",
            "response": fallback,
            "turn": turn,
            "intro_shown": True,
            "history": [user_input, fallback]
        }, ensure_ascii=False).encode("utf-8")
