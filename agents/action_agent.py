from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing, difflib, re

# ✅ 전역 모델 인스턴스 캐싱
LLM_PPI_INSTANCE = {}

def load_ppi_model(model_path: str) -> Llama:
    global LLM_PPI_INSTANCE
    if model_path not in LLM_PPI_INSTANCE:
        print("🚀 PPI 모델 최초 로딩 중...")
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_PPI_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=384,
            n_threads=NUM_THREADS,
            n_batch=8,
            max_tokens=64,
            temperature=0.6,
            top_p=0.85,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["User:", "Assistant:"]
        )
        print(f"✅ PPI 모델 로딩 완료: {model_path}")
    return LLM_PPI_INSTANCE[model_path]

# ✅ 상태 모델 정의
class AgentState(BaseModel):
    stage: Literal["ppi", "action"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_preparation_decision: bool = False
    pending_response: Optional[str] = None

# ✅ 중복 응답 판단
def is_redundant_response(new_text: str, history: List[str], threshold: float = 0.92) -> bool:
    past = history[-1::-2]
    return any(
        difflib.SequenceMatcher(None, new_text.strip(), h.strip()).ratio() > threshold
        for h in past
    )

# ✅ 말투 보정
def normalize_politeness(text: str) -> str:
    if not re.search(r"(요|죠|가요|네요)[.?!]?$", text):
        return text.strip() + " 괜찮으신가요?"
    return text.strip()

# ✅ fallback
def get_fallback_plan(turn: int) -> str:
    fallback = [
        "하루 10분이라도 본인을 칭찬하는 시간을 가져보는 건 어떨까요? 이 계획에 대해 어떻게 생각하세요?",
        "감사한 일을 기록해보는 루틴을 만들어보시는 건 어떠세요? 이 계획에 대해 어떻게 생각하세요?",
        "작은 성공 경험을 적어보는 것으로 시작해도 좋아요. 이 계획에 대해 어떻게 생각하세요?"
    ]
    return fallback[turn % len(fallback)]

# ✅ PPI 응답 생성
async def stream_ppi_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    # ✅ 첫 인트로 메시지
    if state.turn == 0 and not state.intro_shown:
        intro = (
            "📘 안녕하세요. 저는 강점 기반 실천을 돕는 긍정 심리 코치입니다.\n"
            "지금 떠오르는 변화 계획이나 시도하고 싶은 점이 있다면 자유롭게 이야기해 주세요."
        )
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "ppi",
            "turn": 1,
            "response": intro,
            "intro_shown": True,
            "awaiting_preparation_decision": False,
            "history": state.history + [intro]
        }, ensure_ascii=False).encode("utf-8")
        return

    # ✅ 입력 없을 때 fallback
    if not user_input:
        fallback = "떠오르는 아이디어나 하고 싶은 변화가 있으시면 편하게 말씀해 주세요."
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": fallback
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm_ppi = load_ppi_model(model_path)

        instruction = (
            "당신은 긍정 심리 기반 실천 코치입니다. "
            "사용자의 강점과 의지를 바탕으로 정중한 존댓말로 2~3문장으로 실현 가능한 작은 실천 계획을 제안하세요. "
            "실천 계획 뒤에는 항상 '이 계획에 대해 어떻게 생각하세요?'라는 질문으로 마무리하세요.\n\n"
        )

        messages = [{"role": "user", "content": instruction + f"사용자 입력: {user_input}"}]
        if len(state.history) >= 4:
            messages.insert(0, {"role": "user", "content": state.history[-2]})
            messages.insert(1, {"role": "assistant", "content": state.history[-1]})

        full_response = ""
        first_token_sent = False

        for chunk in llm_ppi.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip()

        if is_redundant_response(reply, state.history) or len(reply) < 15:
            reply = get_fallback_plan(state.turn)

        reply = normalize_politeness(reply)

        next_stage = "action" if state.turn + 1 >= 3 else "ppi"
        updated_history = state.history + [user_input, reply]

        if next_stage == "action":
            reply += "\n\n👍 실천 계획을 잘 정리해 주셨어요. 이제 실제 행동에 옮겨볼 준비가 되셨다면 함께 계획을 세워봐요!"

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": 0 if next_stage == "action" else state.turn + 1,
            "response": reply,
            "intro_shown": True,
            "awaiting_preparation_decision": False,
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        err = f"⚠️ PPI 응답 오류가 발생했어요: {e}"
        yield err.encode("utf-8")

__all__ = ["stream_ppi_reply"]
