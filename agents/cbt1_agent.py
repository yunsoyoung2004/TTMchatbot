import os, json, multiprocessing, difflib 
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ CBT1 모델 캐시
LLM_CBT1_INSTANCE = {}

def load_cbt1_model(model_path: str) -> Llama:
    global LLM_CBT1_INSTANCE
    if model_path not in LLM_CBT1_INSTANCE:
        print(f"📦 CBT1 모델 로딩: {model_path}", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT1_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=NUM_THREADS,
            n_batch=8,
            max_tokens=128,
            temperature=0.95,
            top_p=0.92,
            presence_penalty=1.4,
            frequency_penalty=1.2,
            repeat_penalty=1.3,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>"]
        )
    return LLM_CBT1_INSTANCE[model_path]

# ✅ 상태 모델
class AgentState(BaseModel):
    stage: Literal["cbt1", "cbt2"]
    question: str
    response: str
    history: List[str]
    turn: int

def get_cbt1_prompt(enhanced=False) -> str:
    prompt = (
        "당신은 자동사고를 탐색하는 따뜻하고 이성적인 CBT 상담자입니다.\n"
        "- 사용자의 말을 기반으로 자동사고를 도와주시고, 항상 다른 관점에서 질문해 주세요.\n"
        "- 반드시 한 문장 또는 두 문장으로 마무리해 주세요.\n"
        "- 같은 표현, 말투, 어미, 문장 구조, 단어를 반복하지 말고, 매번 다르게 표현해 주세요.\n"
        "- 감정, 근거, 장기적 결과, 타인의 시각, 반복된 패턴, 예외적 상황 등 다양한 각도에서 질문해 주세요.\n"
        "- 예시: '그때 가장 강하게 느낀 감정은 무엇이었나요?', '그 생각을 계속 믿으면 어떤 영향이 생길까요?', '이전과 비슷한 상황이 반복된 적 있나요?'"
    )
    if enhanced:
        prompt += (
            "\n- 최근 대화 흐름이 반복되었거나 방향이 모호했습니다. 이전보다 더욱 구체적이고 이전과는 다른 시각을 제공하는 질문을 시도해 보세요."
        )
    return prompt

# ✅ CBT1 응답 스트리밍 함수
async def stream_cbt1_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()
    history = state.history or []

    print(f"🧠 [CBT1 현재 턴: {state.turn}]")

    if not user_input:
        fallback = "떠오른 생각이나 감정이 있다면 편하게 이야기해 주세요."
        state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt1",
            "turn": state.turn,
            "response": fallback,
            "question": "",
            "history": history
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt1_model(model_path)
        enhanced = any(s == "cbt1" and d for s, d in getattr(state, "drift_trace", [])[-5:])
        system_prompt = get_cbt1_prompt(enhanced)
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                messages.append({"role": "user", "content": history[i]})
                messages.append({"role": "assistant", "content": history[i + 1]})
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

        reply = full_response.strip() or "좋아요. 조금 더 구체적으로 이야기해주실 수 있을까요?"
        state.response = reply

        for past in history[-10:]:
            if isinstance(past, str):
                if difflib.SequenceMatcher(None, reply[:40], past[:40]).ratio() > 0.8:
                    reply += " 그랬군요, 그게 정말 사실일까요? 왜곡되지는 않았나요?"
                    break

        next_turn = state.turn + 1
        next_stage = "cbt2" if next_turn >= 5 else "cbt1"

        updated_history = history.copy()
        if not (len(updated_history) >= 2 and updated_history[-2] == user_input and updated_history[-1] == reply):
            updated_history.extend([user_input, reply])

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": 0 if next_stage == "cbt2" else next_turn,
            "response": reply,
            "question": "",
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT1 오류: {e}", flush=True)
        fallback = "죄송해요. 다시 말씀해 주시겠어요?"
        state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt1",
            "turn": state.turn,
            "response": fallback,
            "question": "",
            "history": history
        }, ensure_ascii=False).encode("utf-8")
