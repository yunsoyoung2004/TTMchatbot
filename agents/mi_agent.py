import os, json, multiprocessing
from typing import AsyncGenerator, Literal, List, Tuple
from pydantic import BaseModel
from llama_cpp import Llama

LLM_MI_INSTANCE = {}

# ✅ 모델 로딩 함수
def load_mi_model(model_path: str) -> Llama:
    global LLM_MI_INSTANCE
    if model_path not in LLM_MI_INSTANCE:
        try:
            print("\U0001F680 MI 모델 로딩 중...", flush=True)
            LLM_MI_INSTANCE[model_path] = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=max(1, multiprocessing.cpu_count() - 1),
                n_batch=4,
                max_tokens=128,
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
            print("✅ MI 모델 로드 완료", flush=True)
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}", flush=True)
            raise RuntimeError("MI 모델 로딩 실패")
    return LLM_MI_INSTANCE[model_path]

# ✅ 상태 정의
class AgentState(BaseModel):
    question: str
    response: str
    history: List[str]
    drift_trace: List[Tuple[str, bool]] = []

# ✅ 시스템 프롬프트 생성기
def get_mi_prompt(context="empathy", enhanced=False) -> str:
    if context == "empathy":
        prompt = (
            "당신은 공감적이고 지지적인 상담자입니다.\n"
            "- 사용자의 감정과 어려움을 공감하면서, 문제 인식을 도와주세요.\n"
            "- 사용자가 자신의 상황을 되돌아볼 수 있도록 질문해주세요.\n"
            "- 예: '지금 가장 힘든 점은 무엇인가요?', '마음속에서 어떤 감정이 오가고 있나요?'"
        )
    else:
        prompt = (
            "당신은 양가감정을 다루는 상담자입니다.\n"
            "- 사용자는 변화의 필요성을 인식했지만, 주저하고 있습니다.\n"
            "- 망설임, 피로감, 실패 경험 등의 감정을 다루고, 실천을 향한 미세한 동기를 탐색하세요.\n"
            "- 예: '변화를 생각할 때 어떤 부담이 드시나요?', '무엇이 망설이게 하나요?', '이전 시도에서 무엇이 어려웠나요?'"
        )
    if enhanced:
        prompt += "\n- 최근 대화 흐름이 반복되었거나 방향이 모호했습니다. 질문을 더 구체적으로 해주세요."
    return prompt

# ✅ MI 스트리밍 응답 생성기
async def stream_mi_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if not user_input or len(user_input) < 2:
        fallback = "조금 더 구체적으로 말씀해주실 수 있을까요?"
        state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": fallback,
            "history": state.history + [user_input, fallback],
            "drift_trace": state.drift_trace
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_mi_model(model_path)

        # ✅ 문맥 설정
        context = "cbt" if any(s.startswith("cbt") for s, _ in state.drift_trace[-3:]) else "empathy"
        enhanced = any(s == "mi" and drift for s, drift in state.drift_trace[-5:])

        # ✅ 멀티턴 메시지 구성
        messages = [{"role": "system", "content": get_mi_prompt(context, enhanced)}]
        history_pairs = list(zip(state.history[::2], state.history[1::2]))[-5:]
        for user_msg, assistant_msg in history_pairs:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_input})

        # ✅ 스트리밍 응답
        full_response, first_token_sent = "", False
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip() or "괜찮아요. 마음을 천천히 들려주셔도 괜찮습니다."
        state.response = reply

        turn_count = len(state.history) // 2
        next_stage = "cbt1" if turn_count + 1 >= 5 else "mi"

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "response": reply,
            "history": state.history + [user_input, reply],
            "drift_trace": state.drift_trace
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}", flush=True)
        fallback = "죄송합니다. 잠시 문제가 발생했어요. 다시 한 번 말씀해 주시겠어요?"
        state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": fallback,
            "history": state.history + [user_input, fallback],
            "drift_trace": state.drift_trace
        }, ensure_ascii=False).encode("utf-8")
