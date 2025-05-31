import os, json, multiprocessing, re, asyncio
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel, Field
from llama_cpp import Llama

LLM_CBT3_INSTANCE = {}

def load_cbt3_model(model_path: str) -> Llama:
    global LLM_CBT3_INSTANCE
    if model_path not in LLM_CBT3_INSTANCE:
        print("🚀 CBT3 모델 최초 로딩 중...", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT3_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=NUM_THREADS,
            n_batch=4,
            max_tokens=128,
            temperature=0.65,
            top_p=0.9,
            presence_penalty=1.0,
            frequency_penalty=0.8,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>", "\n\n"]
        )
    return LLM_CBT3_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int
    preset_questions: List[str] = Field(default_factory=list)
    drift_trace: List = Field(default_factory=list)

# ✅ 시스템 프롬프트

def get_cbt3_prompt(enhanced=False) -> str:
    prompt = (
        "당신은 따뜻하고 논리적인 CBT 상담자입니다.\n"
        "- 사용자의 실천 계획을 탐색하는 질문을 한 문장으로 생성해 주세요.\n"
        "- 질문은 명확하고 구체적이어야 하며, 실제 행동을 유도해야 합니다.\n"
        "- 방해 요인, 감정 변화, 습관 형성, 환경 조정, 자기 피드백 등에 초점을 맞추어 주세요."
    )
    if enhanced:
        prompt += (
            "\n- 최근 대화가 반복되거나 방향이 모호했습니다. \n"
            "더 구체적인 실천 계획을 끌어낼 수 있도록 해주세요."
        )
    return prompt

# ✅ CBT3 멀티턴 응답 생성기
async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    try:
        llm = load_cbt3_model(model_path)
        enhanced = any(s == "cbt3" and d for s, d in getattr(state, "drift_trace", [])[-5:])
        prompt = get_cbt3_prompt(enhanced)

        messages = [{"role": "system", "content": prompt}]

        for i in range(0, len(state.history), 2):
            if i + 1 < len(state.history):
                messages.append({"role": "user", "content": state.history[i]})
                messages.append({"role": "assistant", "content": state.history[i + 1]})

        messages.append({"role": "user", "content": state.question})

        full_response = ""
        first_token_sent = False

        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            await asyncio.sleep(0.015)
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip()
        if not reply.endswith("?"):
            reply = reply.split(".")[0].strip() + "?"

        state.response = reply
        updated_history = state.history + [state.question, reply]
        next_turn = state.turn + 1
        next_stage = "end" if next_turn >= 5 else "cbt3"

        if next_stage == "end":
            end_msg = "\n\n🎯 실천 계획을 잘 정리해주셨어요. 이제 오늘 대화를 마무리할게요."
            for ch in end_msg:
                yield ch.encode("utf-8")
                await asyncio.sleep(0.015)

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": next_turn if next_stage != "end" else 0,
            "response": reply,
            "history": updated_history,
            "preset_questions": state.preset_questions
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT3 오류 발생: {e}", flush=True)
        fallback = "죄송해요. 지금은 잠시 오류가 발생했어요. 다시 이야기해 주시겠어요?"
        state.response = fallback
        for ch in fallback:
            yield ch.encode("utf-8")
            await asyncio.sleep(0.02)
