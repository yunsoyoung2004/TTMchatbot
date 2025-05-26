import os, json, multiprocessing, difflib, re, asyncio
from typing import AsyncGenerator, List
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ 모델 캐시
LLM_CBT2_INSTANCE = {}

def load_cbt2_model(model_path: str) -> Llama:
    global LLM_CBT2_INSTANCE
    if model_path not in LLM_CBT2_INSTANCE:
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT2_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=NUM_THREADS,
            n_batch=4,
            max_tokens=128,
            temperature=0.65,
            top_p=0.9,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>", "---END_STAGE---"]
        )
    return LLM_CBT2_INSTANCE[model_path]

class AgentState(BaseModel):
    question: str
    response: str
    history: List[str]
    turn: int

def get_cbt2_prompt(enhanced=False) -> str:
    prompt = (
        "당신은 인지 재구조화를 도와주는 따뜻한 CBT 상담자입니다.\n"
        "- 사용자의 자동사고를 잘 이해하고, 그것을 탐색할 수 있도록 질문을 하나만 해주세요.\n"
        "- 반드시 질문은 하나만 해주세요.\n"
        "- 사용자의 말을 반복하거나 그대로 흉내내지 말아주세요.\n"
        "- 같은 문장을 반복하지 말고, 다양한 표현을 사용해 주세요."
    )
    if enhanced:
        prompt += (
            "\n- 최근 대화 흐름이 반복되거나 자동사고 탐색이 흐릿했어요. "
            "새로운 시선에서 질문을 던져 사용자가 자신의 자동사고를 더 깊이 바라볼 수 있게 해주세요."
        )
    return prompt

# ✅ 중복 질문 필터링
def is_similar_to_past_response(reply: str, history: List[str]) -> bool:
    recent_responses = [h for i, h in enumerate(history[-10:]) if i % 2 == 1]
    for past in recent_responses:
        ratio = difflib.SequenceMatcher(None, reply[:50], past[:50]).ratio()
        if ratio > 0.8:
            return True
    return False

def contains_user_echo(reply: str, user_input: str) -> bool:
    norm = lambda s: re.sub(r'\s+', '', s.lower())
    return norm(user_input) in norm(reply)

# ✅ 스트리밍 응답 함수
async def stream_cbt2_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()
    history = state.history if state.turn > 0 else []

    if not user_input or re.fullmatch(r"[ㅋㅎㅠㅜ]+", user_input):
        fallback = "조금 더 구체적으로 이야기해주실 수 있을까요?"
        state.response = fallback
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt2",
            "turn": state.turn,
            "response": fallback,
            "question": "",
            "history": history
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt2_model(model_path)
        enhanced = any(s == "cbt2" and d for s, d in getattr(state, "drift_trace", [])[-5:])
        system_prompt = get_cbt2_prompt(enhanced)
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                messages.append({"role": "user", "content": history[i]})
                messages.append({"role": "assistant", "content": history[i + 1]})
        messages.append({"role": "user", "content": user_input})

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

        full_response = full_response.strip()
        first_sentence = re.split(r"[.?!]", full_response)[0].strip()
        if not first_sentence.endswith("?"):
            first_sentence += "?"

        first_sentence = first_sentence.replace("preset_questions", "").replace("{", "").replace("}", "")

        if is_similar_to_past_response(first_sentence, history) or contains_user_echo(first_sentence, user_input):
            first_sentence += " 이 생각은 어디서 비롯된 걸까요?"

        state.response = first_sentence

        next_turn = state.turn + 1
        next_stage = "cbt3" if next_turn >= 5 else "cbt2"
        updated_history = history + [user_input, first_sentence]

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": 0 if next_stage == "cbt3" else next_turn,
            "response": first_sentence,
            "question": "",
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        import traceback
        traceback.print_exc()
        fallback = "죄송해요. 다시 한 번 이야기해주시겠어요?"
        state.response = fallback
        for ch in fallback:
            yield ch.encode("utf-8")
            await asyncio.sleep(0.02)
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt2",
            "turn": state.turn,
            "response": fallback,
            "question": "",
            "history": history
        }, ensure_ascii=False).encode("utf-8")

