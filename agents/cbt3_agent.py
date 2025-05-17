from llama_cpp import Llama
from typing import Literal, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os, json, multiprocessing, random

# ✅ 페르소나 드리프트 감지 유틸
from utils.drift_detector import detect_persona_drift

LLM_CBT_INSTANCE = {}

def load_cbt_model(model_path: str) -> Llama:
    global LLM_CBT_INSTANCE
    if model_path not in LLM_CBT_INSTANCE:
        print("🚀 CBT3 Llama 모델 로딩 중...")
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_threads=NUM_THREADS,
            n_ctx=512,
            n_batch=8,
            max_tokens=96,
            temperature=0.5,
            top_p=0.85,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["User:", "Assistant:"]
        )
        print(f"✅ CBT3 모델 로딩 완료: {model_path}")
    return LLM_CBT_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["cbt3", "action", "mi"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    awaiting_s_turn_decision: bool
    pending_response: Optional[str] = None

async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if state.turn == 0 and not state.intro_shown:
        intro = (
            "이제 마지막 단계입니다. 이번 주에 실천할 수 있는 과제를 함께 정하고, "
            "예상되는 방해요소나 고위험 상황에 대한 대처 계획도 세워볼 거예요."
        )
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt3",
            "turn": 1,
            "response": intro,
            "question": "",
            "intro_shown": True,
            "awaiting_s_turn_decision": False,
            "history": state.history + [intro]
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input:
        fallback = "이번 주에 어떤 행동을 실천해볼 수 있을까요?"
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": state.stage,
            "turn": state.turn,
            "response": fallback
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt_model(model_path)

        predefined_questions = {
            3: "친한 친구가 같은 고민을 한다면, 뭐라고 해줄까요?",
            5: "앞으로 내가 할 수 있는 일은 무엇일까요?"
        }
        if state.turn in predefined_questions:
            user_input += f"\n\n{predefined_questions[state.turn]}"

        ppi_prompts = [
            "요즘 감사한 일이 있었나요?",
            "나의 강점 중 이번 과제에 도움이 될 수 있는 건 뭘까요?",
            "과거에 비슷한 상황에서 잘했던 경험이 있다면요?"
        ]
        if state.turn in [2, 4]:
            user_input += f"\n\n{random.choice(ppi_prompts)}"

        messages = [
            {"role": "system", "content": (
                "당신은 숙련된 CBT 코치입니다. 지금은 사용자의 행동 변화 실천을 돕는 단계이며, 다음 흐름을 따라 대화를 설계하세요:\n"
                "1) 맞춤형 과제 목표 설정\n"
                "2) 구체적 실천 계획 (언제/어디서/어떻게)\n"
                "3) 예상 방해요인 및 대처 전략 수립\n"
                "4) 고위험 상황 시뮬레이션\n"
                "5) 제삼자 조언 질문, 행동 지침 질문\n"
                "6) 긍정 심리 개입 요소 삽입\n"
                "답변은 현실적이고 격려하는 코칭 말투로 작성해주세요."
            )}
        ]

        if len(state.history) >= 2:
            messages.append({"role": "user", "content": state.history[-2]})
            messages.append({"role": "assistant", "content": state.history[-1]})
        messages.append({"role": "user", "content": user_input})

        buffer = ""
        first_token_sent = False
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                buffer += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

    except Exception as e:
        yield f"⚠️ CBT3 응답 오류: {e}".encode("utf-8")
        return

    reply = buffer.strip()

    if not reply.endswith(("다.", "요.", "죠?", "나요?", "까요?", "습니까?")):
        reply += " 이 계획이 현실적으로 가능할까요?"

    if state.history and reply == state.history[-1].strip():
        reply = "같은 주제로 조금 더 구체적으로 계획해볼까요?"

    drifted = detect_persona_drift("cbt3", reply)
    if drifted:
        reply += "\n\n⚠️ 시스템 경고: 코치의 말투가 일관되지 않았습니다. MI 단계로 전환합니다."
        next_stage = "mi"
        turn = 0
    else:
        next_turn = state.turn + 1
        next_stage = "action" if next_turn >= 6 else "cbt3"
        turn = 0 if next_stage != "cbt3" else next_turn
        if next_stage == "action":
            reply += "\n\n📘 실천을 위한 준비가 완료되었습니다. 수고하셨습니다!"

    yield b"\n---END_STAGE---\n" + json.dumps({
        "next_stage": next_stage,
        "turn": turn,
        "response": reply,
        "question": "",
        "intro_shown": True,
        "awaiting_s_turn_decision": False,
        "history": state.history + [user_input, reply]
    }, ensure_ascii=False).encode("utf-8")

__all__ = ["stream_cbt3_reply"]
