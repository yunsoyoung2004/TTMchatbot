from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, List, AsyncGenerator
import json, os, multiprocessing, asyncio

# ✅ 상태 정의
class AgentState(BaseModel):
    stage: Literal["s_turn", "cbt"]
    question: str
    response: str
    history: List[str]
    turn: int
    retry_count: int = 0

# ✅ 고정된 소크라테스 질문 리스트
SOCRATIC_QUESTIONS = [
    "약물을 사용할 때 생기는 생각의 근거는 무엇인가요?",
    "혹시 이 상황을 다르게 해석할 수 있는 방법은 없을까요?",
    "약물을 사용할 때의 생각이 사실이라고 믿는다면, 어떤 결과가 생길까요?",
    "과거에도 이런 경험을 한 적이 있다면, 그때는 어떻게 대처하셨나요?",
    "이 생각이 당신 삶의 중요한 가치나 목표에 어떤 영향을 줄까요?"
]

# ✅ 소크라테스식 스트리밍 응답 생성기
async def stream_s_turn_reply(state: AgentState, model_path: str) -> AsyncGenerator[str, None]:
    user_input = state.question.strip()
    history = state.history or []
    turn = state.turn

    # 0턴: 인트로 메시지
    if turn == 0 and "[s_turn → intro]" not in history:
        intro = (
            "안녕하세요. 지금부터는 '사고 탐색(Socratic Questioning)'을 통해 "
            "당신의 생각을 이해해보는 시간을 가질게요.\n"
        )
        for ch in intro:
            yield ch
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        yield "\n---END_STAGE---\n" + json.dumps({
            "next_stage": "s_turn",
            "turn": 1,
            "question": "",
            "response": intro.strip(),
            "history": history + [user_input, intro.strip(), "[s_turn → intro]"]
        }, ensure_ascii=False)
        return

    # 턴 수 초과 시 초기화
    if turn < 1 or turn > 6:
        warn = "S-TURN 세션을 다시 시작할게요. 사고 탐색 질문을 처음부터 진행하겠습니다.\n"
        for ch in warn:
            yield ch
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        yield "\n---END_STAGE---\n" + json.dumps({
            "next_stage": "s_turn",
            "turn": 1,
            "question": "",
            "response": warn.strip(),
            "history": history + [user_input, warn.strip(), "[s_turn → 강제 초기화]"]
        }, ensure_ascii=False)
        return

    # 1~5턴: 고정된 질문
    if 1 <= turn <= 5:
        question = SOCRATIC_QUESTIONS[turn - 1]
        prompt = f"❓ {question}\n"
        for ch in prompt:
            yield ch
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        yield "\n---END_STAGE---\n" + json.dumps({
            "next_stage": "s_turn",
            "turn": turn + 1,
            "question": "",
            "response": question,
            "history": history + [user_input, question]
        }, ensure_ascii=False)
        return

    # 6턴: CBT 단계로 전환
    cbt_msg = (
        "🧠 지금까지 사고를 잘 정리해주셨어요.\n"
        "📘 이제 사고를 구체적으로 재구성하는 CBT 단계로 넘어가겠습니다.\n"
    )
    for ch in cbt_msg:
        yield ch
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    yield "\n---END_STAGE---\n" + json.dumps({
        "next_stage": "cbt",
        "turn": 0,
        "question": "",
        "response": cbt_msg.strip(),
        "history": history + [user_input, cbt_msg.strip(), "[s_turn 완료 → cbt 전환]"]
    }, ensure_ascii=False)

# ✅ FastAPI 앱 (예: 테스트용 별도 실행 시)
app = FastAPI()

@app.post("/chat/s_turn")
async def chat_s_turn_stream(request: Request):
    data = await request.json()
    state = AgentState(**data["state"])

    async def wrapped_generator():
        async for chunk in stream_s_turn_reply(state, model_path="dummy"):
            yield chunk.encode("utf-8")

    return StreamingResponse(wrapped_generator(), media_type="text/plain")
