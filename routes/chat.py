from fastapi import APIRouter
from models.message import ChatRequest, ChatResponse
from llm.agent import route_message

chat_router = APIRouter()

@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = await route_message(request.message, request.stage)
    return ChatResponse(response=response)