import os, json, multiprocessing, difflib
from typing import AsyncGenerator, Literal, List, Optional, Tuple
from pydantic import BaseModel
from llama_cpp import Llama
from drift.detector import run_detect
import nltk

# ✅ NLTK 리소스 자동 다운로드
for resource in ["punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f'taggers/{resource}' if "tagger" in resource else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# ✅ DETECT 모델 캐시
LLM_DETECT_INSTANCE = {}

def load_detect_model(model_path: str) -> Llama:
    global LLM_DETECT_INSTANCE
    if model_path not in LLM_DETECT_INSTANCE:
        print(f"📦 DETECT 모델 로딩: {model_path}", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_DETECT_INSTANCE[model_path] = Llama(
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
    return LLM_DETECT_INSTANCE[model_path]

# ✅ 상태 모델
class AgentState(BaseModel):
    stage: Literal["cbt1", "cbt2"]
    question: str
    response: str
    history: List[str]
    turn: int
    drift_trace: List[Tuple[str, bool]] = []

# ✅ 프롬프트 생성
def get_detect_prompt(history: List[str]) -> str:
    joined = "\n".join(history[-8:])
    return (
        "다음은 상담자(챗봇)와 사용자 간의 대화 내용입니다.\n"
        "당신은 전문 심리상담가로서, 이 대화 흐름을 평가하여 사용자의 상태와 흐름 적합성을 요약하고,\n"
        "마지막 줄에 [예] 또는 [아니오]로 '상담 흐름 전환이 필요하다'고 판단되는지를 명시하세요.\n"
        "\n"
        "- 감정 표현 방식과 일관성\n"
        "- 주제 중심성\n"
        "- 실천/변화 의지\n"
        "- 흐름 적합성 (현재 상담 흐름이 적절한지)\n"
        "\n# 대화 내용:\n"
        f"{joined}\n"
        "\n# 상담자 요약:\n"
    )

# ✅ 사용자 상태 평가 수행
def evaluate_user_state(state: AgentState, model_path: str) -> Tuple[str, bool]:
    prompt = get_detect_prompt(state.history)
    model = load_detect_model(model_path)
    try:
        response_obj = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        response = response_obj.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"⚠️ 상태 평가 실패: {e}")
        return "요약 실패", False

    summary = response or "요약 실패"
    rollback = summary.endswith("예")
    return summary, rollback

# ✅ 실행 엔트리포인트
async def run_user_state_agent(state: AgentState, model_path: str, mode="drift_profile"):
    if mode == "plain":
        return {}

    drifted = run_detect(state)

    if mode == "drift_only":
        return {"enhanced": drifted}

    if drifted:
        summary, rollback = evaluate_user_state(state, model_path)
        print(f"[DRIFT-EVAL] 요약:\n{summary}\n→ MI로 전환 필요? {rollback}")
        return {
            "need_rollback": rollback,
            "summary": summary
        }

    return {}
