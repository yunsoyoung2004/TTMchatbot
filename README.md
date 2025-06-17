# 🧠 TTMchatbot (Persona Drift Detection Version)

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=8A2BE2&height=240&section=header&text=TTMchatbot%20💡%20AI%20with%20Persona%20Drift&fontSize=35&fontAlign=50" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Stage-Aware%20Chat-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Drift%20Detection-Enabled-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/LLM-HuggingFace-orange?style=for-the-badge&logo=huggingface" />
</p>

---

## 📌 프로젝트 소개

**TTMchatbot**은 Transtheoretical Model(TTM)을 기반으로 한 다단계 AI 챗봇 시스템입니다.  
사용자의 대화 흐름을 실시간 분석하여, 각 단계의 페르소나에서 벗어나는 **드리프트(이탈)**를 감지하고  
자동으로 상담 흐름을 **MI 단계로 전환**하는 고급 기능을 포함합니다.

---

## 🎯 핵심 기능

- ✅ **단계별 대화 흐름**: `Empathy → MI → CBT1 → CBT2 → CBT3`
- 💬 **StreamingResponse**: 실시간 자연어 응답 전송
- 🧠 **Persona Drift 감지**: 사용자의 응답이 현재 단계 페르소나와 어긋나면 탐지
- 🔁 **자동 단계 전이**: 드리프트 발생 시 MI 단계로 자동 리셋
- 📄 **대화 상태 추적**: `AgentState`를 통해 상태 기반 제어

---

## 📂 디렉토리 구조

```
TTMchatbot/
├── agents/          # 단계별 응답 함수
├── drift/           # 드리프트 탐지 로직
├── utils/           # 보조 유틸리티
├── main.py          # FastAPI 진입점
├── requirements.txt # 패키지 의존성
├── Dockerfile
└── README.md
```

---

## ⚙️ 설치 및 실행

```bash
# 1. 클론
git clone https://github.com/yunsoyoung2004/TTMchatbot.git
cd TTMchatbot

# 2. 패키지 설치
pip install -r requirements.txt

# 3. Hugging Face 토큰 설정
export HUGGINGFACE_TOKEN=hf_...

# 4. FastAPI 실행
python main.py
```

---

## 🔗 API 요약

### ✅ POST `/chat/stream`

```json
{
  "state": {
    "session_id": "user123",
    "stage": "cbt2",
    "question": "요즘 어떤 걱정이 있으신가요?",
    "response": "",
    "history": [],
    "turn": 2,
    "intro_shown": true
  }
}
```

### 응답 포맷

- 실시간 응답 조각(`chunk`) 스트리밍
- 마지막에 `---END_STAGE---` JSON 블록 포함:

```json
---END_STAGE---
{
  "next_stage": "mi",
  "turn": 0,
  "response": "...",
  "history": [...]
}
```

---

## 🧠 Drift 감지 예시

```python
if detect_persona_drift(state.stage, full_reply):
    yield "\n[시스템] 페르소나 드리프트 감지됨. MI 단계로 이동합니다.\n"
    state.stage = "mi"
    state.turn = 0
```

- 사용자 응답이 해당 단계의 목표와 어긋날 경우, 자동으로 `MI` 단계로 전환됩니다.

---

## 🛠 기술 스택

| 구분 | 내용 |
|------|------|
| Language | Python 3.10+ |
| Framework | FastAPI, StreamingResponse |
| LLM | Hugging Face Transformers, GGUF, llama.cpp |
| Drift Detection | 커스텀 probe 기반 탐지 로직 |
| 기타 | Docker, Pydantic, asyncio |

---

## 👩‍💻 개발자 정보

- **이름**: 윤소영 (SoYoung Yun)  
- 📧 **개인 이메일**: yunsoyoung2004@gmail.com  
- 📧 **학교 이메일**: thdud041113@g.skku.edu  
- 🔗 **GitHub**: [github.com/yunsoyoung2004](https://github.com/yunsoyoung2004)

---

> 💬 이 레포지토리는 **TTM 이론 기반** + **Drift 감지 기반 행동 전이**가 포함된  
> 완성도 높은 대화형 AI 시스템입니다. 상담 맥락에 따라 **자동 흐름 전환**이 가능한 구조입니다.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=8A2BE2&height=180&section=footer" />
</p>
