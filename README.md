# 🧠 TTM 다중 에이전트 챗봇 서버

> FastAPI 기반 | 🩺 Transtheoretical Model (TTM) | 💬 다중 에이전트 | 🧪 Drift 감지 내장형 AI 챗봇

---

## 📌 프로젝트 개요

이 프로젝트는 TTM(Transtheoretical Model)을 기반으로 한 **다중 에이전트 챗봇 시스템**입니다.  
사용자의 변화 단계(stage)에 따라 다음과 같은 흐름으로 대화가 진행됩니다:

공감 (empathy) → 동기유도 (mi) → 자동사고 탐색 (cbt1) → 인지 재구성 (cbt2) → 과제 설정 (cbt3)

yaml
복사
편집

또한 대화 흐름 중 **drift 감지 모델**이 사용자의 이탈 여부를 자동으로 감지하고, 필요한 경우 **리셋 및 단계 재조정**을 수행합니다.

---

## 🧱 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| 백엔드 서버 | FastAPI |
| 모델 실행 | HuggingFace GGUF (로컬) |
| 감정/행동 처리 에이전트 | Empathy, MI, CBT1–3 |
| Drift 감지 | TinyLLaMA 기반 |
| 데이터 모델링 | Pydantic |
| 응답 방식 | StreamingResponse (비동기 스트리밍) |

---

## 🧠 지원 에이전트 구성

| 스테이지 | 설명 | 모델 경로 |
|----------|------|-----------|
| empathy | 사용자 감정 공감 | `youngbongbong/empathymodel` |
| mi | 동기 유도(MI) 질문 제공 | `youngbongbong/mimodel` |
| cbt1 | 자동사고 탐색 (CBT 1단계) | `youngbongbong/cbt1model` |
| cbt2 | 인지 재구성 (CBT 2단계) | `youngbongbong/cbt2model` |
| cbt3 | 실천 계획 및 긍정심리 (CBT 3단계) | `youngbongbong/cbt3model` |
| detect | Drift 감지용 모델 | `hieupt/TinyLlama-1.1B-Chat` |

---

## 🚀 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
export HUGGINGFACE_TOKEN=your_hf_token_here

# 3. 서버 실행
python main.py
혹은 uvicorn으로 직접 실행:

bash
복사
편집
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
📦 모델 다운로드 경로
서버가 시작되면 아래 Hugging Face 모델들이 자동으로 다운로드됩니다:

bash
복사
편집
/models/empathy/merged-empathy-8.0B-chat-Q4_K_M.gguf
/models/mi/merged-mi-chat-q4_k_m.gguf
/models/cbt1/merged-first-8.0B-chat-Q4_K_M.gguf
/models/cbt2/merged-mid-8.0B-chat-Q4_K_M.gguf
/models/cbt3/merged-cbt3-8.0B-chat-Q4_K_M.gguf
/models/detect/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf
🧪 Drift 평가 내장
서버 시작 시 자동으로 eval_drift.py의 evaluate_drift_detection() 함수가 실행됩니다.
그 결과는 콘솔에 다음과 같이 출력됩니다:

makefile
복사
편집
🏁 Drift Detection 최종 평가 결과 요약:
accuracy: 0.88
precision: 0.83
recall: 0.79
f1_score: 0.81
🌐 API 엔드포인트 요약
GET /
서버 상태 확인용 루트 엔드포인트

GET /status
모델이 정상적으로 다운로드되어 준비되었는지 확인

POST /chat/stream
실시간 스트리밍 응답을 반환하는 메인 대화 엔드포인트

📥 요청 예시 (JSON)
json
복사
편집
{
  "state": {
    "session_id": "user123",
    "stage": "cbt1",
    "question": "왜 자꾸 걱정을 반복하게 될까요?",
    "turn": 3,
    "history": [],
    "preset_questions": [],
    "drift_trace": [],
    "reset_triggered": false
  }
}
📤 응답 예시 (Streaming + 종료 JSON)
css
복사
편집
(대화 응답 스트리밍 중...)

---END_STAGE---
{
  "next_stage": "cbt1",
  "response": "지금 이런 생각이 드신 이유를 함께 살펴보면 어떨까요?",
  "turn": 4,
  "history": [...],
  "preset_questions": [...],
  ...
}
📁 디렉토리 구조
bash
복사
편집
TTMdriftcbtchat/
├── agents/                 # 각 단계별 에이전트 처리
├── drift/                 # Drift 감지 및 평가 모듈
├── llm/                   # 모델 로딩 및 매핑
├── shared/                # 상태 모델 정의
├── eval/                  # Drift 평가 로직
├── main.py                # FastAPI 엔트리 포인트
├── requirements.txt       # 의존성
└── README.md
⚠️ 예외 처리 및 리셋 로직
입력 오류: 상태 파싱 실패 시 공감 단계로 초기화

Drift 감지: reset_triggered = True일 경우 해당 응답으로 즉시 전환

존재하지 않는 stage: 종료 메시지 반환

👩‍💻 개발자 정보
이름: 윤소영

GitHub: @soyoung2004

📄 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

✨ 변화 단계 이론 기반의 AI 챗봇을 통해, 사용자의 감정·행동·인지적 변화를 맞춤형으로 지원합니다.
