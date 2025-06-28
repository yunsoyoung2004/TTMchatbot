# 🤖 TTM Multi-Agent Chatbot Server

TTM(Transtheoretical Model) 기반 멀티 에이전트 챗봇 서버입니다.
각 에이전트는 사용자의 단계(stage)에 맞게 공감(Empathy), 동기강화(MI), 인지치료(CBT1\~3)을 제공합니다.
또한 Drift Detection 기능을 통해 대화 흐름 이탈을 자동 감지하고 조정합니다.

---

## 📌 주요 기능

* 🧠 단계별 에이전트 (Empathy → MI → CBT1 → CBT2 → CBT3)
* 💬 스트리밍 응답 처리 (StreamingResponse)
* 🧭 드리프트 탐지 및 자동 리셋 (Drift Detection)
* 💾 Hugging Face 모델 자동 다운로드
* 📡 RESTful API 제공 (FastAPI 기반)

---

## 🗂️ 폴더 구조

📦 project-root
├── main.py                  # FastAPI 서버 진입점
├── agents/                  # 각 단계별 에이전트 처리 모듈
│   ├── empathy\_agent.py
│   ├── mi\_agent.py
│   ├── cbt1\_agent.py
│   ├── cbt2\_agent.py
│   └── cbt3\_agent.py
├── llm/                     # 모델 로딩 및 실행 관리
│   ├── loader.py
│   ├── agent.py
│   └── stage\_map.py
├── shared/                  # 상태 모델, 유틸 함수 등
│   └── types.py
├── eval/                    # 드리프트 탐지 평가 스크립트
│   └── eval\_drift.py
├── requirements.txt         # 의존성 패키지 목록
└── README.md

---

## ⚙️ 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

### 3. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## 🔗 API 명세

### `/chat/stream` (POST)

사용자의 질문과 상태를 포함한 요청을 보내면, 해당 단계에 맞는 에이전트가 스트리밍 응답을 반환합니다.

#### ✅ 요청 예시

POST /chat/stream
Content-Type: application/json

```json
{
  "state": {
    "session_id": "abc123",
    "stage": "cbt2",
    "question": "전 뭘 해도 실패할 것 같아요",
    "history": [],
    "reset_triggered": false
  }
}
```

#### 🔁 응답 예시

"그렇게 느낄 수 있어요. 하지만 실패는 배움의 일부예요."
\---END\_STAGE---

* 응답은 Streaming 형식입니다.
* `---END_STAGE---`는 응답 완료 시 표시됩니다.

---

## 🧠 사용 모델

모든 모델은 서버 실행 시 자동 다운로드됩니다.
(사전 `HUGGINGFACE_TOKEN` 환경변수 필요)

| Stage   | Hugging Face 모델 경로                            |
| ------- | --------------------------------------------- |
| empathy | youngbongbong/empathymodel                    |
| mi      | youngbongbong/mimodel                         |
| cbt1    | youngbongbong/cbt1model                       |
| cbt2    | youngbongbong/cbt2model                       |
| cbt3    | youngbongbong/cbt3model                       |
| detect  | hieupt/TinyLlama-1.1B-Chat-v1.0-Q4\_K\_M-GGUF |

---

## 📊 Drift Detection 평가

서버 초기화 시 `evaluate_drift_detection()` 함수가 실행되어
`eval/benchmark.json` 파일 기반으로 Drift 탐지 성능을 평가하고 콘솔에 로그를 출력합니다.

지표:

* Precision
* Recall
* F1-score

---

## 💡 추가 팁

* CORS 오류 시 `main.py`의 `add_middleware()` 설정 확인
* 모델이 준비되지 않았다는 메시지가 뜬다면 `/status`로 상태 확인 후 재시작
* 응답 흐름에 맞춰 `---END_STAGE---` 태그를 클라이언트에서 처리하세요

---
