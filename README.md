<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6A5ACD&height=230&section=header&text=TTM-Based%20Multi-Agent%20Chatbot&fontAlign=50&fontSize=35&animation=fadeIn" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Powered%20By-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-HuggingFace-orange?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Persona%20Drift%20Detection-Enabled-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Stage-Aware%20Chat-9cf?style=for-the-badge" />
</p>

---

## 🧠 TTMchatbot 소개

> **TTMchatbot**은 행동 변화 이론인 Transtheoretical Model(TTM)을 기반으로 설계된  
> **다단계 인공지능 상담 챗봇 시스템**입니다. 사용자의 대화 흐름을 분석하여  
> **Empathy → MI → CBT1 → CBT2 → CBT3**로 유연하게 전이하며,  
> 각 단계는 Hugging Face 기반 LLM으로 구동됩니다.

- 🎯 **Stage-Aware Architecture**  
  각 단계별로 독립적인 모델과 스트리밍 응답을 지원

- 🧠 **Persona Drift Detection**  
  사용자의 반응이 각 단계의 페르소나와 어긋날 경우, 자동으로 `MI 단계`로 전이

- 🔁 **LLM Streamed Replies**  
  단계별 응답은 `StreamingResponse`로 실시간 생성됨

---

## 📂 프로젝트 구조

```bash
TTMchatbot/
├── agents/             # 단계별 에이전트 (empathy, mi, cbt1, cbt2, cbt3)
├── drift/              # 페르소나 드리프트 탐지 로직
├── llm/                # 모델 로딩 및 통합 유틸리티
├── models/             # (예시) 모델 저장 디렉토리
├── offload/            # 모델 offloading 유틸리티
├── shared/             # 공통 상수 및 타입
├── utils/              # 드리프트 평가 함수
├── main.py             # FastAPI 엔트리포인트
├── Dockerfile          # 배포용 도커파일
└── requirements.txt    # Python 패키지 목록
