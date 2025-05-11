FROM python:3.11-slim-bullseye

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential cmake gcc g++ git curl pkg-config python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ 루트 디렉토리를 작업 디렉토리로 설정
WORKDIR /

# 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 모델 다운로드
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
print('📥 MI 모델:', hf_hub_download('youngbongbong/mimodel', 'merged-mi-chat-q4_k_m.gguf', token='${HUGGINGFACE_TOKEN}')); \
print('📥 CBT 모델:', hf_hub_download('youngbongbong/cbtmodel', 'merged-cbt-chat-q4_k_m.gguf', token='${HUGGINGFACE_TOKEN}')); \
print('📥 PPI 모델:', hf_hub_download('youngbongbong/ppimodel', 'merged-ppi-prep-chat-q4_k_m.gguf', token='${HUGGINGFACE_TOKEN}'))"

# 전체 소스 복사
COPY . .

# ✅ main.py는 루트에 있으므로 그대로
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
