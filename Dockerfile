FROM python:3.11-slim-bullseye

# 시스템 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential cmake gcc g++ git curl pkg-config python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Hugging Face 인증 토큰 환경 변수 등록
ENV HUGGINGFACE_TOKEN=hf_your_token_here

# pip 최신화 및 의존성 설치
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 미리 다운로드
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='youngbongbong/mimodel', filename='merged-mi-chat-q4_k_m.gguf', token='$HUGGINGFACE_TOKEN')"

RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='youngbongbong/cbtmodel', filename='merged-cbt-chat-q4_k_m.gguf', token='$HUGGINGFACE_TOKEN')"

RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='youngbongbong/ppimodel', filename='merged-ppi-prep-chat-q4_k_m.gguf', token='$HUGGINGFACE_TOKEN')"

# 앱 디렉토리 복사
WORKDIR /app
COPY . .

# 포트 8080 리슨
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
