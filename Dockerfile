FROM gcr.io/ttmchatbotbot/ttmchatbot:latest

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    g++ \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Hugging Face 캐시 경로
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install llama-cpp-python==0.3.8 \
        --no-cache-dir \
        --config-settings=cmake.define.LLAMA_CUBLAS=OFF

# 전체 소스 복사
COPY . .

# 포트 노출
EXPOSE 8080

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
