FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential cmake git g++ curl pkg-config python3-dev \
    libcurl4-openssl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 추가 설치
RUN pip install --no-cache-dir huggingface_hub hf_transfer nltk

# NLTK 리소스 다운로드
RUN python -m nltk.downloader vader_lexicon punkt averaged_perceptron_tagger

# 소스 코드 복사
COPY . .

# 포트 설정
EXPOSE 8080

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]
