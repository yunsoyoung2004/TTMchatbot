FROM python:3.11-slim

# 시스템 패키지 설치 (libcurl 추가됨)
RUN apt-get update && apt-get install -y \
    build-essential cmake git g++ curl pkg-config python3-dev \
    libcurl4-openssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# llama.cpp 최신 CMake 빌드 (CPU-only)
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama && \
    cd /llama && mkdir build && cd build && cmake .. -DLLAMA_CUBLAS=OFF && make -j

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 설치
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# llama-cpp-python 설치
RUN pip install llama-cpp-python --no-cache-dir

# 앱 소스 복사
COPY . .

# 포트 오픈
EXPOSE 8080

# FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
