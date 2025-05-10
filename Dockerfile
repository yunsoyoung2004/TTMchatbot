FROM python:3.11-slim

# 시스템 패키지 업데이트 + 빌드 도구 설치 (gcc, g++, cmake 등)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 지정
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 기본 실행 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
