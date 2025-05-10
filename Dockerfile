FROM python:3.11-slim

# ✅ 빌드 도구 설치 + 컴파일러 환경변수 설정
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    gcc \
    g++ \
    && apt-get clean

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# ✅ pip 업그레이드
RUN pip install --upgrade pip

WORKDIR /app

# ✅ requirements.txt 설치 (핵심 수정: --prefer-binary)
COPY requirements.txt .
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# ✅ 나머지 앱 복사
COPY . .

# ✅ 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
