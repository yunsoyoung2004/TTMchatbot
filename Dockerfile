FROM python:3.11-slim-bullseye

# ✅ 시스템 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    curl \
    pkg-config \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ 설치 확인 (디버깅용)
RUN which gcc && gcc --version && which g++

# ✅ CMake가 사용할 컴파일러를 명시적으로 설정
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# ✅ pip 최신화
RUN pip install --upgrade pip setuptools wheel

# ✅ 작업 디렉토리 설정
WORKDIR /app

# ✅ requirements 설치 (pyproject.toml 빌드 포함 패키지 대응)
COPY requirements.txt .
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# ✅ 앱 코드 복사
COPY . .

# ✅ FastAPI 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
