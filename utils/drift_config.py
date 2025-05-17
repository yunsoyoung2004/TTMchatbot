# utils/drift_config.py

# ✅ 드리프트 감지 점수 임계값 (0~1 사이, 높을수록 민감도 낮음)
DRIFT_THRESHOLD = 0.60  # 한국어 특성상 약간 낮춰 민감도 향상

# ✅ 각 피처별 가중치 (합계 1.0 유지)
FEATURE_WEIGHTS = {
    "repetition": 0.20,         # 반복률
    "teen_slang": 0.30,         # 한국어 신조어 사용 비율
    "past_tense": 0.10,         # 과거형 비율 (한계 있음)
    "uniqueness": 0.25,         # 고유 단어 비율
    "question_ratio": 0.15      # 질문 문장 비율
}
