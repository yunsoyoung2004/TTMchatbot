FEATURE_WEIGHTS = {
    "lexical_redundancy": 0.33,
    "style_shit": 0.37,
    "semantic_repetition": 0.30
}

DRIFT_THRESHOLD = 0.28
effective_threshold = DRIFT_THRESHOLD




# ✅ 허용 가능한 단계 전이 목록 (현재는 사용되지 않지만 보존 가능)
ALLOWED_STAGE_TRANSITIONS = {
    "empathy": ["mi"],
    "mi": ["cbt1"],
    "cbt1": ["cbt2"],
    "cbt2": ["cbt3"],
    "cbt3": []
}

