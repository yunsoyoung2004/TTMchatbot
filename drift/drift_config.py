FEATURE_WEIGHTS = {
    "lexical_redundancy": 0.33,
    "style_shit": 0.2,
    "past_tense": 0.1,
    "question_ratio": 0.07,
    "semantic_repetition": 0.3,
}

DRIFT_THRESHOLD = 0.6

ALLOWED_STAGE_TRANSITIONS = {
    "empathy": ["mi"],
    "mi": ["cbt1"],
    "cbt1": ["cbt2"],
    "cbt2": ["cbt3"]
}