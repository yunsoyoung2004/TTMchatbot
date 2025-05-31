import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from agents.user_state_agent import AgentState
from drift.detector import pure_run_detect  # ✅ 평가용 감지기
from shared.logger import logger

async def evaluate_drift_detection():
    try:
        with open('eval/evaluation.json', 'r', encoding='utf-8') as f:
            examples = json.load(f)
        logger.info(f"📂 Drift 평가 샘플 수: {len(examples)}")
    except FileNotFoundError:
        logger.error("❌ evaluation.json 파일을 찾을 수 없습니다.")
        return None

    y_true, y_pred, scores, times = [], [], [], []

    for idx, ex in enumerate(examples):
        text = ex["text"]
        expected = bool(ex["should_transition"])  # ✅ 정답 확실히 bool 변환
        previous = ex.get("previous_text", "")

        state = AgentState(
            stage="cbt1",
            question="",
            response=text,
            history=[previous, text],
            turn=0,
            drift_trace=[]
        )

        start = time.time()
        try:
            result = pure_run_detect(state)
        except Exception as e:
            logger.warning(f"❌ {idx+1}번 예시 처리 중 오류: {e}")
            continue

        elapsed = time.time() - start
        predicted = bool(result.get("drift", False))  # ✅ 예측도 확실히 bool 변환
        score = result.get("score", 0.0)

        logger.info(f"✅ {idx+1}/{len(examples)} - 예측: {predicted} (정답: {expected}) | 점수: {score:.4f} | 시간: {elapsed:.3f}s")

        y_true.append(expected)
        y_pred.append(predicted)
        scores.append(score)
        times.append(elapsed)

    if not y_true:
        logger.warning("⚠️ 평가에 사용할 유효한 예시가 없습니다.")
        return None

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_time = sum(times) / len(times) if times else 0.0

    result_summary = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "avg_drift_score": round(avg_score, 4),
        "avg_response_time": round(avg_time, 4)
    }

    logger.info("📊 Drift Detection 최종 평가 결과")
    for k, v in result_summary.items():
        logger.info(f"{k}: {v}")

    return result_summary
