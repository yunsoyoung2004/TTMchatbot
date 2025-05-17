# 📁 shared/state_map.py

# 실제 챗봇 단계 흐름을 반영한 명확한 에이전트 전환 흐름
stage_flow = {
    "empathy": "mi",
    "mi": "cbt1",
    "cbt1": "cbt2",
    "cbt2": "cbt3",
    "cbt3": "action",
    "action": "end"
}
