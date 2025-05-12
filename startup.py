# ✅ startup.py
import os
from huggingface_hub import hf_hub_download

REPOS = {
    "mi":     ("youngbongbong/mimodel", "merged-mi-chat-q4_k_m.gguf"),
    "cbt":    ("youngbongbong/cbtmodel", "merged-cbt-chat-q4_k_m.gguf"),
    "ppi":    ("youngbongbong/ppimodel", "merged-ppi-prep-chat-q4_k_m.gguf"),
}

token = os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    print("⚠️ Hugging Face 토큰이 없습니다. 모델 다운로드를 건너뜁니다.")
    exit(0)

for name, (repo, file) in REPOS.items():
    path = hf_hub_download(repo_id=repo, filename=file, token=token)
    print(f"📥 {name.upper()} 모델 다운로드 완료: {path}")
