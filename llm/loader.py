from llama_cpp import Llama
import multiprocessing
from utils.logger import logger  # 로깅 유틸 사용

MODEL_CACHE = {}

def load_llm(path: str, context: int = 512, stop_tokens: list = ["User:", "Assistant:"], chat_format: str = "llama-3") -> Llama:
    """
    Llama 모델을 캐시해서 로딩하며 중복 로딩을 방지합니다.
    """
    if path not in MODEL_CACHE:
        logger.info(f"🚀 모델 로딩 중: {path}")
        try:
            MODEL_CACHE[path] = Llama(
                model_path=path,
                n_ctx=context,
                n_threads=max(1, multiprocessing.cpu_count() - 1),
                n_batch=8,
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.2,
                n_gpu_layers=0,
                low_vram=True,
                use_mlock=False,
                verbose=False,
                chat_format=chat_format,
                stop=stop_tokens
            )
            logger.info(f"✅ 모델 로딩 완료: {path}")
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {path} - {e}")
            raise
    return MODEL_CACHE[path]
