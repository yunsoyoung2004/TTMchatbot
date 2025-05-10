from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_pipeline(model_path: str, dtype="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
