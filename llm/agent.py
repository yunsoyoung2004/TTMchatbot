from llm.loader import load_pipeline
from shared.state import AgentState

def run_llm_agent(state: AgentState, model_path: str, system_prompt: str, max_new_tokens: int = 100) -> AgentState:
    pipe = load_pipeline(model_path)

    prompt = f"""
<|user|>
{state.question}
<|system|>
{system_prompt}
<|assistant|>
""".strip()

    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
    response = output.replace(prompt, "").strip()

    updated_history = state.history + [state.question, response]

    return state.copy(update={
        "response": response,
        "turn": state.turn + 1,
        "history": updated_history
    })
