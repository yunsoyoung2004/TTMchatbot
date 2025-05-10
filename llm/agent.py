# 📁 llm/agent.py
from llm.loader import load_pipeline
from shared.state import AgentState

def run_llm_agent(state: AgentState, model_path: str, system_prompt: str, max_new_tokens: int = 100) -> AgentState:
    pipe = load_pipeline(model_path)
    
    prompt = f"""
<|user|>
{state['question']}
<|system|>
{system_prompt}
<|assistant|>
""".strip()

    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
    response = output.replace(prompt, "").strip()

    return {
        **state,
        "response": response,
        "turn": state.get("turn", 0) + 1,
        "history": state["history"] + [state["stage"]]
    }
