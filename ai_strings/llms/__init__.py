from .gemini import GeminiLLM
from .openai_llm import OpenAILLM

def get_llm(llm: str, model: str, api_key: str, context: str = "You are an ai assistant", temperature: float = 0.5):
    if llm == "gemini":
        return GeminiLLM(model=model, api_key=api_key, temperature=temperature, context=context)
    elif llm == "openai":
        return OpenAILLM(model=model, api_key=api_key, temperature=temperature, context=context)
    else:
        raise ValueError(f"LLM {llm} not supported")