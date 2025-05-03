from openai import OpenAI
from .base import LLM

class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None, temperature: float = 0.5, context: str = None):
        super().__init__(model=model, api_key=api_key, temperature=temperature, context=context)
        self.client = OpenAI(api_key=api_key)

    def invoke(self, prompt: str = None) -> str:
        if prompt is None:
            raise ValueError("Prompt cannot be None")
        response = self.client.responses.create(
            model = self.model,
            instructions = self.context,
            input = prompt,
        )
        return response.output_text.strip()