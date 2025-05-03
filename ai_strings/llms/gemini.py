from google.genai import types
from google import genai
from .base import LLM

class GeminiLLM(LLM):
    def __init__(self, model:str = "gemini-1.5-flash", api_key = None , temperature = 0.5, context = "You are a ai assistant"):
        super().__init__(model=model, api_key=api_key, temperature=temperature, context=context)
        self.client = genai.Client(api_key=api_key)
    
    def chat(self, messages, **kwargs):
        return super().chat(messages, **kwargs)

    def invoke(self, prompt:str = None) -> str:
        if prompt is None:
            raise ValueError("Prompt cannot be None")
        if self.context is not None:
            prompt = self.context + "\n" + prompt
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature
            )
        )
        return response.text.strip()