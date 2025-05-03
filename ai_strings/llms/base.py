class LLM:
    def __init__(self, model:str , api_key:str = None , temperature:float = 0.5, context:str = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.context = context

    def chat(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError
    
    def invoke(self, prompt:str = None, **kwargs) -> str:
        raise NotImplementedError