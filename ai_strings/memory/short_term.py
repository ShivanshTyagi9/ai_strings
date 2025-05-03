from .base import BaseMemory

class ShortTermMemory(BaseMemory):
    def __init__(self):
        self.messages = []

    def load_memory(self):
        return self.messages
    
    def save_memory(self, message: dict):
        self.messages.append(message)

    def clear(self):
        self.messages = []