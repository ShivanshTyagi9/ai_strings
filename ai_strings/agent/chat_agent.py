from typing import Optional
from ai_strings.llms.base import LLM
from ai_strings.memory.base import BaseMemory


class ChatAgent:
    def __init__(self, llm: LLM, memory: BaseMemory, role: str = "You are a helpful assistant.", max_turns: int = 20):
        self.llm = llm
        self.memory = memory
        self.role = role
        self.max_turns = max_turns

    def _build_prompt(self, user_input: str) -> str:
        """
        Convert memory into prompt format for invoke().
        """
        history = self.memory.load_memory()[-self.max_turns * 2:]  # keep last N turns
        prompt_parts = [f"System: {self.role}"]
        for msg in history:
            prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def chat(self, user_input: str) -> str:
        prompt = self._build_prompt(user_input)
        response = self.llm.invoke(prompt=prompt)

        self.memory.save_memory({"role": "user", "content": user_input})
        self.memory.save_memory({"role": "assistant", "content": response})

        return response

    def reset(self):
        self.memory.clear()
