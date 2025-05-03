from typing import Optional
from ai_strings.memory.base import BaseMemory
from ai_strings.llms.base import LLM

class MyAgent:
    def __init__(self, llm: LLM, memory: Optional[BaseMemory] = None, use_chat: bool = True):
        self.llm = llm
        self.memory = memory
        self.use_chat = use_chat
        self.chat_history = []

    def run(self, user_input: str) -> str:
        """
        Unified entry point to run the agent.
        - Fetch memory
        - Format messages or prompt
        - Use chat or invoke
        - Store message in memory
        """

        context = ""
        if self.memory:
            context = self.memory.load_memory()

        if self.use_chat:
            messages = []

            if context:
                messages.append({"role": "system", "content": f"Context:\n{context}"})
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_input})

            response = self.llm.chat(messages)

            if self.memory:
                self.memory.save_memory({"role": "user", "content": user_input})
                self.memory.save_memory({"role": "assistant", "content": response})
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})

            return response

        else:
            prompt = f"{context}\n\n{user_input}".strip()
            response = self.llm.invoke(prompt=prompt)

            if self.memory:
                self.memory.save_memory({"role": "user", "content": user_input})
                self.memory.save_memory({"role": "assistant", "content": response})

            return response

    def reset(self):
        """Reset short-term state (chat history and memory)."""
        self.chat_history.clear()
        if self.memory:
            self.memory.clear()
