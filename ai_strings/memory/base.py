from abc import ABC, abstractmethod

class BaseMemory(ABC):

    @abstractmethod
    def load_memory(self, **kwargs):
        """Load memory from a source."""
        pass

    @abstractmethod
    def save_memory(self, **kwargs):
        """Save memory to a source."""
        pass

    @abstractmethod
    def clear(self, **kwargs):
        """Clear memory."""
        pass