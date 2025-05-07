from abc import ABC, abstractmethod

class BaseEmbedding(ABC):

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Convert a string into a list of floats representing the embedding.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of strings into a list of lists of floats representing the embeddings.
        """
        pass