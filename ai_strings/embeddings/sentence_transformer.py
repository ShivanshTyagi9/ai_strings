from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the SentenceTransformer model.
        :param model_name: Name of the pre-trained model to use.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """
        Convert a single string into a list of floats representing the embedding.
        :param text: The input string to embed.
        :return: A list of floats representing the embedding.
        """
        return self.model.encode(text, convert_to_numpy = True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of strings into a list of lists of floats representing the embeddings.
        :param texts: A list of input strings to embed.
        :return: A list of lists of floats representing the embeddings.
        """
        return self.model.encode(texts, convert_to_numpy = True).tolist()