from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseEmbedding

class TFIDFEmbedding(BaseEmbedding):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts: list[str]):
        """
        Fit the TF-IDF model to the provided texts.
        """
        self.vectorizer.fit(texts)
    
    def embed(self, text: str) -> list[float]:
        """
        Convert a single string into a list of floats representing the embedding.
        """
        return self.vectorizer.transform([text]).toarray()[0].tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of strings into a list of lists of floats representing the embeddings.
        """
        return self.vectorizer.transform(texts).toarray().tolist()