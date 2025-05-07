from google.genai import types
from google import genai
from .base import BaseEmbedding

# Task type: SEMANTIC_SIMILARITY
# Description: Used to generate embeddings that are optimized to assess text similarity.

# Task type: CLASSIFICATION
# Description: Used to generate embeddings that are optimized to classify texts according to preset labels.

# Task type: CLUSTERING
# Description: Used to generate embeddings that are optimized to cluster texts based on their similarities.

# Task type: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, YoutubeING, and FACT_VERIFICATION
# Description: Used to generate embeddings that are optimized for document search or information retrieval.

# Task type: CODE_RETRIEVAL_QUERY
# Description: Used to retrieve a code block based on a natural language query, such as sort an array or reverse a linked list. Embeddings of the code blocks are computed using RETRIEVAL_DOCUMENT.

class GeminiEmbedding(BaseEmbedding):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-exp-03-07",
        task_type: str = "SEMANTIC_SIMILARITY"
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.config = types.EmbedContentConfig(task_type=task_type)

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=self.config
        )
        return result.embeddings.values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = self.client.models.batch_embed_contents(
            model=self.model,
            contents=texts,
            config=self.config
        )
        return [res.embeddings.values for res in results]