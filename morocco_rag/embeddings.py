import os
from typing import Optional
from langchain_core.embeddings import Embeddings

class LocalFastEmbedWrapper(Embeddings):
    """
    Lightweight, high-speed local embedding wrapper using FastEmbed ONNX runtime.
    Runs locally on Apple Silicon / CPU with 0 API keys and 0 cost.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = list(self.model.embed([text]))[0]
        return embedding.tolist()

def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Get embedding model.
    Defaults to local FastEmbed (BAAI/bge-small-en-v1.5) for fast, free, local indexing.
    Falls back to OpenAI if explicitly requested and key is present.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set to use OpenAI embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    
    return LocalFastEmbedWrapper()
