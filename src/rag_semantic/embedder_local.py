from __future__ import annotations

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """
    Local (open-source) embeddings via sentence-transformers.

    Default recommended models:
      - BAAI/bge-small-en-v1.5 (fast + strong)
      - BAAI/bge-base-en-v1.5 (higher quality, slower)
      - intfloat/multilingual-e5-small (multilingual)

    We normalize embeddings to unit length to make cosine similarity consistent.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        # device can be "cpu", "cuda", "mps" (if available). None lets the lib decide.
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) >= 64,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll do it ourselves
        )
        vecs = self._normalize(vecs)
        return vecs.astype(np.float32).tolist()

    def embed_query(self, query: str) -> List[float]:
        vec = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vec = self._normalize(vec)[0]
        return vec.astype(np.float32).tolist()
