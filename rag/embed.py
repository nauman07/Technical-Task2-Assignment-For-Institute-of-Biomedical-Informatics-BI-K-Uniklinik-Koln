from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

# Default is BGE-small v1.5 (great on CPU). For E5 set model_name accordingly.
_DEFAULT = "BAAI/bge-small-en-v1.5"

def _detect_family(model_name: str) -> str:
    m = model_name.lower()
    if "bge" in m:
        return "bge"
    if "e5" in m:
        return "e5"
    return "generic"

class Embedder:
    """
    Properly formats queries vs passages for BGE/E5.
    - BGE v1.5:   query="Represent this query for retrieving relevant documents: {q}"
                  passage="{p}"
    - E5:         query="query: {q}"
                  passage="passage: {p}"
    - generic:    no prefixing
    Embeddings are L2-normalized (cosine).
    """
    def __init__(self, model_name: str = _DEFAULT, max_seq_len: int = 256):
        self.model_name = model_name
        self.family = _detect_family(model_name)
        self.model = SentenceTransformer(model_name)
        try:
            self.model.max_seq_length = max_seq_len
        except Exception:
            pass

    # ----- formatting helpers -----
    def _fmt_query(self, q: str) -> str:
        q = (q or "").strip()
        if self.family == "bge":
            return f"Represent this query for retrieving relevant documents: {q}"
        if self.family == "e5":
            return f"query: {q}"
        return q

    def _fmt_passage(self, p: str) -> str:
        p = (p or "").strip()
        if self.family == "e5":
            return f"passage: {p}"
        # BGE/generic: no change
        return p

    # ----- public encode APIs -----
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        texts = [self._fmt_query(q) for q in queries]
        emb = self.model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, batch_size=32, show_progress_bar=False
        )
        return emb.astype("float32")

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        texts = [self._fmt_passage(p) for p in passages]
        emb = self.model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, batch_size=32, show_progress_bar=False
        )
        return emb.astype("float32")

