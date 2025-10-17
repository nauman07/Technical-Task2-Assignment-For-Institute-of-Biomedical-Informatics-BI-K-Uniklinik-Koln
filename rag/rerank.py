# rag/rerank.py
from __future__ import annotations
from typing import List, Tuple
from sentence_transformers import CrossEncoder

# small & fast cross-encoder (~90MB)
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None

def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model

def rerank(query: str, passages: List[str], top_k: int = 3) -> List[Tuple[float, int]]:
    if not passages:
        return []
    model = _get_model()
    pairs = [(query, p or "") for p in passages]
    scores = model.predict(pairs, convert_to_numpy=True)  # higher is better
    ranked = sorted([(float(s), i) for i, s in enumerate(scores)], reverse=True)[:top_k]
    return ranked
