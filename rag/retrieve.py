from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from rag.rerank import rerank as ce_rerank

def _rrf(ranks: List[List[int]], k: float = 60.0) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion across rank lists of indices.
    ranks: list of lists, e.g., [[2,0,5,...], [0,3,1,...]]
    returns: dict idx -> fused_score
    """
    fused: Dict[int, float] = {}
    for rlist in ranks:
        for r, idx in enumerate(rlist):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + r + 1.0)
    return fused

def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_idxs: List[int],
         top_k: int, lambda_: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance on candidate vectors.
    Returns indices into cand_idxs in selected order.
    """
    if not cand_idxs:
        return []
    selected: List[int] = []
    # precompute sim to query
    q_sim = cand_vecs @ query_vec / (np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8)
    remaining = list(range(len(cand_idxs)))
    while remaining and len(selected) < top_k:
        if not selected:
            i = int(max(remaining, key=lambda j: q_sim[j]))
            selected.append(i)
            remaining.remove(i)
            continue
        def mmr_score(j):
            sim_to_q = q_sim[j]
            sim_to_sel = max(
                (cand_vecs[j] @ cand_vecs[s]) /
                (np.linalg.norm(cand_vecs[j]) * np.linalg.norm(cand_vecs[s]) + 1e-8)
                for s in selected
            )
            return lambda_ * sim_to_q - (1 - lambda_) * sim_to_sel
        j_best = max(remaining, key=mmr_score)
        selected.append(j_best)
        remaining.remove(j_best)
    return [cand_idxs[i] for i in selected]


def retrieve(query: str, embedder, store, top_k: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
    try:
        # ---- early guards ----
        if getattr(store, "index", None) is None or getattr(store, "metadocs", None) is None:
            return [], []
        if store.index.ntotal == 0 or len(store.metadocs) == 0:
            return [], []

        qv = embedder.encode_queries([query])

        # 1) vector candidates
        vec_hits = store.search(qv, 24)  # [(score, meta)]
        vec_idxs = [store.metadocs.index(h[1]) for h in vec_hits if h and h[1] in store.metadocs]

        # 2) BM25, only if corpus not empty
        bm_idxs = []
        if not getattr(store, "bm25", None):
            tokens = []
            for m in store.metadocs:
                t = (m.get("text") or m.get("snippet") or "").strip().lower()
                if t:
                    tokens.append(t.split())
            if tokens:
                store.bm25 = BM25Okapi(tokens)  # type: ignore[attr-defined]
        if getattr(store, "bm25", None):
            toks = query.lower().split()
            if toks:
                bm_scores = store.bm25.get_scores(toks)  # type: ignore[attr-defined]
                bm_idxs = [i for _, i in sorted(((float(s), i) for i, s in enumerate(bm_scores)), reverse=True)[:50]]

        # 3) RRF fusion
        rank_lists = []
        if vec_idxs: rank_lists.append(vec_idxs)
        if bm_idxs:  rank_lists.append(bm_idxs)
        if not rank_lists:
            return [], []

        rrf_scores = _rrf(rank_lists, k=60.0)
        cand_idxs = [i for i, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
        if not cand_idxs:
            return [], []

        # 4) MMR (diversify)
        texts = [(store.metadocs[i].get("text") or store.metadocs[i].get("snippet") or "") for i in cand_idxs]
        cand_vecs = embedder.encode_passages(texts)
        mmr_selected = _mmr(qv[0], cand_vecs, list(range(len(cand_idxs))), top_k=min(8, len(cand_idxs)), lambda_=0.7)
        chosen_idxs = [cand_idxs[i] for i in mmr_selected]
        if not chosen_idxs:
            return [], []

        # 5) Cross-encoder rerank → top-3
        passages = [(store.metadocs[i].get("text") or store.metadocs[i].get("snippet") or "") for i in chosen_idxs]
        ce_rank = ce_rerank(query, passages, top_k=min(3, len(passages)))
        final_idxs = [chosen_idxs[i] for _, i in ce_rank]
        if not final_idxs:
            return [], []

        contexts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for rank, idx in enumerate(final_idxs, 1):
            meta = store.metadocs[idx] if idx < len(store.metadocs) else {}
            snippet = (meta.get("text") or meta.get("snippet") or "")
            contexts.append(snippet)
            sources.append({
                **meta,
                "rank": rank,
                "snippet": snippet[:500],
                "chunk_index": idx,
            })
        return contexts, sources

    except Exception:
        # Last-resort safety: never propagate an exception to the endpoint
        return [], []

