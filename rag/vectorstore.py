import faiss, json, uuid
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple
from rank_bm25 import BM25Okapi

class VectorStore:
    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = index_dir
        self.index_path = index_dir / "vectors.faiss"
        self.meta_path  = index_dir / "metadata.jsonl"
        self.dim = dim

        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.metadocs: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

        self._load()

    def _load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        self.metadocs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # align lengths
        n = int(self.index.ntotal)
        if len(self.metadocs) > n:
            self.metadocs = self.metadocs[:n]
        elif len(self.metadocs) < n:
            self.metadocs.extend({} for _ in range(n - len(self.metadocs)))

        # build BM25
        tokens = []
        for m in self.metadocs:
            t = (m.get("text") or m.get("snippet") or "").strip().lower()
            tokens.append(t.split())
        if tokens:
            self.bm25 = BM25Okapi(tokens)

    def _persist(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        assert embeddings.shape[0] == len(metadatas)
        # persist metadata
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("a", encoding="utf-8") as f:
            for m in metadatas:
                m.setdefault("chunk_id", str(uuid.uuid4()))
                if isinstance(m.get("text"), str):
                    m["text"] = m["text"][:1500]
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                self.metadocs.append(m)

        # add vectors and persist
        self.index.add(embeddings.astype("float32"))
        self._persist()

        # rebuild bm25
        corpus = []
        for md in self.metadocs:
            t = (md.get("text") or md.get("snippet") or "").strip().lower()
            corpus.append(t.split())
        if corpus:
            self.bm25 = BM25Okapi(corpus)

    def search(self, qvec: np.ndarray, k: int) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(qvec.astype("float32"), k)
        out: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            meta = self.metadocs[idx] if idx < len(self.metadocs) else {}
            out.append((float(score), meta))
        return out

    def lexical_topk(self, query: str, k: int) -> List[Tuple[float, int]]:
        if not self.bm25:
            return []
        toks = query.lower().split()
        scores = self.bm25.get_scores(toks)
        top = sorted([(float(s), i) for i, s in enumerate(scores)], reverse=True)[:k]
        return top

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
