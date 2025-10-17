from pathlib import Path
from typing import Iterable, Dict, Any, List
from .chunk import split_into_chunks
from .embed import Embedder

def iter_texts_from_files(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        yield {"doc_id": p.stem, "source": str(p.name), "text": text}

def build_records(raw_texts: Iterable[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    records = []
    for item in raw_texts:
        chunks = split_into_chunks(item["text"], chunk_size, overlap)
        for i, ch in enumerate(chunks):
            records.append({
                "doc_id": item["doc_id"],
                "source": item["source"],
                "chunk_idx": i,
                "text": ch
            })
    return records

def embed_and_store(records, embedder: Embedder, store):
    texts = [r["text"] for r in records]
    embs = embedder.encode(texts)
    # ensure text snippet is persisted for retrieval + GUI
    metas = [{
        "doc_id": r["doc_id"],
        "source": r["source"],
        "chunk_idx": r["chunk_idx"],
        "text": r["text"][:1000]  # truncate to keep metadata JSONL small
    } for r in records]
    store.add(embs, metas)
    return len(records)
