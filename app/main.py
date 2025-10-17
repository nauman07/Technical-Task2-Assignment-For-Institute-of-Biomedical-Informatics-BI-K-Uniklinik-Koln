from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from app.settings import settings
from rag.embed import Embedder
from rag.vectorstore import VectorStore
from rag.retrieve import retrieve
from app.llm import synthesize_answer
from rag.universal_chunk import make_chunks

app = FastAPI(title="Minimal RAG API")

_embedder = Embedder()  # defaults to BGE-small
_dim = _embedder.encode_passages(["probe"]).shape[1]
_store = VectorStore(settings.INDEX_DIR, dim=_dim)

class IngestTextReq(BaseModel):
    texts: List[str]
    doc_id: Optional[str] = None
    source: Optional[str] = "inline"
    title: Optional[str] = "Inline Document"

class QueryReq(BaseModel):
    query: str
    top_k: int = settings.TOP_K
    strict: bool = True

@app.get("/health")
def health():
    try:
        idx_size = _store.size
    except Exception:
        idx_size = 0
    return {"status": "ok", "index_size": idx_size, "llm_ready": True}

@app.post("/ingest/text")
def ingest_text(req: IngestTextReq):
    metadatas, chunks = [], []
    title = req.title or req.doc_id or "Inline Document"
    for t in req.texts:
        for piece, meta in make_chunks(
            t or "",
            title=title,
            target_tokens=180,
            overlap_tokens=30,
            max_chars=1500,
        ):
            chunks.append(piece)
            # merge meta + our identifiers
            metadatas.append({
                **meta,
                "doc_id": req.doc_id or "inline",
                "source": req.source or "inline",
                "title": title,
            })
    if not chunks:
        return {"ingested_chunks": 0, "index_size": _store.size}

    embs = _embedder.encode_passages(chunks)
    _store.add(embs, metadatas)
    return {"ingested_chunks": len(chunks), "index_size": _store.size}

@app.post("/ingest/files")
def ingest_files(files: List[UploadFile] = File(...)):
    metadatas, chunks = [], []
    for f in files:
        raw = f.file.read()
        # universal path: decode, normalize, chunk
        try:
            text = raw.decode("utf-8-sig", errors="ignore")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
        title = Path(f.filename).name
        for piece, meta in make_chunks(
            text,
            title=title,
            target_tokens=180,
            overlap_tokens=30,
            max_chars=1500,
        ):
            chunks.append(piece)
            metadatas.append({
                **meta,
                "doc_id": title,
                "source": "file",
                "title": title,
            })

    if not chunks:
        return {"ingested_chunks": 0, "index_size": _store.size}

    embs = _embedder.encode_passages(chunks)
    _store.add(embs, metadatas)
    return {"ingested_chunks": len(chunks), "index_size": _store.size}
    
@app.post("/query")
def query(req: QueryReq):
    contexts, sources = retrieve(req.query, _embedder, _store, top_k=req.top_k)
    answer, used = synthesize_answer(req.query, contexts, strict=req.strict)
    # mark which sources were actually cited
    for s in sources:
        s["cited"] = s.get("rank") in used
    return {"answer": answer, "sources": sources, "strict": req.strict}

