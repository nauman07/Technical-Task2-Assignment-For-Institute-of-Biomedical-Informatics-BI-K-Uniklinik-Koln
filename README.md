# Minimal RAG System - Universal Ingestion, Lightweight LLM, and Streamlit GUI

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that can ingest text from almost any file type, convert it into meaningful semantic chunks, index it efficiently, and provide contextual or conversational answers through a **FastAPI backend** and **Streamlit web interface**.
> Designed to be re-runnable on any machine with Docker.  
> Uses open-source embeddings (`BAAI/bge-small-en-v1.5`) via SentenceTransformers.  
> Lightweight FAISS vector store on local disk (`./data/index`).  
> No commercial API keys needed.

---

## Features Overview

### Smart Ingestion & Chunking
- **Universal text pipeline**: works for `.txt`, `.csv`, `.md`, `.html`, `.xml`, logs, and key:value style notes without separate modules.
- **Structure-aware chunking**:
  - Sentence- and field-aware for prose, tables, and configs.
  - Adaptive grouping (~180 tokens per chunk, ~30-token overlap).
  - Auto-detects rows, headings, and key:value patterns.
  - Adds `Title > Section` prefixes for better retrieval grounding.
- **De-noising & normalization**:
  - Skips boilerplate, GUIDs, and numeric noise.
  - Normalizes Unicode, trims long cells (>500 chars).

### Embedding & Retrieval
- **Proper query/passages prompts:**
  - Queries → `"Represent this query for retrieving relevant documents: {query}"`
  - Passages → plain text (no prefix)
- **Normalized vectors** for cosine similarity.
- **Hybrid retrieval**:
  - Vector@24 + BM25@50 → RRF fusion → MMR(λ = 0.7) → Cross-encoder rerank to top-3.
- **Synthesis**:
  - Extracts top sentences from top-2 chunks, fuses to ≤ 60 words.
  - Cites used chunks `[1][2]`.
  - Strict guard: returns *“I don’t know based on the indexed documents.”* if not grounded.

### Dual Prompt Modes (LLM behavior)
- **Strict mode** → answers *only* from retrieved context.
- **Chatty mode** → small-talk & fallback answers for casual queries (“hi”, “how are you?”, “thanks”, etc.).
- Toggle between modes in the GUI.

### Lightweight Backend
- **FastAPI** + **FAISS** + **Rank-BM25**
- Thread-safe, async-ready design.
- Background ingestion jobs for large files.
- Exposes `/query`, `/ingest`, `/status`, `/health`.

### GUI (Streamlit)
- Clean interface on `http://localhost:8051`.
- Tabs for:
  - File & inline ingestion
  - Querying with strict/chatty toggle
  - Real-time ingest progress for large files
- Handles API unavailability gracefully.

---

## Project Structure
```
├─ app/
│ ├─ main.py # FastAPI app with /ingest and /query
│ ├─ llm.py # Lightweight extractive answerer (no external model)
│ ├─ schemas.py # Pydantic request/response models
│ ├─ settings.py # Config (paths, model names)
├─ rag/
│ ├─ embed.py # SentenceTransformers wrapper (BGE/E5 query/passages formatting)
│ ├─ vectorstore.py # FAISS index + metadata + BM25 lexical helper
│ ├─ retrieve.py # Hybrid retrieval + RRF + simple rerank
│ ├─ rerank.py # Cross-encoder hook (optional)
│ ├─ ingest.py # Programmatic ingestion helpers
│ ├─ universal_chunk.py # Structure-aware chunking pipeline
├─ ui/
│ └─ streamlit_app.py # Optional GUI (talks to FastAPI)
├─ data/
│ └─ index/ # FAISS index + metadata.jsonl (created/used at runtime)
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

---

## Setup and Running the System

### Prerequisites
- [Docker](https://www.docker.com/)
- At least **2 GB RAM** for BGE-small.
- No GPU required.

### Clone the repository

```bash
git clone https://github.com/nauman07/Technical-Task2-Assignment-For-Institute-of-Biomedical-Informatics-BI-K-Uniklinik-Koln.git
cd <your-project-folder>
```
### Build and start

```bash
docker-compose up --build
```

Services:

* rag-api → FastAPI backend on port 8000

* rag-gui → Streamlit dashboard on port 8051

Open http://localhost:8051 in your browser.

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | Returns API health status. |
| **POST** | `/ingest/text` | Ingests inline text (JSON payload). |
| **POST** | `/ingest/files` | Uploads and ingests small files. |
| **POST** | `/query` | Queries the knowledge base. Supports a `strict` flag for chatty/strict response modes. |

## How It Works

* Ingestion
    Files or text are decoded, normalized, and split into coherent chunks (rag/universal_chunk.py).

* Embedding
    Each chunk is embedded with the BGE-small encoder (rag/embed.py), using passage formatting.

* Indexing
    Embeddings are stored in FAISS; corresponding metadata lives in metadocs.jsonl.

* Querying
    When a question arrives:

    * The query is formatted as a BGE query prompt.

    * Its embedding is computed.

    * Hybrid retrieval pulls top candidates.

    * Reranker reorders them for semantic relevance.

* Answer Synthesis

    * Extractive summarisation fuses top sentences from top chunks.

    * If strict mode is on and overlap < 2 tokens → answer: “I don’t know…”.

    * Otherwise, the system forms a short natural sentence with citations.

* Small-talk Handling
    In chatty mode, greetings or thank-yous trigger pre-defined conversational responses (app/smalltalk.py).

## Example Usage
### A. Inline Ingestion
```bash
curl -X POST http://127.0.0.1:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Name: Nauman Khan\nRole: ML Engineer\nTeam: Applied AI\nEmployer: Acme Analytics"],
    "doc_id": "persona",
    "title": "About Me"
  }'
  ```
For Windows PowerShell
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest/text" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "texts": ["Name: Nauman Khan\nRole: ML Engineer\nTeam: Applied AI\nEmployer: Acme Analytics"],   
    "doc_id": "persona",
    "title": "About Me"
  }'
```

  Response:
```json
{"ingested_chunks": 1, "index_size": 1}
```

### B. Query

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are you?", "strict": false}'
```
For Windows PowerShell
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "query": "Who are you?",
    "strict": false
  }'
```


Response:
```json
{
  "answer": "I’m Nauman Khan, an ML Engineer.",
  "sources": [{"doc_id": "persona", "section": "About Me"}],
  "strict": false
}
```

### Streamlit GUI Guide

    Visit http://localhost:8051

* Tabs

    * Ingest: upload files or paste text.

    * Query: type questions, toggle “Strict (context-only)” mode.

* UI Behaviors

    * Chatty mode handles greetings, small talk, and fallback.

    * Strict mode sticks to your ingested corpus.
    * We also have a TOP-K variable which controls how many chunks from FAISS (vector search) or BM25 (lexical search) you want to consider as potentially relevant.

## Tech Stack

| Component | Purpose | Key Library |
| :--- | :--- | :--- |
| **Backend** | Handles API requests and orchestrates the RAG pipeline. | FastAPI + FAISS vector store (`fastapi`, `faiss-cpu`) |
| **Embeddings** | Semantic encoding of documents and queries into vector representations. | sentence-transformers ($\text{BGE-small}$) |
| **Retrieval** | Executes Dense + Sparse Hybrid search to find relevant context. | `rank-bm25`, `rapidfuzz` |
| **LLM Layer** | Performs Extractive synthesis and provides chatty logic for responses. | Custom |
| **GUI** | Provides a web interface for user interaction and visualisation. | Streamlit Dashboard (`streamlit`) |
| **Background Jobs** | Manages asynchronous ingestion tracking and large data processing. | Custom threading manager |

## Why It Works

* RAG systems fail when embeddings and chunking are naïve. This design fixes the classic pitfalls:

* Universal chunker keeps each chunk a coherent “thought”.

* Query/passages prompt split ensures semantic embeddings align.

* Hybrid retrieval + rerank balances recall and precision.

* Extractive synthesis grounds every answer in a real context.


* Strict/chatty toggle separates factual QA from conversational polish.





