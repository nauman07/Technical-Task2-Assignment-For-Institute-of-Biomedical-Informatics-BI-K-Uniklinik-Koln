from pydantic import BaseModel
from typing import List, Optional

class IngestText(BaseModel):
    texts: List[str]
    doc_id: Optional[str] = None
    source: Optional[str] = "inline"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
