import os
from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    # data paths
    DATA_DIR: Path = Path("/app/data")
    INDEX_DIR: Path = DATA_DIR / "index"
    DOCS_DIR: Path = DATA_DIR / "docs"

    # chunking / retrieval
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # generation caps (used by extractive formatter too)
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "60"))

settings = Settings()
settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
