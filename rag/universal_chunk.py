# rag/universal_chunk.py
from __future__ import annotations
from typing import List, Tuple, Dict
import re
import unicodedata

# ---------- lightweight helpers ----------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')  # cheap sentence split
_MD_HEADER  = re.compile(r'^\s{0,3}(#{1,6})\s+(.*)$')       # markdown-like headings
_KV_LINE    = re.compile(r'^[^\n:]{1,40}:\s+.+$')           # "Key: Value"
_ROW_DELIMS = [",",";","|","\t"]

def _normalize_text(t: str) -> str:
    # Unicode normalize, strip control chars, collapse spaces but keep paragraph breaks
    t = unicodedata.normalize("NFC", t or "")
    # keep \n, \r, convert others
    t = t.replace("\x00", " ")
    t = re.sub(r"[^\S\r\n]+", " ", t)     # collapse runs of spaces/tabs
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t) # trim around newlines
    return t.strip()

def _approx_tokens(s: str) -> int:
    # sentence-transformers ~<=256 tokens sweet spot; coarse estimate
    return int(1.3 * max(1, len(s.split())))

def _mk_prefix(title: str, section: str | None) -> str:
    if title and section:
        return f"{title} > {section}\n"
    if title:
        return f"{title}\n"
    return ""

def _is_heading(line: str) -> str | None:
    m = _MD_HEADER.match(line)
    if m:
        return m.group(2).strip()
    # heuristic: short title-case line with no trailing punctuation
    if 2 <= len(line.split()) <= 12 and line[0].isupper() and line[-1].isalnum():
        words = line.split()
        cap_ratio = sum(w[:1].isupper() for w in words) / max(1, len(words))
        if cap_ratio > 0.6:
            return line.strip()
    return None

def _row_like(line: str) -> str | None:
    # treat as row if it contains a repeated delimiter with >=2 columns
    for d in _ROW_DELIMS:
        c = line.count(d)
        if c >= 2:
            return d
    return None

def _sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', (text or '').strip())
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

# ---------- core: annotate & assemble ----------

def _annotate_lines(text: str) -> List[Dict]:
    """
    returns list of {"type": heading|kv|row|plain, "text": str, "section": current_section}
    """
    out: List[Dict] = []
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # heading?
        h = _is_heading(line)
        if h:
            section = h
            out.append({"type": "heading", "text": h, "section": section})
            continue
        # key:value?
        if _KV_LINE.match(line):
            out.append({"type": "kv", "text": line, "section": section})
            continue
        # row-like?
        d = _row_like(line)
        if d:
            out.append({"type": "row", "text": line, "section": section, "delim": d})
            continue
        # plain
        out.append({"type": "plain", "text": line, "section": section})
    return out

def _row_group_mode(window: List[Dict]) -> bool:
    if not window:
        return False
    rows = sum(1 for x in window if x["type"] == "row")
    return rows >= (len(window) * 0.5)

def make_chunks(
    text: str,
    *,
    title: str = "Document",
    target_tokens: int = 180,
    overlap_tokens: int = 30,
    max_chars: int = 1500,
) -> List[Tuple[str, Dict]]:
    """
    Universal, structure-aware chunker:
      - sentence-aware for prose
      - key:value fused
      - row-grouping if row density is high
      - Title > Section prefix
      - ~180 token target, ~30 token overlap
    returns list[(chunk_text, metadata)]
    """
    t = _normalize_text(text)
    lines = _annotate_lines(t)

    chunks: List[Tuple[str, Dict]] = []

    # sliding window to detect row bursts
    WIN = 20
    buf_sents: List[str] = []
    buf_tokens = 0
    current_section: str | None = None
    row_buf: List[str] = []
    row_start = 1
    row_count = 0

    def flush_prose():
        nonlocal buf_sents, buf_tokens, current_section
        if not buf_sents:
            return
        body = " ".join(buf_sents)
        prefix = _mk_prefix(title, current_section)
        out = (prefix + body)[:max_chars]
        meta = {
            "doc_id": title,
            "source": "file",
            "title": title,
            "section": current_section or "",
            "text": out,
        }
        chunks.append((out, meta))
        # overlap by sentences until hitting ~overlap_tokens
        back, tokens = [], 0
        for s in reversed(buf_sents):
            st = _approx_tokens(s)
            if tokens + st >= overlap_tokens:
                break
            back.insert(0, s)
            tokens += st
        buf_sents = back
        buf_tokens = sum(_approx_tokens(s) for s in buf_sents)

    def flush_rows():
        nonlocal row_buf, row_start, row_count, current_section
        if not row_buf:
            return
        r_end = row_start + len(row_buf) - 1
        prefix = f"{title} > {current_section or 'rows'}\n"
        body = "\n".join(row_buf)
        out = (prefix + body)[:max_chars]
        meta = {
            "doc_id": title,
            "source": "file",
            "title": title,
            "section": f"rows {row_start}-{r_end}",
            "row_range": f"{row_start}-{r_end}",
            "text": out,
        }
        chunks.append((out, meta))
        # overlap ~5 rows
        back = row_buf[-5:] if len(row_buf) > 5 else row_buf[:]
        row_start = r_end - len(back) + 1
        row_buf = back

    window: List[Dict] = []
    for item in lines:
        window.append(item)
        if len(window) > WIN:
            window.pop(0)

        # update section on headings
        if item["type"] == "heading":
            current_section = item["text"]
            continue

        in_rows = _row_group_mode(window)

        if in_rows:
            # flush any prose we were building
            flush_prose()
            # add row sentences (normalize delimiter to '; ')
            row_count += 1
            delim = item.get("delim") or ","
            cols = [c.strip() for c in item["text"].split(delim)]
            row_buf.append(" ; ".join(c for c in cols if c))
            if len(row_buf) >= 30:   # ~30 rows per chunk
                flush_rows()
            continue

        # prose mode
        # if item is key:value, keep as its own sentence; else sentence-split
        if item["type"] == "kv":
            sents = [item["text"]]
        else:
            sents = _sentences(item["text"])

        for s in sents:
            st = _approx_tokens(s)
            if buf_tokens + st <= target_tokens or not buf_sents:
                buf_sents.append(s)
                buf_tokens += st
            else:
                flush_prose()
                # start new buf with this sentence
                buf_sents.append(s)
                buf_tokens = _approx_tokens(s)

    # flush tails
    flush_rows()
    flush_prose()

    # final de-noise: drop chunks with too few alphabetic tokens
    clean = []
    for chunk, meta in chunks:
        alpha = re.findall(r"[A-Za-z]{2,}", chunk)
        if len(alpha) >= 5:
            clean.append((chunk, meta))

    return clean
