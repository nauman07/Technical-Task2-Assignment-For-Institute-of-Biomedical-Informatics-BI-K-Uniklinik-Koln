from __future__ import annotations
from typing import List, Tuple
import re

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _wordset(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (t or "").lower()))

def _sentences(t: str) -> List[str]:
    t = re.sub(r'\s+', ' ', (t or '').strip())
    if not t:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]

def _best_sents_for_query(query: str, chunk: str, max_sents: int = 2) -> List[str]:
    q = _wordset(query)
    sents = _sentences(chunk)
    scored = []
    for s in sents:
        overlap = len(q & _wordset(s))
        if overlap > 0:
            scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sents]]

def _trim_to_words(text: str, limit_words: int = 60) -> str:
    toks = text.split()
    if len(toks) <= limit_words:
        return text
    return " ".join(toks[:limit_words]).rstrip(",.;:") + "…"

def synthesize_answer(query: str, contexts: List[str], strict: bool = True
                      ) -> Tuple[str, List[int]]:
    """
    - strict guard: if overlap across all chosen sentences < 2 unique tokens → 'I don't know...'
    - pick 1–2 sentences from the top-2 chunks that overlap the query
    - fuse with a tiny connective, cap at ≤60 words
    - return (answer, used_chunk_indices) where indices map to provided contexts order (1-based for citations)
    """
    if not strict:
        chit = _smalltalk_or_none(query)
        if chit:
            return chit, []
    if not contexts:
        return "I don't know based on the indexed documents.", []

    chosen_sents: List[str] = []
    used_idxs: List[int] = []

    for i, chunk in enumerate(contexts[:2], start=1):
        sents = _best_sents_for_query(query, chunk, max_sents=2)
        if sents:
            chosen_sents.extend(sents)
            used_idxs.append(i)

    # strict guard
    q = _wordset(query)
    overlap_tokens = set()
    for s in chosen_sents:
        overlap_tokens |= (q & _wordset(s))
    if len(overlap_tokens) < 2:
        return "I don't know based on the indexed documents.", []

    if not chosen_sents:
        return "I don't know based on the indexed documents.", []

    fused = " ".join(chosen_sents)
    fused = _trim_to_words(fused, 60)

    # append citations like [1][2] based on which chunks contributed
    cites = "".join(f"[{i}]" for i in used_idxs) if used_idxs else ""
    return (f"{fused} {cites}".strip(), used_idxs)

def _smalltalk_or_none(q: str) -> str | None:
    ql = (q or "").strip().lower()
    greetings = {"hi", "hello", "hey", "hola", "hi there", "good morning", "good evening"}
    thanks = {"thanks", "thank you", "thx", "ty", "much appreciated"}
    howare = {"how are you", "how r u", "how are u", "hows it going", "how’s it going"}
    bye = {"bye", "goodbye", "see ya", "see you", "later", "catch you later"}

    if ql in greetings or any(ql.startswith(g + " ") for g in greetings):
        return "Hey! 👋 What would you like to explore?"
    if any(p in ql for p in howare):
        return "Doing well—curious as ever. What can I help you dig into?"
    if ql in thanks or any(ql.endswith(t) for t in thanks):
        return "You’re welcome!"
    if ql in bye or any(ql.startswith(b + " ") for b in bye):
        return "Bye! If another question pops up, I’m here."

    return None
