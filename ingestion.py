"""Text/PDF ingestion: extract -> chunk (with page numbers) -> embed -> store (Phase 3)."""
import hashlib
import os
import re
from datetime import date
from pathlib import Path

import numpy as np

from .config import CHUNK_CHARS, CHUNK_OVERLAP, ST_MODEL

STOPWORDS = set(
    "the and for are was were with that this from have has had not but you your our their they them "
    "what which who whom when where why how can could should would will shall may might into onto over "
    "about than then there here been being its it's his her she him out all any some more most such "
    "also just very does did doing done during".split()
)


def _stem(tok: str) -> str:
    """Very light suffix stripping so 'stalled'/'stall', 'tests'/'test' match (applied to both sides)."""
    for _ in range(2):
        for suf in ("ing", "ed", "es", "s"):
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                tok = tok[: -len(suf)]
                break
    return tok


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"\w+", text.lower()) if len(t) > 2 and t not in STOPWORDS]


_TYPE_CUES = [
    ("solution", r"\b(fix(ed)?|solv(ed|ution)|workaround|resolved)\b"),
    ("problem", r"\b(problem|issue|error|bug|failed|failure|stuck|not working)\b"),
    ("decision", r"\b(decid\w*|decision|chose|chosen|going with|will use)\b"),
    ("result", r"\b(results?|measured|scored|outcome|passed)\b"),
    ("research", r"\b(research\w*|read|studied|paper|article|learned|found that)\b"),
    ("idea", r"\b(idea|want to|plan to build|thinking of|what if)\b"),
]


def classify_type(text: str) -> str:
    """Keyword baseline for idea/research/decision/problem/solution/result/note. Never invents dates."""
    t = text[:2000].lower()
    for name, pattern in _TYPE_CUES:
        if re.search(pattern, t):
            return name
    return "note"


# ---------- embeddings ----------
_ST_MODEL = None


def hash_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Tiny offline bag-of-words embedder. For tests/fallback only."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokens(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def get_embedder():
    """Return fn(list[str]) -> normalized float32 matrix. Read env at call time."""
    global _ST_MODEL
    if os.getenv("RECALL_EMBEDDER", "st") == "hash":
        return hash_embed
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        _ST_MODEL = SentenceTransformer(ST_MODEL)
    model = _ST_MODEL
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


# ---------- chunking ----------
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------- extraction ----------
def extract_pdf(path) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text()))
    return pages


# ---------- storage ----------
def _existing_memory(conn, sha: str):
    row = conn.execute(
        "SELECT m.id FROM memories m JOIN source_files f ON f.id = m.source_id WHERE f.sha256 = ?", (sha,)
    ).fetchone()
    return row["id"] if row else None


def ingest_pages(conn, title, pages, filename, kind, sha, embedder=None, captured_at=None,
                 origin="user", memory_type=None) -> int:
    """pages: list[(page_number | None, text)]. Returns memory id (existing one if duplicate)."""
    existing = _existing_memory(conn, sha)
    if existing:
        return existing
    pieces = [(page, c) for page, text in pages for c in chunk_text(text)]
    if not pieces:
        raise ValueError("No text found to store.")
    embedder = embedder or get_embedder()
    vecs = embedder([c for _, c in pieces])
    stamp = f"{date.fromisoformat(captured_at)} 12:00:00" if captured_at else None  # raises ValueError if bad
    mtype = memory_type or classify_type(" ".join(t for _, t in pages))
    body = "\n\n".join(t.strip() for _, t in pages if t.strip())
    with conn:
        src = conn.execute(
            "INSERT INTO source_files(filename, kind, sha256) VALUES (?,?,?)", (filename, kind, sha)
        ).lastrowid
        mem = conn.execute(
            "INSERT INTO memories(title, source_id, memory_type, origin, body, created_at) VALUES (?,?,?,?,?,COALESCE(?, CURRENT_TIMESTAMP))",
            (title, src, mtype, origin, body, stamp),
        ).lastrowid
        for idx, ((page, chunk), vec) in enumerate(zip(pieces, vecs)):
            conn.execute(
                "INSERT INTO memory_chunks(memory_id, page, chunk_index, text, embedding) VALUES (?,?,?,?,?)",
                (mem, page, idx, chunk, np.asarray(vec, dtype=np.float32).tobytes()),
            )
    return mem


def ingest_text(conn, title: str, text: str, filename: str | None = None, embedder=None, captured_at=None,
                origin="user", memory_type=None) -> int:
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ingest_pages(conn, title, [(None, text)], filename or f"{title}.txt", "text", sha, embedder, captured_at, origin, memory_type)


def ingest_file(conn, path, embedder=None) -> int:
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if p.suffix.lower() == ".pdf":
        pages, kind = extract_pdf(p), "pdf"
    elif p.suffix.lower() in {".txt", ".md"}:
        pages, kind = [(None, raw.decode("utf-8", errors="ignore"))], "text"
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return ingest_pages(conn, p.stem, pages, p.name, kind, sha, embedder)
