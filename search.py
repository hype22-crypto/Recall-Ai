"""Hybrid search: SQLite FTS5 (keywords) + cosine over stored embeddings (Phase 4)."""
from dataclasses import dataclass

import numpy as np

from .ingestion import get_embedder, tokens


@dataclass
class Hit:
    chunk_id: int
    memory_id: int
    title: str
    source: str | None
    page: int | None
    text: str
    score: float
    lex: float = 0.0
    vec: float = 0.0


def hybrid_search(conn, query: str, k: int = 5, embedder=None, w_vec: float = 0.6, pool: int = 20, include_ai: bool = False) -> list[Hit]:
    """score = w_vec * cosine + (1 - w_vec) * share of query keywords found in the chunk (both 0..1)."""
    qtoks = list(dict.fromkeys(tokens(query)))
    if not qtoks:
        return []
    embedder = embedder or get_embedder()

    # lexical candidates (FTS5, ranked by bm25)
    fts_q = " OR ".join(f'"{t}"*' for t in qtoks)  # prefix match: stall* finds "stalled"
    lex_ids = [
        r[0]
        for r in conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts_q, pool),
        )
    ]

    # vector scores (brute-force cosine: fine for a personal notes store)
    q = embedder([query])[0]
    origin_sql = "" if include_ai else " AND m.origin = 'user'"  # AI-written notes are never evidence
    rows = conn.execute(
        "SELECT c.id AS id, c.embedding AS embedding FROM memory_chunks c JOIN memories m ON m.id = c.memory_id "
        "WHERE c.embedding IS NOT NULL" + origin_sql
    ).fetchall()
    rows = [r for r in rows if len(r["embedding"]) == q.nbytes]  # skip vectors from a different embedder
    vecs: dict[int, float] = {}
    if rows:
        ids = [r["id"] for r in rows]
        mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        cos = mat @ q
        vecs = dict(zip(ids, map(float, cos)))
        vec_ids = [ids[i] for i in np.argsort(-cos)[:pool]]
    else:
        vec_ids = []

    cand = list(dict.fromkeys(lex_ids + vec_ids))
    if not cand:
        return []
    marks = ",".join("?" * len(cand))
    chunk_rows = conn.execute(
        f"""SELECT c.id, c.memory_id, c.page, c.text, m.title, f.filename
            FROM memory_chunks c JOIN memories m ON m.id = c.memory_id
            LEFT JOIN source_files f ON f.id = m.source_id WHERE c.id IN ({marks}){origin_sql}""",
        cand,
    ).fetchall()

    qset = set(qtoks)
    hits = []
    for r in chunk_rows:
        lex = len(qset & set(tokens(r["text"]))) / len(qset)
        vec = max(vecs.get(r["id"], 0.0), 0.0)
        hits.append(
            Hit(r["id"], r["memory_id"], r["title"], r["filename"], r["page"], r["text"],
                round(w_vec * vec + (1 - w_vec) * lex, 4), round(lex, 4), round(vec, 4))
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]
