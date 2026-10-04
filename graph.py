"""Relationship engine (Phases 5-6): suggest 0-3 candidate links per new memory.
Links are always created as 'pending' and only count once the user accepts them.

Reading a link (src -> dst, relation): "src is <relation> dst", e.g.
  decision-for : src is a decision made for dst
  tested-by    : src was tested by dst
  caused-by    : src was caused by dst
  solved-by    : src was solved by dst
  supports     : src supports dst
`classify_pair` is a transparent keyword baseline; pass your own `classifier`
(e.g. one that asks a local Gemma model) with the same signature.
"""
import re
import sqlite3
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from .database import RELATIONS
from .ingestion import tokens

_CUES = {
    "decision-for": re.compile(r"\b(decid\w*|decision|chose|chosen|going with|we will use|agreed)\b", re.I),
    "solved-by": re.compile(r"\b(fix(ed|es)?|solv\w*|resolv\w*|workaround|patched)\b", re.I),
    "caused-by": re.compile(r"\b(because|caused?|due to|root cause|reason (was|is))\b", re.I),
    "tested-by": re.compile(r"\b(test(ed|s|ing)?|measur\w*|benchmark\w*|experiment\w*|results?)\b", re.I),
}
_PRIORITY = ["decision-for", "solved-by", "caused-by", "tested-by", "supports"]


def classify_pair(src_text: str, dst_text: str) -> Optional[str]:
    """Decision cues are read from the source; test/cause/fix cues from the destination."""
    if _CUES["decision-for"].search(src_text):
        return "decision-for"
    for rel in ("solved-by", "caused-by", "tested-by"):
        if _CUES[rel].search(dst_text):
            return rel
    return "supports"


def _memory_text(conn, memory_id: int, limit: int = 3000) -> str:
    rows = conn.execute(
        "SELECT text FROM memory_chunks WHERE memory_id = ? ORDER BY chunk_index", (memory_id,)
    ).fetchall()
    return " ".join(r["text"] for r in rows)[:limit]


def _memory_vectors(conn, dim_bytes: int) -> dict[int, np.ndarray]:
    sums: dict[int, list] = {}
    for r in conn.execute("SELECT c.memory_id, c.embedding FROM memory_chunks c JOIN memories m ON m.id = c.memory_id WHERE c.embedding IS NOT NULL AND m.origin = 'user'"):
        if len(r["embedding"]) == dim_bytes:
            sums.setdefault(r["memory_id"], []).append(np.frombuffer(r["embedding"], dtype=np.float32))
    out = {}
    for mid, vs in sums.items():
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        out[mid] = v / n if n else v
    return out


def _explain(conn, src, dst, rel, sim, memory_id, new_text, other_text) -> str:
    """Plain-language reasons for a suggested link (shown to the user before they accept or reject)."""
    src_text, dst_text = (new_text, other_text) if src == memory_id else (other_text, new_text)
    shared = sorted(set(tokens(src_text)) & set(tokens(dst_text)), key=len, reverse=True)[:4]
    lines = ["Both notes mention: " + ", ".join(shared)] if shared else []
    lines.append(f"Text similarity {sim:.2f}")
    dates = [conn.execute("SELECT created_at FROM memories WHERE id = ?", (i,)).fetchone()[0] for i in (src, dst)]
    try:
        days = abs((datetime.fromisoformat(dates[0]) - datetime.fromisoformat(dates[1])).days)
        lines.append("Dated the same day" if days == 0 else f"Dated {days} day(s) apart")
    except (TypeError, ValueError):
        pass
    cue = _CUES.get(rel)
    m = cue.search(src_text if rel == "decision-for" else dst_text) if cue else None
    if m:
        lines.append(f'Wording: the {"first" if rel == "decision-for" else "second"} note says "{m.group(0)}", which fits "{rel}"')
    return "\n".join(lines)


def suggest_links(conn, memory_id: int, classifier: Callable = classify_pair,
                  max_links: int = 3, min_sim: float = 0.3) -> list[dict]:
    """Find the most similar other memories and store up to `max_links` pending links."""
    me = conn.execute(
        "SELECT embedding FROM memory_chunks WHERE memory_id = ? AND embedding IS NOT NULL LIMIT 1", (memory_id,)
    ).fetchone()
    if not me:
        return []
    vecs = _memory_vectors(conn, len(me["embedding"]))
    if memory_id not in vecs:
        return []
    sims = sorted(
        ((float(vecs[memory_id] @ v), mid) for mid, v in vecs.items() if mid != memory_id), reverse=True
    )
    new_text = _memory_text(conn, memory_id)
    created = []
    for sim, other in sims:
        if sim < min_sim or len(created) >= max_links:
            break
        other_text = _memory_text(conn, other)
        options = [
            (memory_id, other, classifier(new_text, other_text)),
            (other, memory_id, classifier(other_text, new_text)),
        ]
        options = [o for o in options if o[2] in RELATIONS]
        if not options:
            continue
        src, dst, rel = min(options, key=lambda o: _PRIORITY.index(o[2]))  # ties -> new->other
        reason = _explain(conn, src, dst, rel, sim, memory_id, new_text, other_text)
        cur = conn.execute(
            "INSERT OR IGNORE INTO memory_links(src_memory_id, dst_memory_id, relation, reason) VALUES (?,?,?,?)",
            (src, dst, rel, reason),
        )
        conn.commit()
        if cur.rowcount:
            created.append({"id": cur.lastrowid, "src": src, "dst": dst, "relation": rel,
                            "status": "pending", "similarity": round(sim, 3)})
    return created


def pending_links(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT l.id, l.src_memory_id, l.dst_memory_id, l.relation, l.reason,
                  a.title AS src_title, b.title AS dst_title
           FROM memory_links l JOIN memories a ON a.id = l.src_memory_id
           JOIN memories b ON b.id = l.dst_memory_id WHERE l.status = 'pending' ORDER BY l.id"""
    ).fetchall()
    return [dict(r) for r in rows]


def set_link_status(conn, link_id: int, status: str) -> bool:
    if status not in ("accepted", "rejected"):
        raise ValueError("status must be 'accepted' or 'rejected'")
    with conn:
        cur = conn.execute("UPDATE memory_links SET status = ? WHERE id = ?", (status, link_id))
    return cur.rowcount == 1


def timeline(conn) -> list[dict]:
    """Ordered by the stored timestamp (when saved, or the date the user gave). Dates are never invented."""
    rows = conn.execute(
        "SELECT id, title, memory_type, created_at FROM memories WHERE origin = 'user' ORDER BY created_at ASC, id ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def memory_detail(conn, memory_id: int):
    m = conn.execute(
        """SELECT m.id, m.title, m.memory_type, m.origin, m.body, m.created_at, f.filename AS source
           FROM memories m LEFT JOIN source_files f ON f.id = m.source_id WHERE m.id = ?""",
        (memory_id,),
    ).fetchone()
    if not m:
        return None
    chunks = conn.execute(
        "SELECT page, text FROM memory_chunks WHERE memory_id = ? ORDER BY chunk_index", (memory_id,)
    ).fetchall()
    links = conn.execute(
        """SELECT l.id, l.relation, l.src_memory_id, l.dst_memory_id, a.title AS src_title, b.title AS dst_title
           FROM memory_links l JOIN memories a ON a.id = l.src_memory_id JOIN memories b ON b.id = l.dst_memory_id
           WHERE (l.src_memory_id = ? OR l.dst_memory_id = ?) AND l.status = 'accepted'""",
        (memory_id, memory_id),
    ).fetchall()
    return {**dict(m), "chunks": [dict(c) for c in chunks], "links": [dict(x) for x in links]}


def edit_link(conn, link_id: int, relation: str) -> bool:
    """User changes the relation; editing counts as accepting."""
    if relation not in RELATIONS:
        raise ValueError("Unknown relation type.")
    try:
        with conn:
            cur = conn.execute(
                "UPDATE memory_links SET relation = ?, status = 'accepted', reason = COALESCE(reason,'') || ' (edited by user)' WHERE id = ?",
                (relation, link_id),
            )
    except sqlite3.IntegrityError:
        raise ValueError("That link already exists.")
    return cur.rowcount == 1


def delete_memory(conn, memory_id: int) -> bool:
    """Removes the note, its chunks, search index entries and links (and its source row, so it can be re-added)."""
    row = conn.execute("SELECT source_id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        return False
    with conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if row["source_id"]:
            conn.execute("DELETE FROM source_files WHERE id = ?", (row["source_id"],))
    return True


def study_list(conn) -> list[dict]:
    rows = conn.execute("SELECT id, title, created_at FROM memories WHERE origin = 'ai' ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def graph_data(conn) -> dict:
    """Nodes = the student's own notes; links = accepted (solid) and pending (dashed)."""
    nodes = timeline(conn)
    ids = {n["id"] for n in nodes}
    links = [dict(r) for r in conn.execute(
        "SELECT id, src_memory_id AS src, dst_memory_id AS dst, relation, status, reason FROM memory_links "
        "WHERE status IN ('accepted','pending')")]
    return {"nodes": nodes, "links": [l for l in links if l["src"] in ids and l["dst"] in ids]}
