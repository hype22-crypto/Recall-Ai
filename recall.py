"""Grounded recall (Phases 7-8): three states, strict refusal rule.
  grounded      -> strong evidence: answer only from the retrieved notes, with citations
  weak_evidence -> show the closest snippets, make NO claim
  no_evidence   -> refuse: no evidence = no personal claim
"""
import os
import re

from .config import OLLAMA_MODEL, STRONG, WEAK
from .ingestion import tokens
from .graph import memory_detail
from .search import Hit, hybrid_search

REFUSAL = "I couldn't find anything in your notes about that, so I won't guess."


def recall_state(evidence: list[Hit], strong: float = STRONG, weak: float = WEAK) -> str:
    if not evidence:
        return "no_evidence"
    top = max(h.score for h in evidence)
    if top >= strong:
        return "grounded"
    return "weak_evidence" if top >= weak else "no_evidence"


def verify_answer(answer: str, evidence: list[Hit], min_support: float = 0.6) -> bool:
    """Evidence check: most content words of the answer must appear in the cited notes."""
    words = set(tokens(answer))
    if not words:
        return False
    support = set(t for h in evidence for t in tokens(h.text))
    return len(words & support) / len(words) >= min_support


def ollama_llm(prompt: str) -> str:
    import ollama  # lazy import

    r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], keep_alive="30m",
                    options={"num_ctx": 1536, "num_predict": 120})
    return r["message"]["content"].strip()


def get_llm():
    return None if os.getenv("RECALL_LLM", "ollama") == "none" else ollama_llm


def _cite(h: Hit) -> dict:
    return {"chunk_id": h.chunk_id, "memory_id": h.memory_id, "title": h.title, "source": h.source, "page": h.page,
            "text": h.text, "score": h.score}


def build_response(question: str, evidence: list[Hit], llm=None, strong: float = STRONG, weak: float = WEAK) -> dict:
    state = recall_state(evidence, strong, weak)
    if state == "no_evidence":
        return {"state": state, "answer": REFUSAL, "verified": False, "citations": []}
    ranked = sorted(evidence, key=lambda h: h.score, reverse=True)[:3]
    cites = [_cite(h) for h in ranked]
    if state == "weak_evidence":
        return {"state": state, "answer": None, "verified": False, "citations": cites}
    # grounded
    answer, verified = ranked[0].text, True  # extractive default: quote the best note
    if llm:
        context = "\n---\n".join(h.text for h in ranked[:2])
        prompt = ("Answer in at most 2 short sentences, using ONLY the notes below. If they do not contain the answer, "
                  "reply exactly: NOT IN NOTES.\n\nNOTES:\n" + context + f"\n\nQUESTION: {question}\nANSWER:")
        candidate = llm(prompt)
        if "NOT IN NOTES" in candidate.upper():
            return {"state": "no_evidence", "answer": REFUSAL, "verified": False, "citations": []}
        if verify_answer(candidate, ranked):
            answer = candidate
        else:
            verified = False  # model drifted from the notes -> fall back to quoting them
    return {"state": state, "answer": answer, "verified": verified, "citations": cites}


def recall(conn, question: str, k: int = 5, llm=None, embedder=None) -> dict:
    evidence = hybrid_search(conn, question, k=k, embedder=embedder)
    llm = get_llm() if llm is None else (llm or None)  # llm=False -> skip the model, quote the best note
    try:
        return build_response(question, evidence, llm=llm)
    except Exception:  # local model unavailable or too slow: fall back to quoting the best note
        return build_response(question, evidence, llm=None)


_ALT = re.compile(r"\b(alternatives?|instead of|rather than|compared|versus|vs|other option|considered)\b", re.I)


def _snip(text, n=400):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def why_trace(conn, question: str, embedder=None) -> dict:
    """Why did I decide X? Anchor on the best matching note, then show only what the user approved around it,
    and list what has no saved evidence (no guessing)."""
    hits = hybrid_search(conn, question, k=5, embedder=embedder)
    state = recall_state(hits)
    if state == "no_evidence":
        return {"state": state, "message": REFUSAL, "decision": None, "chain": [], "missing": [], "pending": 0}
    types = {h.memory_id: conn.execute("SELECT memory_type FROM memories WHERE id = ?", (h.memory_id,)).fetchone()[0] for h in hits}
    good = [h for h in hits if h.score >= WEAK]
    anchor = next((h for h in good if types[h.memory_id] == "decision"), good[0] if good else hits[0])
    d = memory_detail(conn, anchor.memory_id)
    chain = []
    for l in d["links"]:
        oid = l["dst_memory_id"] if l["src_memory_id"] == anchor.memory_id else l["src_memory_id"]
        o = memory_detail(conn, oid)
        chain.append({"memory_id": oid, "title": o["title"], "type": o["memory_type"], "date": o["created_at"][:10],
                      "relation": f'{l["src_title"]} {l["relation"]} {l["dst_title"]}', "text": _snip(o["body"] or "")})
    chain.sort(key=lambda c: c["date"])
    alltext = (d["body"] or "") + " ".join(c["text"] for c in chain)
    kinds = {c["type"] for c in chain}
    missing = []
    if not kinds & {"problem", "research", "idea"}:
        missing.append("what led to it (no saved problem or research note is linked)")
    if not _ALT.search(alltext):
        missing.append("alternatives you considered")
    if "result" not in kinds and not any("tested-by" in c["relation"] for c in chain):
        missing.append("test results or evidence behind it")
    pending = conn.execute("SELECT COUNT(*) FROM memory_links WHERE status='pending' AND (src_memory_id=? OR dst_memory_id=?)",
                           (anchor.memory_id, anchor.memory_id)).fetchone()[0]
    decision = {"memory_id": d["id"], "title": d["title"], "type": d["memory_type"], "date": d["created_at"][:10],
                "source": d["source"], "text": _snip(d["body"] or anchor.text)}
    return {"state": state, "message": None, "decision": decision, "chain": chain, "missing": missing, "pending": pending}
