"""Chat: a local ChatGPT-style assistant (open-source Gemma via Ollama) grounded in the student's saved notes."""
import os

from .config import OLLAMA_MODEL
from .ingestion import ingest_text
from .recall import recall_state
from .search import hybrid_search
from .study import generate_study_notes, wants_study

CHAT_MIN = float(os.getenv("RECALL_CHAT_MIN", "0.40"))  # casual chat uses old notes only on a fairly strong match

# Speed settings (override with environment variables if needed)
KEEP_ALIVE = os.getenv("RECALL_KEEP_ALIVE", "30m")      # keep the model in memory between messages
NUM_CTX = int(os.getenv("RECALL_NUM_CTX", "2048"))      # smaller context window = faster
NUM_PREDICT = int(os.getenv("RECALL_NUM_PREDICT", "700"))  # max reply length (study notes need room)

MAX_HISTORY = 6
SYSTEM = (
    "You are a friendly study assistant for a B.Tech student, running locally on his own computer. "
    "You also have his saved NOTES: things he told other AIs earlier. Rules:\n"
    "1) For general questions, answer normally and clearly.\n"
    "2) When he asks about what he said, decided or shared before, answer ONLY from the NOTES and name the note title.\n"
    "3) If the NOTES do not contain it, say it is not in his notes and ask him to share it. Never invent his past. Never offer to search or browse.\n"
    "4) If he asks what is missing, compare his request with the NOTES and list what is not covered.\n"
    "5) Mention his past notes only when directly relevant. When he shares an opinion, chat naturally and give your own view."
)


def ollama_chat(messages: list[dict]) -> str:
    import ollama  # lazy import

    r = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        keep_alive=KEEP_ALIVE,
        options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    )
    return r["message"]["content"].strip()


def build_messages(message: str, hits, history: list[dict]):
    used = [h for h in hits if h.score >= CHAT_MIN][:4]
    notes = "\n".join(f"[{h.title}] {h.text}" for h in used) or "(no matching notes)"
    msgs = [{"role": "system", "content": SYSTEM + "\n\nNOTES:\n" + notes}]
    for m in history[-MAX_HISTORY:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": str(m["content"])})
    msgs.append({"role": "user", "content": message})
    return msgs, used


def should_save(message: str) -> bool:
    """Save what the student tells us (statements), not his questions, and never the model's own replies."""
    m = message.strip()
    return len(m) >= 40 and not m.endswith("?")


def handle_chat(conn, message: str, history=None, chat_fn=None, embedder=None, mode: str = "chat") -> dict:
    message = message.strip()
    fn = chat_fn or ollama_chat
    if mode == "study":  # explicit toggle/button only: AI-generated notes are stored, labelled, never used as evidence
        res = generate_study_notes(conn, message, fn, embedder)
        return {"mode": "study", "reply": res["notes"], "study_id": res["memory_id"], "state": None,
                "saved": False, "suggest_study": False, "citations": []}
    hits = hybrid_search(conn, message, k=4, embedder=embedder)  # search BEFORE saving, so it can't find itself
    state = recall_state(hits)
    msgs, used = build_messages(message, hits, history or [])
    reply = fn(msgs)
    saved = False
    if should_save(message):
        ingest_text(conn, "[Chat] " + message[:40], message, embedder=embedder)
        saved = True
    return {"mode": "chat", "reply": reply, "state": state, "saved": saved, "suggest_study": wants_study(message),
            "citations": [{"memory_id": h.memory_id, "title": h.title, "score": h.score, "text": h.text} for h in used]}