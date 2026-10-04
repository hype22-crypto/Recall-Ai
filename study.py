"""Import a pasted ChatGPT/Gemini conversation: only the student's own messages become notes."""
import re

from .ingestion import ingest_text

_USER_RE = re.compile(r"^\s*(?:(?:you said|you|user|me|human|student)\s*:\s*(.*)|(?:you said|you)\s*)$", re.I)
_AI_RE = re.compile(r"^\s*(?:(?:chatgpt said|chatgpt|gemini|assistant|ai|claude|copilot|bot|model|deepseek)\s*(?::\s*.*)?)$", re.I)


def split_user_turns(text: str):
    """Returns (his_messages, labels_found). AI turns are dropped: only his own words become notes."""
    turns, speaker, buf, seen = [], None, [], False
    for line in text.splitlines():
        m = _USER_RE.match(line)
        if m or _AI_RE.match(line):
            seen = True
            if speaker == "user" and buf:
                turns.append(" ".join(buf).strip())
            speaker = "user" if m else "ai"
            buf = [(m.group(1) or "").strip()] if m else []
            continue
        if speaker == "user" and line.strip():
            buf.append(line.strip())
    if speaker == "user" and buf:
        turns.append(" ".join(buf).strip())
    return [t for t in turns if t], seen


def import_chat(conn, text: str, source_ai: str = "Other", embedder=None) -> dict:
    turns, seen = split_user_turns(text)
    if not seen:
        return {"saved": 0, "found": 0, "hint": "No speaker labels found. Paste the chat with lines like 'You:' and 'ChatGPT:'."}
    ids = {ingest_text(conn, f"[{source_ai}] {t[:40]}", t, embedder=embedder) for t in turns if len(t) >= 25}
    return {"saved": len(ids), "found": len(turns), "hint": None}
