"""FastAPI entrypoint. Run: uvicorn app.main:app --reload"""
import os
import tempfile
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import OLLAMA_MODEL
from . import graph, ingestion, recall as recall_mod, search, study
from .database import get_conn
from .models import MemoryIn, RecallIn

app = FastAPI(title="RecallBuddy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def db():
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


@app.on_event("startup")
def warm_up():
    """Pre-load the embedding model and Gemma in the background, so the first question is as fast as the rest."""
    def work():
        ingestion.get_embedder()
        if os.getenv("RECALL_LLM", "ollama") != "none":
            try:
                import ollama

                ollama.generate(model=OLLAMA_MODEL, prompt="", keep_alive="30m")  # loads the model into memory
            except Exception:
                pass  # Ollama not running: Ask still works by quoting your note

    threading.Thread(target=work, daemon=True).start()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/memories")
def add_memory(body: MemoryIn, conn=Depends(db)):
    try:
        mid = ingestion.ingest_text(conn, body.title, body.text, captured_at=body.captured_at)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"memory_id": mid, "suggested_links": graph.suggest_links(conn, mid)}


@app.post("/upload")
async def upload(file: UploadFile = File(...), conn=Depends(db)):
    name = Path(file.filename or "upload").name  # real filename becomes the title/source
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / name
        path.write_bytes(await file.read())
        try:
            mid = ingestion.ingest_file(conn, path)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"memory_id": mid, "suggested_links": graph.suggest_links(conn, mid)}


@app.get("/memories")
def list_memories(conn=Depends(db)):
    rows = conn.execute("SELECT id, title, created_at FROM memories ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/search")
def do_search(q: str, k: int = 5, conn=Depends(db)):
    return [h.__dict__ for h in search.hybrid_search(conn, q, k=k)]


@app.post("/recall")
def do_recall(body: RecallIn, conn=Depends(db)):
    return recall_mod.recall(conn, body.question, k=body.k, llm=None if body.use_llm else False)


@app.get("/links/pending")
def pending(conn=Depends(db)):
    return graph.pending_links(conn)


@app.post("/links/{link_id}/accept")
def accept(link_id: int, conn=Depends(db)):
    if not graph.set_link_status(conn, link_id, "accepted"):
        raise HTTPException(404, "link not found")
    return {"id": link_id, "status": "accepted"}


@app.post("/links/{link_id}/reject")
def reject(link_id: int, conn=Depends(db)):
    if not graph.set_link_status(conn, link_id, "rejected"):
        raise HTTPException(404, "link not found")
    return {"id": link_id, "status": "rejected"}






@app.get("/timeline")
def get_timeline(conn=Depends(db)):
    return graph.timeline(conn)


@app.get("/memories/{memory_id}")
def get_memory(memory_id: int, conn=Depends(db)):
    detail = graph.memory_detail(conn, memory_id)
    if not detail:
        raise HTTPException(404, "memory not found")
    return detail


class EditLinkIn(BaseModel):
    relation: str


@app.post("/links/{link_id}/edit")
def edit_link(link_id: int, body: EditLinkIn, conn=Depends(db)):
    try:
        ok = graph.edit_link(conn, link_id, body.relation)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if not ok:
        raise HTTPException(404, "link not found")
    return {"id": link_id, "status": "accepted", "relation": body.relation}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: int, conn=Depends(db)):
    if not graph.delete_memory(conn, memory_id):
        raise HTTPException(404, "memory not found")
    return {"deleted": memory_id}




class ImportIn(BaseModel):
    text: str
    source: str = "Other"


@app.post("/import")
def import_chat(body: ImportIn, conn=Depends(db)):
    return study.import_chat(conn, body.text, body.source)


class WhyIn(BaseModel):
    question: str


@app.post("/why")
def why(body: WhyIn, conn=Depends(db)):
    return recall_mod.why_trace(conn, body.question)


@app.get("/graph")
def get_graph(conn=Depends(db)):
    return graph.graph_data(conn)
