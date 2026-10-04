"""Settings and model-provider toggles (override with environment variables)."""
import os
from pathlib import Path

DB_PATH = Path(os.getenv("RECALL_DB", "recallbuddy.db"))

# Embeddings: "st" = SentenceTransformers (open-source, local) | "hash" = tiny offline fallback for tests
EMBEDDER = os.getenv("RECALL_EMBEDDER", "st")
ST_MODEL = os.getenv("RECALL_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Answer generation: "ollama" (local Gemma) | "none" (extractive answer, no LLM)
LLM_PROVIDER = os.getenv("RECALL_LLM", "none")
OLLAMA_MODEL = os.getenv("RECALL_OLLAMA_MODEL", "gemma3:4b")

CHUNK_CHARS = int(os.getenv("RECALL_CHUNK_CHARS", "800"))
CHUNK_OVERLAP = int(os.getenv("RECALL_CHUNK_OVERLAP", "100"))

# Grounded-recall thresholds on the hybrid score (0..1). Tune on your own notes.
STRONG = float(os.getenv("RECALL_STRONG", "0.50"))
WEAK = float(os.getenv("RECALL_WEAK", "0.30"))
