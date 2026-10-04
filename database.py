"""SQLite connection + schema (Phase 2). FTS5 gives instant keyword search."""
import sqlite3
from pathlib import Path
from .config import DB_PATH

RELATIONS = ("decision-for", "tested-by", "caused-by", "solved-by", "supports", "related-to")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS source_files(
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    kind TEXT,
    sha256 TEXT UNIQUE,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_id INTEGER REFERENCES source_files(id),
    memory_type TEXT NOT NULL DEFAULT 'note',
    origin TEXT NOT NULL DEFAULT 'user',
    body TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS memory_chunks(
    id INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    page INTEGER,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS memory_links(
    id INTEGER PRIMARY KEY,
    src_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK(relation IN ({",".join("'%s'" % r for r in RELATIONS)})),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected')),
    reason TEXT,
    UNIQUE(src_memory_id, dst_memory_id, relation)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='memory_chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON memory_chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON memory_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON memory_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def get_conn(path=None) -> sqlite3.Connection:
    path = str(path or DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Upgrade databases created by earlier versions (adds memory_type and the related-to link type)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    if "memory_type" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'note'")
    if "origin" not in cols:  # 'user' = the student's own words (evidence), 'ai' = AI-generated study notes
        conn.execute("ALTER TABLE memories ADD COLUMN origin TEXT NOT NULL DEFAULT 'user'")
    if "body" not in cols:  # original text with line breaks preserved
        conn.execute("ALTER TABLE memories ADD COLUMN body TEXT")
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_links'").fetchone()
    if row and "'related-to'" not in row[0]:
        rels = ",".join("'%s'" % r for r in RELATIONS)
        conn.executescript(f"""
        ALTER TABLE memory_links RENAME TO memory_links_old;
        CREATE TABLE memory_links(
            id INTEGER PRIMARY KEY,
            src_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            dst_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            relation TEXT NOT NULL CHECK(relation IN ({rels})),
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected')),
            reason TEXT,
            UNIQUE(src_memory_id, dst_memory_id, relation)
        );
        INSERT INTO memory_links(id, src_memory_id, dst_memory_id, relation, status, reason)
            SELECT id, src_memory_id, dst_memory_id, relation, status, reason FROM memory_links_old;
        DROP TABLE memory_links_old;
        """)
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
