# src/rag_semantic/memory_store.py
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ThreadContext:
    thread_id: str
    title: str
    summary: str
    state: Dict[str, Any]
    messages: List[Dict[str, Any]]  # [{"role":"user"/"assistant","content":"...","ts":...}, ...]


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              summary_text TEXT NOT NULL DEFAULT '',
              state_json TEXT NOT NULL DEFAULT '{}',
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              thread_id TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
              content TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              FOREIGN KEY(thread_id) REFERENCES threads(id)
            )
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at)")
        conn.commit()
        conn.close()

    # ----------------------------
    # Thread ops
    # ----------------------------
    def create_thread(self, title: str = "New thread") -> str:
        tid = str(uuid.uuid4())
        now = int(time.time())
        conn = self._conn()
        conn.execute(
            "INSERT INTO threads(id,title,summary_text,state_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (tid, title, "", "{}", now, now),
        )
        conn.commit()
        conn.close()
        return tid

    def list_threads(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT id,title,summary_text,created_at,updated_at FROM threads ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def rename_thread(self, thread_id: str, new_title: str) -> None:
        now = int(time.time())
        conn = self._conn()
        conn.execute("UPDATE threads SET title=?, updated_at=? WHERE id=?", (new_title, now, thread_id))
        conn.commit()
        conn.close()

    def delete_thread(self, thread_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE id=?", (thread_id,))
        conn.commit()
        conn.close()

    # ----------------------------
    # Message ops
    # ----------------------------
    def append_message(self, thread_id: str, role: str, content: str) -> None:
        now = int(time.time())
        mid = str(uuid.uuid4())
        conn = self._conn()
        conn.execute(
            "INSERT INTO messages(id,thread_id,role,content,created_at) VALUES(?,?,?,?,?)",
            (mid, thread_id, role, content, now),
        )
        conn.execute("UPDATE threads SET updated_at=? WHERE id=?", (now, thread_id))
        conn.commit()
        conn.close()

    def get_thread_context(self, thread_id: str, last_n_messages: int = 6) -> ThreadContext:
        conn = self._conn()
        t = conn.execute(
            "SELECT id,title,summary_text,state_json FROM threads WHERE id=?",
            (thread_id,),
        ).fetchone()
        if not t:
            conn.close()
            raise KeyError(f"Thread not found: {thread_id}")

        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE thread_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (thread_id, last_n_messages),
        ).fetchall()
        conn.close()

        msgs = [{"role": r["role"], "content": r["content"], "ts": r["created_at"]} for r in reversed(rows)]
        state = json.loads(t["state_json"] or "{}")

        return ThreadContext(
            thread_id=t["id"],
            title=t["title"],
            summary=t["summary_text"] or "",
            state=state if isinstance(state, dict) else {},
            messages=msgs,
        )

    # ----------------------------
    # Memory update
    # ----------------------------
    def update_memory(self, thread_id: str, *, summary_text: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> None:
        now = int(time.time())
        conn = self._conn()
        cur = conn.cursor()

        if summary_text is not None and state is not None:
            cur.execute(
                "UPDATE threads SET summary_text=?, state_json=?, updated_at=? WHERE id=?",
                (summary_text, json.dumps(state), now, thread_id),
            )
        elif summary_text is not None:
            cur.execute(
                "UPDATE threads SET summary_text=?, updated_at=? WHERE id=?",
                (summary_text, now, thread_id),
            )
        elif state is not None:
            cur.execute(
                "UPDATE threads SET state_json=?, updated_at=? WHERE id=?",
                (json.dumps(state), now, thread_id),
            )

        conn.commit()
        conn.close()
