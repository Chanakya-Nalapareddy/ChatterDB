from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ThreadRow:
    id: str
    title: str
    summary_text: str
    state_json: Dict[str, Any]
    created_at: int
    updated_at: int


class SQLiteMemoryStore:
    """
    Persists:
      - threads (title + summary + state_json)
      - messages (role/content + assistant sql/preview as JSON blobs)
    """

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
              sql_text TEXT,
              preview_json TEXT,
              created_at INTEGER NOT NULL,
              FOREIGN KEY(thread_id) REFERENCES threads(id)
            )
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at ASC)")
        conn.commit()
        conn.close()

    # ----------------------------
    # Threads
    # ----------------------------
    def create_thread(self, title: str = "New chat (RAG)") -> str:
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

    def list_threads(self, limit: int = 200) -> List[ThreadRow]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT id,title,summary_text,state_json,created_at,updated_at FROM threads ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        out: List[ThreadRow] = []
        for r in rows:
            try:
                state = json.loads(r["state_json"] or "{}")
                if not isinstance(state, dict):
                    state = {}
            except Exception:
                state = {}
            out.append(
                ThreadRow(
                    id=r["id"],
                    title=r["title"],
                    summary_text=r["summary_text"] or "",
                    state_json=state,
                    created_at=int(r["created_at"]),
                    updated_at=int(r["updated_at"]),
                )
            )
        return out

    def get_thread(self, thread_id: str) -> ThreadRow:
        conn = self._conn()
        r = conn.execute(
            "SELECT id,title,summary_text,state_json,created_at,updated_at FROM threads WHERE id=?",
            (thread_id,),
        ).fetchone()
        conn.close()
        if not r:
            raise KeyError(f"Thread not found: {thread_id}")
        try:
            state = json.loads(r["state_json"] or "{}")
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}
        return ThreadRow(
            id=r["id"],
            title=r["title"],
            summary_text=r["summary_text"] or "",
            state_json=state,
            created_at=int(r["created_at"]),
            updated_at=int(r["updated_at"]),
        )

    def rename_thread(self, thread_id: str, title: str) -> None:
        now = int(time.time())
        conn = self._conn()
        conn.execute("UPDATE threads SET title=?, updated_at=? WHERE id=?", (title, now, thread_id))
        conn.commit()
        conn.close()

    def update_thread_memory(self, thread_id: str, *, summary_text: Optional[str] = None, state_json: Optional[Dict[str, Any]] = None) -> None:
        now = int(time.time())
        conn = self._conn()
        if summary_text is not None and state_json is not None:
            conn.execute(
                "UPDATE threads SET summary_text=?, state_json=?, updated_at=? WHERE id=?",
                (summary_text, json.dumps(state_json), now, thread_id),
            )
        elif summary_text is not None:
            conn.execute(
                "UPDATE threads SET summary_text=?, updated_at=? WHERE id=?",
                (summary_text, now, thread_id),
            )
        elif state_json is not None:
            conn.execute(
                "UPDATE threads SET state_json=?, updated_at=? WHERE id=?",
                (json.dumps(state_json), now, thread_id),
            )
        conn.commit()
        conn.close()

    def delete_thread(self, thread_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE id=?", (thread_id,))
        conn.commit()
        conn.close()

    # ----------------------------
    # Messages
    # ----------------------------
    def append_message(
        self,
        *,
        thread_id: str,
        role: str,
        content: str,
        sql_text: Optional[str] = None,
        preview: Optional[Any] = None,
    ) -> None:
        now = int(time.time())
        mid = str(uuid.uuid4())
        preview_json = json.dumps(preview) if preview is not None else None

        conn = self._conn()
        conn.execute(
            "INSERT INTO messages(id,thread_id,role,content,sql_text,preview_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (mid, thread_id, role, content, sql_text, preview_json, now),
        )
        conn.execute("UPDATE threads SET updated_at=? WHERE id=?", (now, thread_id))
        conn.commit()
        conn.close()

    def get_messages(self, thread_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT role, content, sql_text, preview_json, created_at
            FROM messages
            WHERE thread_id=?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (thread_id, limit),
        ).fetchall()
        conn.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            preview = None
            if r["preview_json"]:
                try:
                    preview = json.loads(r["preview_json"])
                except Exception:
                    preview = None
            out.append(
                {
                    "role": r["role"],
                    "content": r["content"],
                    "sql": r["sql_text"],
                    "preview": preview,
                    "ts": int(r["created_at"]),
                }
            )
        return out
