# src/rag_semantic/thread_store_sqlite.py
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag_semantic.rag_app import ConversationState


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _json_dumps(x: Any) -> str:
    # ✅ default=str avoids Timestamp/Decimal serialization issues
    return json.dumps(x, ensure_ascii=False, default=str)


def _json_loads(s: Optional[str], default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _ensure_column(con: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table});").fetchall()]
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {ddl};")


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sql TEXT NULL,
            preview_json TEXT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            question TEXT NOT NULL,
            sql TEXT NOT NULL,
            result_columns_json TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );
        """
    )

    # ✅ migration: add chart_json to messages if missing
    _ensure_column(con, "messages", "chart_json", "chart_json TEXT NULL")

    con.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_turns_thread_id ON turns(thread_id);")

    con.commit()
    con.close()


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def list_threads(db_path: Path) -> List[Dict[str, Any]]:
    con = _connect(db_path)
    rows = con.execute(
        "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def create_thread(db_path: Path, title: str = "New chat") -> Dict[str, Any]:
    tid = str(uuid.uuid4())
    now = _now_iso()

    con = _connect(db_path)
    con.execute(
        "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (tid, title, now, now),
    )
    con.commit()
    con.close()

    return {"id": tid, "title": title, "created_at": now, "updated_at": now}


def touch_thread(db_path: Path, thread_id: str) -> None:
    con = _connect(db_path)
    con.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (_now_iso(), thread_id))
    con.commit()
    con.close()


def set_thread_title(db_path: Path, thread_id: str, title: str) -> None:
    con = _connect(db_path)
    con.execute(
        "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
        (title, _now_iso(), thread_id),
    )
    con.commit()
    con.close()


def delete_thread(db_path: Path, thread_id: str) -> None:
    con = _connect(db_path)
    con.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
    con.commit()
    con.close()


def add_message(
    db_path: Path,
    thread_id: str,
    role: str,
    content: str,
    sql: Optional[str] = None,
    preview: Optional[List[Dict[str, Any]]] = None,
    chart: Optional[Dict[str, Any]] = None,
) -> None:
    con = _connect(db_path)
    con.execute(
        """
        INSERT INTO messages (thread_id, role, content, sql, preview_json, chart_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            role,
            content,
            sql,
            _json_dumps(preview) if preview is not None else None,
            _json_dumps(chart) if chart is not None else None,
            _now_iso(),
        ),
    )
    con.commit()
    con.close()
    touch_thread(db_path, thread_id)


def add_turn(
    db_path: Path,
    thread_id: str,
    question: str,
    sql: str,
    result_columns: List[str],
    row_count: int,
) -> None:
    con = _connect(db_path)
    con.execute(
        """
        INSERT INTO turns (thread_id, question, sql, result_columns_json, row_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, question, sql, _json_dumps(result_columns), int(row_count), _now_iso()),
    )
    con.commit()
    con.close()
    touch_thread(db_path, thread_id)


def load_thread(db_path: Path, thread_id: str) -> Dict[str, Any]:
    con = _connect(db_path)

    t = con.execute(
        "SELECT id, title, created_at, updated_at FROM threads WHERE id = ?",
        (thread_id,),
    ).fetchone()
    if not t:
        con.close()
        raise ValueError(f"Thread not found: {thread_id}")

    msg_rows = con.execute(
        """
        SELECT role, content, sql, preview_json, chart_json, created_at
        FROM messages
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (thread_id,),
    ).fetchall()

    turn_rows = con.execute(
        """
        SELECT question, sql, result_columns_json, row_count, created_at
        FROM turns
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (thread_id,),
    ).fetchall()

    con.close()

    messages: List[Dict[str, Any]] = []
    last_assistant_preview: List[Dict[str, Any]] = []
    last_assistant_cols: List[str] = []

    for r in msg_rows:
        preview = _json_loads(r["preview_json"], default=[])
        chart = _json_loads(r["chart_json"], default=None)
        msg = {
            "role": r["role"],
            "content": r["content"],
            "sql": r["sql"],
            "preview": preview,
            "chart": chart,
            "created_at": r["created_at"],
        }
        messages.append(msg)

        if r["role"] == "assistant" and isinstance(preview, list) and preview:
            last_assistant_preview = preview
            last_assistant_cols = list(preview[0].keys()) if isinstance(preview[0], dict) else []

    state = ConversationState()

    for tr in turn_rows:
        state.history.append(
            {
                "question": tr["question"],
                "sql": tr["sql"],
                "result_columns": _json_loads(tr["result_columns_json"], default=[]),
                "row_count": int(tr["row_count"]),
            }
        )

    if len(state.history) > 6:
        state.history = state.history[-6:]

    # ✅ restore last preview cache for chart-only followups
    state.last_preview = last_assistant_preview or []
    state.last_columns = last_assistant_cols or []

    return {
        "id": t["id"],
        "title": t["title"],
        "created_at": t["created_at"],
        "updated_at": t["updated_at"],
        "messages": messages,
        "state": state,
    }
