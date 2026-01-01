from __future__ import annotations

from typing import Any, Dict


def update_state(thread_state: Dict[str, Any], *, user_question: str, rag_output: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(thread_state or {})
    s["last_user_question"] = user_question
    s["last_sql"] = rag_output.get("sql")
    s["last_tables_used"] = rag_output.get("tables_used")
    s["last_joins_used"] = rag_output.get("joins_used")

    # If ask_rag returns structured info, keep it
    for k in ["plan", "tables_used", "joins_used", "assumptions"]:
        if k in rag_output and rag_output.get(k) is not None:
            s[k] = rag_output.get(k)

    # Keep last result hints
    s["last_answer_kind"] = "rag"
    return s


def build_summary(prev_summary: str, *, user_question: str, rag_output: Dict[str, Any]) -> str:
    """
    Keep it SHORT (1-3 lines). This is not chain-of-thought.
    """
    sql = rag_output.get("sql") or ""
    dims_hint = ""
    if "billingcountry" in sql.lower() and "genre" in sql.lower():
        dims_hint = "Revenue analysis by country and genre."
    elif "billingcountry" in sql.lower():
        dims_hint = "Revenue analysis by country."
    elif "genre" in sql.lower():
        dims_hint = "Revenue analysis by genre."

    base = dims_hint or (prev_summary or "").strip()
    if not base:
        base = "Recent questions involve revenue-style aggregations in the music store dataset."

    # Add a tiny last intent line
    tail = f"Last question: {user_question.strip()}"
    out = f"{base}\n{tail}".strip()
    return out[:900]
