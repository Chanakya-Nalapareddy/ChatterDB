# src/rag_semantic/memory_updater.py
from __future__ import annotations

from typing import Any, Dict


def build_thread_summary(question: str, result: Dict[str, Any], prev_summary: str = "") -> str:
    """
    Keep summaries short and stable. This is *not* chain-of-thought.
    """
    plan = result.get("plan") or {}
    dims = plan.get("dimensions") or []
    topn = plan.get("topn") or {}
    topk = plan.get("topk_per_group") or {}

    parts = []
    if dims:
        parts.append(f"Dimensions: {', '.join(dims)}.")
    if topn:
        parts.append(f"TopN: {topn}.")
    if topk:
        parts.append(f"TopK-per-group: {topk}.")

    base = " ".join(parts).strip()
    if not base:
        base = "Recent analysis involves revenue-style aggregations over invoices."

    # Keep it small
    return base[:900]


def update_state_from_result(thread_state: Dict[str, Any], question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(thread_state or {})
    s["last_question"] = question
    s["last_sql"] = result.get("sql")
    s["last_plan"] = result.get("plan")
    s["tables_used"] = result.get("tables_used")
    s["joins_used"] = result.get("joins_used")

    plan = result.get("plan") or {}
    s["last_dimensions"] = plan.get("dimensions")
    s["last_topn"] = plan.get("topn")
    s["last_topk_per_group"] = plan.get("topk_per_group")
    return s
