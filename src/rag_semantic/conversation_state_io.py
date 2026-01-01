from __future__ import annotations

from typing import Any, Dict, List

from src.rag_semantic.rag_app import ConversationState


def state_to_json(state: ConversationState) -> Dict[str, Any]:
    return {"history": list(state.history or [])}


def state_from_json(obj: Dict[str, Any]) -> ConversationState:
    st = ConversationState()
    hist = (obj or {}).get("history") or []
    if isinstance(hist, list):
        cleaned: List[Dict[str, Any]] = [h for h in hist if isinstance(h, dict)]
        st.history = cleaned[-6:]
    return st


def merge_state_json(existing: Dict[str, Any], *, conv_state: ConversationState) -> Dict[str, Any]:
    out = dict(existing or {})
    out["history"] = list(conv_state.history or [])
    return out
