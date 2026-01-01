# streamlit_app.py
# Run: streamlit run streamlit_app.py
#
# Uses your existing src/app.py (ask(), ConversationState, SEMANTIC_YAML_PATH, DUCKDB_PATH)
# and provides a multi-thread ChatGPT-like UI with "New chat" and per-thread memory.

import uuid
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# IMPORTANT:
# Ensure you can import src.app (recommended: create an empty file src/__init__.py)
from src.app import ask, ConversationState, SEMANTIC_YAML_PATH, DUCKDB_PATH


# -----------------------------
# Thread helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def new_thread() -> Dict[str, Any]:
    tid = str(uuid.uuid4())
    return {
        "id": tid,
        "title": "New chat",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],  # list of {"role": "user"/"assistant", "content": "...", "sql": "...", "preview": [...]}
        # per-thread backend state (ConversationState instance)
        "state": ConversationState(),
    }


def get_active_thread() -> Dict[str, Any]:
    tid = st.session_state.active_thread_id
    for t in st.session_state.threads:
        if t["id"] == tid:
            return t
    # fallback
    t = new_thread()
    st.session_state.threads.append(t)
    st.session_state.active_thread_id = t["id"]
    return t


def auto_title_thread(thread: Dict[str, Any]) -> None:
    """Set title based on first user message if still default."""
    if thread.get("title") and thread["title"] != "New chat":
        return
    for m in thread["messages"]:
        if m["role"] == "user" and m["content"].strip():
            text = m["content"].strip().replace("\n", " ")
            thread["title"] = (text[:48] + "…") if len(text) > 48 else text
            return


# -----------------------------
# Streamlit init
# -----------------------------
st.set_page_config(page_title="ChatterDB", layout="wide")
st.title("ChatterDB")

if "threads" not in st.session_state:
    st.session_state.threads = [new_thread()]

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = st.session_state.threads[0]["id"]

active = get_active_thread()


# -----------------------------
# Sidebar: thread list + new chat
# -----------------------------
with st.sidebar:
    st.header("Chats")

    if st.button("➕ New chat", use_container_width=True):
        t = new_thread()
        st.session_state.threads.append(t)
        st.session_state.active_thread_id = t["id"]
        st.rerun()

    # Thread picker
    labels = [t["title"] for t in st.session_state.threads]
    ids = [t["id"] for t in st.session_state.threads]
    idx = ids.index(st.session_state.active_thread_id) if st.session_state.active_thread_id in ids else 0

    selected_label = st.radio("Conversation threads", labels, index=idx)
    selected_id = ids[labels.index(selected_label)]
    if selected_id != st.session_state.active_thread_id:
        st.session_state.active_thread_id = selected_id
        st.rerun()

    active = get_active_thread()

    st.divider()
    st.caption("Backend")
    st.code(f"YAML: {SEMANTIC_YAML_PATH}\nDB:   {DUCKDB_PATH}", language="text")


# -----------------------------
# Render conversation
# -----------------------------
for m in active["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # Optional: show SQL + preview for assistant messages
        if m["role"] == "assistant" and m.get("sql"):
            with st.expander("SQL", expanded=False):
                st.code(m["sql"], language="sql")

        if m["role"] == "assistant" and m.get("preview") is not None:
            preview = m["preview"]
            if isinstance(preview, list) and len(preview) > 0:
                with st.expander("Data preview", expanded=False):
                    try:
                        st.dataframe(pd.DataFrame(preview), use_container_width=True)
                    except Exception:
                        st.write(preview)


# -----------------------------
# Chat input -> call backend ask()
# -----------------------------
user_text = st.chat_input("Ask a question about the data…")

if user_text:
    # Append user message
    active["messages"].append({"role": "user", "content": user_text})
    active["updated_at"] = _now_iso()

    # Call backend with per-thread ConversationState (so follow-ups work later)
    try:
        out = ask(user_text, SEMANTIC_YAML_PATH, DUCKDB_PATH, active["state"])

        # Choose what to show as the assistant "content"
        # Prefer the NL summary you just implemented; fall back to brief_explanation.
        assistant_text = out.get("natural_language_answer") or out.get("brief_explanation") or "Done."

        active["messages"].append(
            {
                "role": "assistant",
                "content": assistant_text,
                "sql": out.get("sql"),
                "preview": out.get("preview", []),  # list-of-dicts
            }
        )

        auto_title_thread(active)
        active["updated_at"] = _now_iso()

    except Exception as e:
        active["messages"].append(
            {
                "role": "assistant",
                "content": f"Error:\n\n`{e}`",
            }
        )

    st.rerun()
