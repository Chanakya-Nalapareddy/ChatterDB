# streamlit_rag_app.py
# Run: streamlit run streamlit_rag_app.py

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.rag_semantic.rag_app import ask_rag, ConversationState, SEMANTIC_YAML_PATH, DUCKDB_PATH
from src.rag_semantic.thread_store_sqlite import (
    init_db,
    list_threads,
    create_thread,
    load_thread,
    add_message,
    add_turn,
    set_thread_title,
    delete_thread,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CHAT_DB_PATH = PROJECT_ROOT / "data" / "chat_threads.sqlite"


# -----------------------------
# ✅ UI CSS: hide "File change / Rerun / Always rerun" ONLY
# Keep the Streamlit toolbar/header so the 3-dots menu stays visible.
# -----------------------------
st.markdown(
    """
    <style>
    /* Hide ONLY the status widget (where Streamlit shows "File change", "Rerun", etc.) */
    [data-testid="stStatusWidget"] { display: none !important; }

    /* Keep toolbar/header visible so the 3-dots menu (screencast etc.) remains */
    [data-testid="stToolbar"] { display: flex !important; }
    header { visibility: visible !important; }

    /* Some Streamlit versions show status near deploy button; hide it if present */
    [data-testid="stDeployButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def ensure_one_thread_exists() -> str:
    threads = list_threads(CHAT_DB_PATH)
    if threads:
        return threads[0]["id"]
    t = create_thread(CHAT_DB_PATH, title="New chat")
    return t["id"]


def auto_title_thread_if_needed(thread_id: str, messages: List[Dict[str, Any]]) -> None:
    threads = list_threads(CHAT_DB_PATH)
    this = next((t for t in threads if t["id"] == thread_id), None)
    if not this:
        return
    if this.get("title") and this["title"] != "New chat":
        return

    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            text = m["content"].strip().replace("\n", " ")
            title = (text[:48] + "…") if len(text) > 48 else text
            set_thread_title(CHAT_DB_PATH, thread_id, title)
            return


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def render_chart_to_fig(df: pd.DataFrame, chart: Dict[str, Any]) -> plt.Figure:
    """
    Grounded chart renderer. Uses ONLY df + chart spec.
    chart = {chart_type,x,y,y2,title}
    """
    chart_type = (chart.get("chart_type") or "bar").lower()
    x = chart.get("x")
    y = chart.get("y")
    y2 = chart.get("y2")
    title = chart.get("title") or ""

    fig, ax = plt.subplots()

    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_axis_off()
        return fig

    cols = list(df.columns)

    def col_or_none(c: Optional[str]) -> Optional[str]:
        return c if c in cols else None

    x = col_or_none(x)
    y = col_or_none(y)
    y2 = col_or_none(y2)

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in cols if c not in numeric_cols]

    # Defaults if missing
    if chart_type in ("bar", "pie"):
        if x is None:
            x = non_numeric_cols[0] if non_numeric_cols else cols[0]
        if y is None:
            y = numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else cols[0])

    if chart_type in ("line", "area"):
        if x is None:
            x = non_numeric_cols[0] if non_numeric_cols else cols[0]
        if y is None:
            y = numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else None)

    if chart_type == "hist":
        if x is None:
            x = numeric_cols[0] if numeric_cols else cols[0]

    if chart_type == "scatter":
        if x is None:
            x = numeric_cols[0] if numeric_cols else cols[0]
        if y is None:
            y = numeric_cols[1] if len(numeric_cols) >= 2 else (numeric_cols[0] if numeric_cols else None)

    # PIE (aggregate by label)
    if chart_type == "pie":
        if x is None or y is None:
            ax.text(0.5, 0.5, "Not enough columns for pie chart", ha="center", va="center")
            ax.set_axis_off()
            return fig

        tmp = df[[x, y]].copy()
        tmp[x] = tmp[x].astype(str)
        tmp[y] = _safe_to_numeric(tmp[y]).fillna(0.0)
        tmp = tmp.groupby(x, as_index=False)[y].sum()

        ax.pie(tmp[y], labels=tmp[x], autopct="%1.1f%%")
        ax.set_title(title or f"{y} by {x}")
        return fig

    # HIST
    if chart_type == "hist":
        series = _safe_to_numeric(df[x]).dropna()
        ax.hist(series)
        ax.set_title(title or f"Distribution of {x}")
        ax.set_xlabel(x)
        ax.set_ylabel("Count")
        return fig

    # SCATTER
    if chart_type == "scatter":
        if x is None or y is None:
            ax.text(0.5, 0.5, "Not enough numeric columns for scatter plot", ha="center", va="center")
            ax.set_axis_off()
            return fig
        xs = _safe_to_numeric(df[x])
        ys = _safe_to_numeric(df[y])
        ax.scatter(xs, ys)
        ax.set_title(title or f"{y} vs {x}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return fig

    # LINE / AREA (aggregate by x and sort)
    if chart_type in ("line", "area"):
        if x is None or y is None:
            ax.text(0.5, 0.5, "Not enough columns for time series chart", ha="center", va="center")
            ax.set_axis_off()
            return fig

        xdt = _safe_to_datetime(df[x])
        yv = _safe_to_numeric(df[y])

        tmp = pd.DataFrame({"x": xdt, "y": yv}).dropna(subset=["x"])
        tmp["y"] = tmp["y"].fillna(0.0)
        tmp = tmp.groupby("x", as_index=False)["y"].sum().sort_values("x")

        ax.plot(tmp["x"], tmp["y"])
        if chart_type == "area":
            ax.fill_between(tmp["x"], tmp["y"], alpha=0.3)

        ax.set_title(title or f"{y} over {x}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.autofmt_xdate()
        return fig

    # BAR (aggregate by x)
    if x is None or y is None:
        ax.text(0.5, 0.5, "Not enough columns for bar chart", ha="center", va="center")
        ax.set_axis_off()
        return fig

    tmp = df[[x, y]].copy()
    tmp[x] = tmp[x].astype(str)
    tmp[y] = _safe_to_numeric(tmp[y]).fillna(0.0)
    tmp = tmp.groupby(x, as_index=False)[y].sum()

    ax.bar(tmp[x], tmp[y])
    ax.set_title(title or f"{y} by {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def collect_thread_charts_as_zip(messages: List[Dict[str, Any]]) -> Optional[bytes]:
    """
    For every assistant message that has chart + preview, render and zip as PNGs.
    Returns None if no charts exist.
    """
    chart_msgs = [
        m
        for m in messages
        if m.get("role") == "assistant"
        and m.get("chart")
        and isinstance(m.get("preview"), list)
        and len(m["preview"]) > 0
    ]
    if not chart_msgs:
        return None

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        idx = 1
        for m in chart_msgs:
            chart = m["chart"]
            preview = m["preview"]
            df = pd.DataFrame(preview)

            fig = render_chart_to_fig(df, chart)
            png = fig_to_png_bytes(fig)

            title = (chart.get("title") or f"chart_{idx}").strip()
            safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)[:60].strip()
            filename = f"{idx:02d}_{safe or 'chart'}.png"
            zf.writestr(filename, png)
            idx += 1

    out.seek(0)
    return out.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ChatterDB", layout="wide")
st.title("ChatterDB")

init_db(CHAT_DB_PATH)

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = ensure_one_thread_exists()

# Sidebar
with st.sidebar:
    st.header("Chats")

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("➕ New chat", use_container_width=True):
            t = create_thread(CHAT_DB_PATH, title="New chat")
            st.session_state.active_thread_id = t["id"]
            st.rerun()

    with cols[1]:
        if st.button("🗑️ Delete chat", use_container_width=True):
            tid = st.session_state.active_thread_id
            delete_thread(CHAT_DB_PATH, tid)
            st.session_state.active_thread_id = ensure_one_thread_exists()
            st.rerun()

    threads = list_threads(CHAT_DB_PATH)
    labels = [t["title"] for t in threads]
    ids = [t["id"] for t in threads]

    if ids and st.session_state.active_thread_id not in ids:
        st.session_state.active_thread_id = ids[0]
        st.rerun()

    if ids:
        idx = ids.index(st.session_state.active_thread_id)
        selected_label = st.radio("Conversation threads", labels, index=idx)
        selected_id = ids[labels.index(selected_label)]
        if selected_id != st.session_state.active_thread_id:
            st.session_state.active_thread_id = selected_id
            st.rerun()

    st.divider()

    active_thread = load_thread(CHAT_DB_PATH, st.session_state.active_thread_id)
    zip_bytes = collect_thread_charts_as_zip(active_thread["messages"])

    st.download_button(
        "⬇️ Download all plots (ZIP)",
        data=zip_bytes or b"",
        file_name="plots.zip",
        mime="application/zip",
        use_container_width=True,
        disabled=(zip_bytes is None),
    )

# Load active thread
active = load_thread(CHAT_DB_PATH, st.session_state.active_thread_id)

# Render history
for i, m in enumerate(active["messages"]):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        if m["role"] == "assistant" and m.get("sql"):
            with st.expander("SQL", expanded=False):
                st.code(m["sql"], language="sql")

        preview = m.get("preview")
        if m["role"] == "assistant" and isinstance(preview, list) and len(preview) > 0:
            with st.expander("Data preview", expanded=False):
                st.dataframe(pd.DataFrame(preview), use_container_width=True)

        # Chart rendering + overrides + download
        chart = m.get("chart")
        if m["role"] == "assistant" and chart and isinstance(preview, list) and len(preview) > 0:
            df = pd.DataFrame(preview)
            cols = list(df.columns)
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            y_candidates = numeric_cols or cols

            with st.expander("Chart", expanded=True):
                c1, c2, c3 = st.columns([1, 1, 2])

                with c1:
                    chart_type = st.selectbox(
                        "Type",
                        options=["bar", "line", "pie", "area", "hist", "scatter"],
                        index=["bar", "line", "pie", "area", "hist", "scatter"].index(
                            (chart.get("chart_type") or "bar")
                        ),
                        key=f"chart_type_{i}",
                    )

                with c2:
                    if chart_type == "hist":
                        x_sel = st.selectbox(
                            "Column",
                            options=cols,
                            index=cols.index(chart.get("x")) if chart.get("x") in cols else 0,
                            key=f"x_{i}",
                        )
                        y_sel = None
                    else:
                        x_sel = st.selectbox(
                            "X",
                            options=cols,
                            index=cols.index(chart.get("x")) if chart.get("x") in cols else 0,
                            key=f"x_{i}",
                        )
                        y_sel = st.selectbox(
                            "Y",
                            options=y_candidates,
                            index=y_candidates.index(chart.get("y")) if chart.get("y") in y_candidates else 0,
                            key=f"y_{i}",
                        )

                with c3:
                    title = st.text_input(
                        "Title",
                        value=chart.get("title") or "",
                        key=f"title_{i}",
                    )

                chart_over = dict(chart)
                chart_over["chart_type"] = chart_type
                chart_over["x"] = x_sel
                chart_over["y"] = y_sel
                chart_over["title"] = title

                fig = render_chart_to_fig(df, chart_over)
                st.pyplot(fig, use_container_width=True)

                png_bytes = fig_to_png_bytes(fig)
                st.download_button(
                    "⬇️ Download plot (PNG)",
                    data=png_bytes,
                    file_name=f"plot_{i+1:02d}.png",
                    mime="image/png",
                    use_container_width=True,
                )

# Input (bottom)
user_text = st.chat_input("Ask a question about the data…")

if user_text:
    thread_id = active["id"]
    state: ConversationState = active["state"]

    add_message(CHAT_DB_PATH, thread_id, role="user", content=user_text)

    try:
        out = ask_rag(user_text, SEMANTIC_YAML_PATH, DUCKDB_PATH, state)

        assistant_text = out.get("natural_language_answer") or out.get("explanation") or "Done."
        sql = out.get("sql")
        preview_rows = out.get("preview", [])
        chart = out.get("chart")

        add_message(
            CHAT_DB_PATH,
            thread_id,
            role="assistant",
            content=assistant_text,
            sql=sql,
            preview=preview_rows,
            chart=chart,
        )

        # only persist a "turn" when there is SQL (chart-only followups have sql=None)
        if sql:
            add_turn(
                CHAT_DB_PATH,
                thread_id,
                question=user_text,
                sql=sql,
                result_columns=list(pd.DataFrame(preview_rows).columns) if isinstance(preview_rows, list) else [],
                row_count=len(preview_rows) if isinstance(preview_rows, list) else 0,
            )

        msgs_for_title = active["messages"] + [{"role": "user", "content": user_text}]
        auto_title_thread_if_needed(thread_id, msgs_for_title)

    except Exception as e:
        add_message(CHAT_DB_PATH, thread_id, role="assistant", content=f"Error:\n\n`{e}`")

    st.rerun()
