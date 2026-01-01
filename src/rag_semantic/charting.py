# src/rag_semantic/charting.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------
# Chart intent parsing
# -----------------------------
_CHART_WORDS = [
    "chart", "plot", "graph", "visualize", "visualisation", "visualization",
    "bar", "bar chart", "column chart",
    "line", "line chart",
    "pie", "pie chart",
    "area", "area chart",
    "hist", "histogram",
    "scatter", "scatter plot",
]

_CHART_TYPE_PATTERNS = [
    ("bar", r"\b(bar|bar\s+chart|column\s+chart)\b"),
    ("line", r"\b(line|line\s+chart|time\s*series)\b"),
    ("pie", r"\b(pie|pie\s+chart)\b"),
    ("area", r"\b(area|area\s+chart)\b"),
    ("hist", r"\b(hist|histogram)\b"),
    ("scatter", r"\b(scatter|scatter\s+plot)\b"),
]


def is_chart_request(text: str) -> bool:
    q = (text or "").lower()
    return any(w in q for w in _CHART_WORDS)


def parse_requested_chart_type(text: str) -> Optional[str]:
    q = (text or "").lower()
    for typ, pat in _CHART_TYPE_PATTERNS:
        if re.search(pat, q):
            return typ
    if re.search(r"\b(chart|plot|graph|visualize|visualization)\b", q):
        return None
    return None


def strip_chart_words(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    q = s
    q = re.sub(
        r"(?i)^\s*(make|generate|create|show)\s+(a\s+)?(bar|line|pie|area|scatter|hist)?\s*(chart|plot|graph)\s*(for|of|from)?\s*",
        "",
        q,
    )
    q = re.sub(r"(?i)\s*(as|in)\s+(a\s+)?(bar|line|pie|area|scatter|hist)\s*(chart|plot|graph)\s*$", "", q)
    q = re.sub(r"(?i)\s*(bar|line|pie|area|scatter|hist)\s*(chart|plot|graph)\s*$", "", q)
    q = re.sub(r"(?i)\s*\b(chart|plot|graph)\b\s*$", "", q)
    return q.strip()


# -----------------------------
# Chart spec + inference
# -----------------------------
_ALLOWED_TYPES = {"bar", "line", "pie", "area", "hist", "scatter"}


@dataclass
class ChartSpec:
    chart_type: str  # bar|line|pie|area|hist|scatter
    x: Optional[str] = None
    y: Optional[str] = None
    y2: Optional[str] = None
    title: Optional[str] = None


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    try:
        parsed = pd.to_datetime(series.dropna().head(20), errors="coerce")
        return parsed.notna().any()
    except Exception:
        return False


def _pick_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _pick_categorical_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def infer_default_chart_spec(df: pd.DataFrame, requested_type: Optional[str], title: Optional[str] = None) -> ChartSpec:
    if df is None or df.empty:
        return ChartSpec(chart_type=requested_type or "bar", title=title)

    numeric_cols = _pick_numeric_cols(df)
    cat_cols = _pick_categorical_cols(df)
    datetime_cols = [c for c in df.columns if _is_datetime_like(df[c])]

    chart_type = requested_type
    if chart_type is None:
        if datetime_cols and numeric_cols:
            chart_type = "line"
        elif len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            chart_type = "bar"
        elif len(numeric_cols) >= 1:
            chart_type = "hist"
        else:
            chart_type = "bar"

    if chart_type in ("line", "area"):
        x = datetime_cols[0] if datetime_cols else (cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) else None))
        y = numeric_cols[0] if numeric_cols else None
        return ChartSpec(chart_type=chart_type, x=x, y=y, title=title)

    if chart_type == "scatter":
        a = numeric_cols[0] if len(numeric_cols) >= 1 else None
        b = numeric_cols[1] if len(numeric_cols) >= 2 else None
        return ChartSpec(chart_type="scatter", x=a, y=b, title=title)

    if chart_type == "pie":
        x = cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) else None)
        y = numeric_cols[0] if numeric_cols else None
        return ChartSpec(chart_type="pie", x=x, y=y, title=title)

    if chart_type == "hist":
        y = numeric_cols[0] if numeric_cols else None
        return ChartSpec(chart_type="hist", x=y, title=title)

    x = cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) else None)
    y = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) >= 2 else None)
    return ChartSpec(chart_type="bar", x=x, y=y, title=title)


def chart_spec_to_dict(spec: ChartSpec) -> Dict[str, Any]:
    return {
        "chart_type": spec.chart_type,
        "x": spec.x,
        "y": spec.y,
        "y2": spec.y2,
        "title": spec.title,
    }


def chart_spec_from_dict(d: Optional[Dict[str, Any]]) -> Optional[ChartSpec]:
    if not d:
        return None
    return ChartSpec(
        chart_type=d.get("chart_type") or "bar",
        x=d.get("x"),
        y=d.get("y"),
        y2=d.get("y2"),
        title=d.get("title"),
    )


# -----------------------------
# ✅ Strict validator (grounding)
# -----------------------------
def validate_chart_spec(df: pd.DataFrame, spec: ChartSpec) -> ChartSpec:
    if df is None or df.empty:
        # allow chart without axes; UI can show message
        spec.chart_type = spec.chart_type if spec.chart_type in _ALLOWED_TYPES else "bar"
        return spec

    cols = set(df.columns)

    ctype = (spec.chart_type or "bar").lower().strip()
    if ctype not in _ALLOWED_TYPES:
        ctype = "bar"
    spec.chart_type = ctype

    def exists(col: Optional[str]) -> Optional[str]:
        if not col:
            return None
        return col if col in cols else None

    spec.x = exists(spec.x)
    spec.y = exists(spec.y)
    spec.y2 = exists(spec.y2)

    numeric_cols = _pick_numeric_cols(df)
    cat_cols = _pick_categorical_cols(df)
    datetime_cols = [c for c in df.columns if _is_datetime_like(df[c])]

    # Fill required fields deterministically if missing / invalid
    if spec.chart_type in ("bar", "pie"):
        if spec.x is None:
            spec.x = cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) else None)
        if spec.y is None:
            spec.y = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) >= 2 else None)

    elif spec.chart_type in ("line", "area"):
        if spec.x is None:
            spec.x = datetime_cols[0] if datetime_cols else (cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) else None))
        if spec.y is None:
            spec.y = numeric_cols[0] if numeric_cols else None

    elif spec.chart_type == "hist":
        # hist uses x as numeric column
        if spec.x is None:
            spec.x = numeric_cols[0] if numeric_cols else (df.columns[0] if len(df.columns) else None)

    elif spec.chart_type == "scatter":
        # scatter uses x + y as numeric columns
        if spec.x is None:
            spec.x = numeric_cols[0] if len(numeric_cols) >= 1 else None
        if spec.y is None:
            spec.y = numeric_cols[1] if len(numeric_cols) >= 2 else (numeric_cols[0] if len(numeric_cols) == 1 else None)

    return spec


# -----------------------------
# ✅ LLM chooser (returns only columns that exist)
# -----------------------------
def choose_chart_spec_with_llm(
    *,
    llm,  # AzureChatOpenAI (or compatible)
    user_text: str,
    df: pd.DataFrame,
    requested_type: Optional[str],
    default_title: str,
) -> ChartSpec:
    """
    Uses LLM to choose chart_type + columns, but STRICTLY validates against df.columns.
    Falls back to deterministic inference on any failure.
    """
    try:
        if df is None or df.empty:
            return infer_default_chart_spec(df, requested_type, title=default_title)

        # Keep payload small + grounded
        cols = list(df.columns)
        sample = df.head(8).to_dict("records")
        numeric_cols = _pick_numeric_cols(df)
        categorical_cols = _pick_categorical_cols(df)

        system = (
            "You select chart settings for a dataframe.\n"
            "HARD RULES:\n"
            "1) You may ONLY reference columns from the provided 'columns' list.\n"
            "2) Output MUST be strict JSON with keys: chart_type, x, y, y2, title.\n"
            "3) chart_type must be one of: bar, line, pie, area, hist, scatter.\n"
            "4) If the user asked for a type (requested_type), prefer it unless impossible.\n"
            "5) Pick sensible defaults: x is categorical/date; y is numeric.\n"
            "6) Do NOT invent columns.\n"
        )

        user_payload = {
            "user_text": user_text,
            "requested_type": requested_type,
            "columns": cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "sample_rows": sample,
        }

        raw = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
            ]
        ).content

        # Extract JSON (best-effort)
        s = (raw or "").strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM did not return JSON.")

        obj = json.loads(s[start : end + 1])

        spec = ChartSpec(
            chart_type=(obj.get("chart_type") or (requested_type or "bar")),
            x=obj.get("x"),
            y=obj.get("y"),
            y2=obj.get("y2"),
            title=obj.get("title") or default_title,
        )

        spec = validate_chart_spec(df, spec)
        return spec

    except Exception:
        # Safe fallback
        spec = infer_default_chart_spec(df, requested_type, title=default_title)
        return validate_chart_spec(df, spec)
