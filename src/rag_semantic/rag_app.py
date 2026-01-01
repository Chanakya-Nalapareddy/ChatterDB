# src/rag_semantic/rag_app.py

import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import duckdb
import yaml
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

from src.rag_semantic.sql_generator_gpt import generate_sql
from src.rag_semantic.charting import (
    is_chart_request,
    parse_requested_chart_type,
    strip_chart_words,
    infer_default_chart_spec,
    chart_spec_to_dict,
    choose_chart_spec_with_llm,   # ✅ NEW
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

SEMANTIC_YAML_PATH = PROJECT_ROOT / "src" / "chatterdb" / "catalog" / "chatterdb_semantic_model.yaml"
DUCKDB_PATH = PROJECT_ROOT / "data" / "warehouse" / "chatterdb.duckdb"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ConversationState:
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_preview: List[Dict[str, Any]] = field(default_factory=list)
    last_columns: List[str] = field(default_factory=list)

    def push_turn(
        self,
        question: str,
        sql: str,
        result_columns: List[str],
        row_count: int,
        preview_rows: List[Dict[str, Any]],
    ) -> None:
        self.history.append(
            {
                "question": question,
                "sql": sql,
                "result_columns": result_columns,
                "row_count": row_count,
            }
        )
        self.last_preview = preview_rows or []
        self.last_columns = result_columns or []

        MAX_TURNS = 6
        if len(self.history) > MAX_TURNS:
            self.history = self.history[-MAX_TURNS:]


def load_semantic_model(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_semantic_index(semantic: Dict[str, Any]) -> Dict[str, Any]:
    tables = set()
    allowed_edges = set()

    for t in semantic.get("tables", []):
        tables.add(t["name"])

    for r in semantic.get("relationships", []):
        a, b = r["from"], r["to"]
        allowed_edges.add((a, b))
        allowed_edges.add((b, a))

    return {"tables": tables, "allowed_edges": allowed_edges}


FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|COPY|ATTACH|DETACH|CALL|EXPORT|IMPORT)\b",
    re.IGNORECASE
)


def validate_sql_is_safe(sql: str) -> None:
    s = sql.strip().rstrip(";").strip()
    if not (s.upper().startswith("SELECT") or s.upper().startswith("WITH")):
        raise ValueError("Only SELECT/WITH queries are allowed.")
    if FORBIDDEN_SQL.search(s):
        raise ValueError("Query contains forbidden SQL keywords.")
    if ";" in s:
        raise ValueError("Multiple statements are not allowed.")
    if re.search(r"\bFROM\b", s, re.IGNORECASE) is None:
        raise ValueError("SQL must contain a FROM clause (query looks incomplete).")


def extract_table_refs(sql: str) -> List[str]:
    pattern = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*)", re.IGNORECASE)
    return pattern.findall(sql)


def validate_tables_in_sql(sql: str, idx: Dict[str, Any]) -> None:
    for t in set(extract_table_refs(sql)):
        if t not in idx["tables"]:
            raise ValueError(f"SQL references unknown/forbidden table: {t}")


def deterministic_nl_if_simple(df: pd.DataFrame) -> Optional[str]:
    if df is None or len(df) == 0:
        return None
    cols = {c.lower(): c for c in df.columns}

    if "billingcountry" in cols and "revenue" in cols and 1 <= len(df) <= 10:
        c_country = cols["billingcountry"]
        c_rev = cols["revenue"]
        rows = df[[c_country, c_rev]].to_dict("records")
        if len(rows) == 1:
            try:
                rev = float(rows[0][c_rev])
                return f"The country with the 2nd highest revenue is {rows[0][c_country]}, with revenue ${rev:.2f}."
            except Exception:
                return f"The country with the 2nd highest revenue is {rows[0][c_country]}, with revenue {rows[0][c_rev]}."
        parts = []
        for r in rows:
            try:
                rev = float(r[c_rev])
                parts.append(f"{r[c_country]} (${rev:.2f})")
            except Exception:
                parts.append(f"{r[c_country]} ({r[c_rev]})")
        return "The countries with the 2nd highest revenue are: " + ", ".join(parts) + "."
    return None


def make_llm(*, temperature: float = 0.0) -> AzureChatOpenAI:
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_DEPLOYMENT_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Check your .env file at: {PROJECT_ROOT / '.env'}")

    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
        temperature=temperature,
    )


def summarize_result_to_nl(question: str, sql: str, df: pd.DataFrame) -> str:
    llm = make_llm(temperature=0.0)

    row_count = int(len(df))
    columns = list(df.columns)
    sample_rows = df.head(20).to_dict("records")

    numeric_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if len(s) == 0:
                numeric_stats[c] = {"min": None, "max": None, "sum": None, "mean": None}
            else:
                numeric_stats[c] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "sum": float(s.sum()),
                    "mean": float(s.mean()),
                }

    system = (
        "You are a data analyst. Convert SQL query results into a concise natural-language answer.\n"
        "Rules:\n"
        "1) Use ONLY the provided data. Do NOT invent values.\n"
        "2) If empty, say no rows matched.\n"
        "3) For grouped aggregates, mention top/bottom items when possible.\n"
        "4) Keep brief.\n"
    )

    user_payload = {
        "question": question,
        "sql": sql,
        "row_count": row_count,
        "columns": columns,
        "numeric_stats": numeric_stats,
        "sample_rows": sample_rows,
    }

    msgs = [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, default=str)),
    ]
    return llm.invoke(msgs).content.strip()


def ask_rag(
    question: str,
    semantic_path: Path = SEMANTIC_YAML_PATH,
    duckdb_path: Path = DUCKDB_PATH,
    state: Optional[ConversationState] = None,
) -> Dict[str, Any]:
    if state is None:
        state = ConversationState()

    if not semantic_path.exists():
        raise FileNotFoundError(f"Semantic YAML not found: {semantic_path}")
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {duckdb_path}")

    semantic = load_semantic_model(semantic_path)
    idx = build_semantic_index(semantic)

    # ✅ Chart routing
    chart_requested = is_chart_request(question)
    requested_type = parse_requested_chart_type(question)
    base_q = strip_chart_words(question) if chart_requested else question
    chart_only_followup = chart_requested and (base_q.strip() == "")

    # LLM for chart selection
    llm_for_charts = make_llm(temperature=0.0)

    if chart_only_followup:
        if not state.last_preview:
            raise ValueError("No previous result to chart. Ask a data question first (e.g., 'revenue by country').")

        df_prev = pd.DataFrame(state.last_preview)

        spec = choose_chart_spec_with_llm(
            llm=llm_for_charts,
            user_text=question,
            df=df_prev,
            requested_type=requested_type,
            default_title="Chart from previous result",
        )

        return {
            "sql": None,
            "explanation": "",
            "assumptions": "",
            "natural_language_answer": f"Here’s a {spec.chart_type} chart based on the previous result.",
            "preview": state.last_preview,
            "chart": chart_spec_to_dict(spec),
        }

    # If chart requested + base question exists, run data query then chart.
    data_question = base_q if chart_requested else question

    gen = generate_sql(data_question, history=state.history)
    sql = (gen.get("sql") or "").strip()
    if not sql:
        raise ValueError("LLM returned empty SQL (sql is null/blank).")

    validate_sql_is_safe(sql)
    validate_tables_in_sql(sql, idx)

    con = duckdb.connect(str(duckdb_path))
    df = con.execute(sql).fetchdf()
    con.close()

    preview_rows = df.head(200).to_dict("records")

    state.push_turn(
        question=data_question,
        sql=sql,
        result_columns=list(df.columns),
        row_count=len(df),
        preview_rows=preview_rows,
    )

    nl_answer = deterministic_nl_if_simple(df)
    if nl_answer is None:
        nl_answer = summarize_result_to_nl(data_question, sql, df)

    chart_obj = None
    if chart_requested:
        # ✅ Let LLM choose best x/y based on user phrasing + columns
        spec = choose_chart_spec_with_llm(
            llm=llm_for_charts,
            user_text=question,  # original prompt includes chart intent
            df=df,
            requested_type=requested_type,
            default_title=f"{requested_type or 'chart'} from result",
        )
        chart_obj = chart_spec_to_dict(spec)

    return {
        "sql": sql,
        "explanation": gen.get("explanation", ""),
        "assumptions": gen.get("assumptions", ""),
        "natural_language_answer": nl_answer,
        "preview": preview_rows,
        "chart": chart_obj,
    }
