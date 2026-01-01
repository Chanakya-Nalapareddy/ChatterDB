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
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# -----------------------------
# Paths (robust on Windows)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../src
PROJECT_ROOT = BASE_DIR.parent                      # project root

SEMANTIC_YAML_PATH = PROJECT_ROOT / "src" / "chatterdb" / "catalog" / "chatterdb_semantic_model.yaml"
DUCKDB_PATH = PROJECT_ROOT / "data" / "warehouse" / "chatterdb.duckdb"

# Load .env from project root explicitly
load_dotenv(PROJECT_ROOT / ".env")


# -----------------------------
# LLM response contracts
# -----------------------------
class LLMResponse(BaseModel):
    """
    Permissive model so we can normalize used_joins before strict validation.
    """
    sql: str
    used_tables: List[str] = Field(default_factory=list)
    used_joins: List[Any] = Field(default_factory=list)  # can be list[str], list[list[str]], list[dict], etc.
    brief_explanation: str


class LLMResponseStrict(BaseModel):
    """
    Strict, final response shape after normalization.
    """
    sql: str
    used_tables: List[str] = Field(default_factory=list)
    used_joins: List[List[str]] = Field(default_factory=list)  # MUST be pairs
    brief_explanation: str


# -----------------------------
# Conversation state (for follow-ups)
# -----------------------------
@dataclass
class ConversationState:
    last_question: Optional[str] = None
    last_sql: Optional[str] = None
    last_result_columns: List[str] = field(default_factory=list)


# -----------------------------
# Semantic model loading + index
# -----------------------------
def load_yaml_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_semantic_model(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_semantic_index(semantic: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds allowlists for validation:
    - tables: set of schema.table
    - allowed_edges: set of (colA, colB) edges for joins, includes reverse direction
    """
    tables = set()
    allowed_edges = set()

    for t in semantic.get("tables", []):
        tables.add(t["name"])

    for r in semantic.get("relationships", []):
        a, b = r["from"], r["to"]
        allowed_edges.add((a, b))
        allowed_edges.add((b, a))

    return {"tables": tables, "allowed_edges": allowed_edges}


# -----------------------------
# LLM output sanitizer + normalizer
# -----------------------------
def sanitize_llm_json(raw: str) -> str:
    """
    Removes ```json fences and extracts the first {...} JSON object if extra text exists.
    """
    if raw is None:
        return ""

    s = raw.strip()

    # Strip triple-backtick fences (``` or ```json)
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()

    # If there's still extra text, extract first JSON object
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1].strip()

    return s


def normalize_used_joins(used_joins: List[Any]) -> List[List[str]]:
    """
    Accepts different shapes and converts to list of [left,right] pairs.

    Supported inputs:
      1) [["a","b"],["c","d"]]
      2) ["a","b","c","d"]  -> pairs: [["a","b"],["c","d"]]
      3) [{"from":"a","to":"b"}, {"from":"c","to":"d"}]
    """
    if not used_joins:
        return []

    # Already list-of-pairs
    if all(isinstance(x, list) and len(x) == 2 for x in used_joins):
        return [[str(x[0]), str(x[1])] for x in used_joins]

    # List of dicts with from/to
    if all(isinstance(x, dict) for x in used_joins):
        pairs: List[List[str]] = []
        for d in used_joins:
            if "from" in d and "to" in d:
                pairs.append([str(d["from"]), str(d["to"])])
        return pairs

    # Flat list of strings
    if all(isinstance(x, str) for x in used_joins):
        if len(used_joins) % 2 != 0:
            raise ValueError(f"used_joins flat list must have even length; got {len(used_joins)}")
        pairs: List[List[str]] = []
        for i in range(0, len(used_joins), 2):
            pairs.append([str(used_joins[i]), str(used_joins[i + 1])])
        return pairs

    # Mixed/unknown format
    raise ValueError(f"Unrecognized used_joins format: {used_joins}")


# -----------------------------
# Safety + reference validation
# -----------------------------
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


def extract_table_refs(sql: str) -> List[str]:
    """
    Extracts schema.table occurrences after FROM/JOIN.
    """
    pattern = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*)", re.IGNORECASE)
    return pattern.findall(sql)


def validate_tables_in_sql(sql: str, idx: Dict[str, Any]) -> None:
    for t in set(extract_table_refs(sql)):
        if t not in idx["tables"]:
            raise ValueError(f"SQL references unknown/forbidden table: {t}")


def validate_used_tables(obj: LLMResponseStrict, idx: Dict[str, Any]) -> None:
    for t in obj.used_tables:
        if t not in idx["tables"]:
            raise ValueError(f"LLM used_tables contains unknown/forbidden table: {t}")


def validate_joins(obj: LLMResponseStrict, idx: Dict[str, Any]) -> None:
    """
    Validates joins against allowed relationship edges.

    Accepts:
      - Column-level join edges from YAML relationships
      - Table-level join edges when a relationship exists between tables
      - Self-joins on the same column (needed for patterns like invoice-line pairs)
    """
    allowed_edges = idx["allowed_edges"]

    # Derive allowed table pairs from allowed column edges
    allowed_table_pairs = set()
    for a, b in allowed_edges:
        ta = ".".join(a.split(".")[:2])  # source.Invoice
        tb = ".".join(b.split(".")[:2])  # source.Customer
        allowed_table_pairs.add((ta, tb))
        allowed_table_pairs.add((tb, ta))

    for a, b in obj.used_joins:
        # NEW: allow self-join equality on same column (e.g., il1.InvoiceId = il2.InvoiceId)
        if a == b:
            continue

        # Column-level join edge
        if (a, b) in allowed_edges:
            continue

        # Table-level join edge
        if a.count(".") == 1 and b.count(".") == 1:
            if (a, b) in allowed_table_pairs:
                continue

        raise ValueError(f"Join not allowed by semantic relationships: {a} <-> {b}")

# -----------------------------
# Azure OpenAI model
# -----------------------------
def make_llm() -> AzureChatOpenAI:
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_DEPLOYMENT_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Check your .env file at: {PROJECT_ROOT / '.env'}")

    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
        temperature=0,
    )


# -----------------------------
# Build LLM messages (full YAML)
# -----------------------------
def build_messages(question: str, semantic_yaml_text: str, state: ConversationState) -> List:
    system = (
        "You convert natural language questions into DuckDB SQL using ONLY the provided semantic model YAML.\n\n"
        "Hard rules:\n"
        "1) Use ONLY tables/columns present in the YAML.\n"
        "2) Use ONLY joins present in YAML relationships.\n"
        "3) Use DuckDB SQL and keep identifiers as schema.table (e.g., source.Customer).\n"
        "4) Revenue correctness rule: If the question requires joining beyond Invoice to line-level entities "
        "(Track/Album/Artist/Genre/Playlist), compute revenue as "
        "SUM(source.InvoiceLine.UnitPrice * source.InvoiceLine.Quantity). "
        "Do NOT sum Invoice.Total after joining Invoice to InvoiceLine or beyond.\n"
        "5) Dont have any limits in sql"
        "6) used_joins should be join column pairs like "
        "[['source.Invoice.CustomerId','source.Customer.CustomerId'], ...].\n"
        "7) Return STRICT JSON only with keys: sql, used_tables, used_joins, brief_explanation.\n"
        "8) Do NOT wrap JSON in ``` fences.\n"
        "9) Date/time rule for DuckDB: NEVER use 'now' as a date literal (no DATE 'now' or 'now'). "
        "Use CURRENT_DATE / CURRENT_TIMESTAMP, or (preferred for static datasets) anchor relative windows using "
        "(SELECT MAX(InvoiceDate) FROM source.Invoice) and INTERVAL arithmetic.\n"
        "10) SQL must be complete and executable: it must include SELECT and FROM, and it must not be a fragment.\n"
        "Numeric precision rule: avoid direct equality/inequality comparisons on computed numeric values (especially money). "
        "Do not use '=' or '<>' between aggregates/expressions and stored totals. "
        "Instead compare using rounding or tolerance, e.g., ROUND(x, 2) = ROUND(y, 2) or ABS(x - y) <= 0.01 (and for mismatches ABS(x - y) > 0.01). "
        "Use an appropriate tolerance for currency (typically 0.01).\n"
        "SQL grouping rule: If a query uses GROUP BY, every selected column must either be aggregated (SUM/COUNT/AVG/etc.) "
        "or explicitly included in the GROUP BY list. Do not select non-aggregated columns that are not grouped.\n"
        "Vague-threshold rule: If the user uses subjective terms like high/low/large/small/cheap/expensive/most/least "
        "and no numeric threshold is provided, DO NOT invent hard-coded thresholds in WHERE/HAVING. "
        "Instead return a ranked result (ORDER BY) to let the user decide. "
        "You may optionally use relative thresholds (e.g., top N, percentile) but must state the choice in brief_explanation.\n"
        "Text search rule: when the user asks to match words in text (contains/starts with/ends with), default to case-insensitive matching "
        "(use ILIKE or LOWER(column) LIKE). Avoid case-sensitive LIKE unless the user explicitly requests case-sensitive behavior.\n"
        "Column source rule: Only reference a column from the table that actually contains it. "
        "Do not assume date fields exist on line-item tables. For purchase dates, use the header table (e.g., Invoice.InvoiceDate). "
        "Always verify column-table membership using the semantic model before writing SQL.\n"
        "Self-join rule: self-joins are allowed when joining a table to itself on the same key column "
        "(e.g., InvoiceLine.InvoiceId = InvoiceLine.InvoiceId for pair mining). Include such joins in used_joins.\n"
    )

    followup_context = {}
    if state.last_question and state.last_sql:
        followup_context = {
            "previous_question": state.last_question,
            "previous_sql": state.last_sql,
            "previous_result_columns": state.last_result_columns,
        }

    user_payload = {
        "question": question,
        "semantic_model_yaml": semantic_yaml_text,
        "followup_context": followup_context,
        "output_schema": LLMResponseStrict.model_json_schema(),
    }

    return [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(user_payload)),
    ]


# -----------------------------
# Main ask() loop
# -----------------------------
def ask(question: str, semantic_path: Path, duckdb_path: Path, state: ConversationState) -> Dict[str, Any]:
    if not semantic_path.exists():
        raise FileNotFoundError(f"Semantic YAML not found: {semantic_path}")
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {duckdb_path}")

    semantic = load_semantic_model(semantic_path)
    idx = build_semantic_index(semantic)
    semantic_text = load_yaml_text(semantic_path)

    llm = make_llm()
    messages = build_messages(question, semantic_text, state)

    raw = llm.invoke(messages).content
    clean = sanitize_llm_json(raw)

    # permissive parse first (so used_joins can be weird)
    tmp = LLMResponse.model_validate_json(clean)

    # normalize joins then strict-validate
    normalized = {
        "sql": tmp.sql,
        "used_tables": tmp.used_tables,
        "used_joins": normalize_used_joins(tmp.used_joins),
        "brief_explanation": tmp.brief_explanation,
    }
    obj = LLMResponseStrict.model_validate(normalized)

    # Safety + semantic validation
    validate_sql_is_safe(obj.sql)
    validate_tables_in_sql(obj.sql, idx)
    validate_used_tables(obj, idx)
    validate_joins(obj, idx)

    # Execute
    con = duckdb.connect(str(duckdb_path))
    df = con.execute(obj.sql).fetchdf()
    con.close()

    # Update state
    state.last_question = question
    state.last_sql = obj.sql
    state.last_result_columns = list(df.columns)

    return {
        "sql": obj.sql,
        "brief_explanation": obj.brief_explanation,
        "preview": df.to_dict("records"),
    }


if __name__ == "__main__":
    state = ConversationState()

    out = ask("For each country, show the top 3 artists by revenue.", SEMANTIC_YAML_PATH, DUCKDB_PATH, state)
    print("SQL:\n", out["sql"])
    print("\nExplanation:\n", out["brief_explanation"])
    print("\nPreview:\n", pd.DataFrame(out["preview"]))
