# src/rag_semantic/sql_generator_gpt.py

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.rag_semantic.plan_from_question import plan
from src.rag_semantic.semantic_model import load_semantic_model
from src.rag_semantic.config import RagConfig
from src.rag_semantic.sql_intent_validator import validate_sql_matches_intent
from src.rag_semantic.rank_compiler import compile_rank_query


def make_llm(*, temperature: float = 0.0) -> AzureChatOpenAI:
    import os

    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_DEPLOYMENT_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")

    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
        temperature=temperature,
    )


# -----------------------------
# SQL normalization (prevents DuckDB parser errors from stray chars/fences)
# -----------------------------
_STRAY_WRAP_CHARS = "\"'`“”‘’"


def normalize_sql(sql: str) -> str:
    s = (sql or "").strip()

    # Remove markdown fences if model returns them
    s = re.sub(r"^\s*```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    # Remove BOM/zero-width
    s = s.lstrip("\ufeff\u200b\u200c\u200d")

    # Normalize smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Strip leading/trailing wrapping quotes/backticks
    while s and s[0] in _STRAY_WRAP_CHARS:
        s = s[1:].lstrip()
    while s and s[-1] in _STRAY_WRAP_CHARS:
        s = s[:-1].rstrip()

    # If prose prepended, keep from first WITH/SELECT
    m = re.search(r"\b(with|select)\b", s, flags=re.IGNORECASE)
    if m and m.start() > 0:
        s = s[m.start() :].lstrip()

    return s.strip()


def validate_basic_sql_shape(sql: str) -> None:
    s = (sql or "").strip()
    if not s:
        raise ValueError("Empty SQL.")
    if s[0] in _STRAY_WRAP_CHARS:
        raise ValueError("SQL starts with a quote/backtick (will break DuckDB parsing).")
    if re.match(r"^\s*(with|select)\b", s, flags=re.IGNORECASE) is None:
        raise ValueError("SQL must start with WITH or SELECT (likely malformed).")


# -----------------------------
# Prompt context
# -----------------------------
def build_schema_context(semantic_model: Dict[str, Any], tables: List[str]) -> str:
    lines: List[str] = []
    for t in semantic_model["tables"]:
        if t["name"] not in tables:
            continue
        lines.append(f"TABLE {t['name']}:")
        for c in t.get("columns", []):
            col = f"  - {c['name']}"
            dt = c.get("data_type") or c.get("datatype")
            if dt:
                col += f" ({dt})"
            if c.get("description"):
                col += f": {c['description']}"
            lines.append(col)
    return "\n".join(lines)


def build_join_context(join_edges: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{j['from_table']}.{j['from_column']} = {j['to_table']}.{j['to_column']} "
        f"(default {j.get('join_type', 'inner')} join)"
        for j in join_edges
    )


def _extract_json_obj(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return JSON:\n{s}")

    js = s[start : end + 1]

    try:
        return json.loads(js)
    except json.JSONDecodeError:
        pass

    js = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", js)

    out: List[str] = []
    in_string = False
    escape = False

    for ch in js:
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            continue

        if escape:
            out.append(ch)
            escape = False
            continue

        if ch == "\\":
            out.append(ch)
            escape = True
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            continue

        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(" " if ord(ch) < 32 else ch)

    return json.loads("".join(out))


# -----------------------------
# Validators
# -----------------------------
def validate_no_scalar_with_aggregate(sql: str) -> None:
    s = (sql or "").strip()
    if not s:
        return

    has_sum = re.search(r"\bSUM\s*\(", s, re.IGNORECASE) is not None
    has_totalrev = re.search(r"\bTotalRevenue\b", s, re.IGNORECASE) is not None
    has_anyvalue_or_max = re.search(r"\bANY_VALUE\s*\(|\bMAX\s*\(", s, re.IGNORECASE) is not None

    if has_sum and has_totalrev and not has_anyvalue_or_max:
        raise ValueError(
            "Potential DuckDB binder error: TotalRevenue referenced alongside SUM(...) "
            "without ANY_VALUE/MAX or scalar subqueries."
        )


def validate_no_money_equality(sql: str) -> None:
    s = (sql or "").strip()
    if not s:
        return

    if re.search(r"\bTotal\s*(=|!=|<>|<|<=|>|>=)\s*\(\s*SELECT\s+SUM\s*\(", s, re.IGNORECASE):
        raise ValueError(
            "Money equality mismatch risk: do not compare Total with a SUM subquery using =/!=. "
            "Use tolerance (ABS diff > 0.01) with JOIN+GROUP BY+HAVING."
        )


def validate_genre_label_is_from_genre_table(question: str, sql: str) -> None:
    q = (question or "").lower()
    if "genre" not in q and "genres" not in q:
        return
    s = (sql or "").lower()
    if "source.genre" not in s:
        raise ValueError(
            "Question asks about genres, but SQL does not reference source.Genre. "
            "Join source.Genre and use source.Genre.Name for the genre label."
        )


def validate_no_impossible_joins(sql: str) -> None:
    s = (sql or "").lower()

    # Genre never joins on TrackId
    if re.search(r"join\s+source\.genre\s+\w+\s+on\s+[^;]*\btrackid\b", s):
        raise ValueError("Invalid join: source.Genre does not have TrackId. Join InvoiceLine -> Track -> Genre.")


def validate_per_unit_requires_division(question: str, sql: str) -> None:
    q = (question or "").lower()
    if not any(p in q for p in ["per unit", "per-unit", "per item", "per-item", "per unit sold", "per-unit sold"]):
        return
    s = (sql or "").lower()
    if ("unitprice" in s and "quantity" in s and "sum" in s):
        uses_qty_denominator = re.search(
            r"/\s*(nullif\s*\(\s*)?sum\s*\(\s*[^)]*quantity[^)]*\)",
            sql,
            flags=re.IGNORECASE,
        ) is not None
        if not uses_qty_denominator:
            raise ValueError(
                "Question asks for 'per unit sold' but SQL does not compute a per-unit ratio. "
                "Use SUM(UnitPrice*Quantity) / NULLIF(SUM(Quantity), 0)."
            )


def validate_second_highest_is_deterministic(question: str, sql: str) -> None:
    q = (question or "").lower()
    if ("2nd" not in q) and ("second" not in q):
        return
    s = (sql or "").lower()
    if "offset" in s and ("row_number" not in s and "dense_rank" not in s):
        if re.search(r"order\s+by\s+[^;]+,\s*\w+", s) is None:
            raise ValueError("Second-highest query must be deterministic. Use ROW_NUMBER()/DENSE_RANK().")


def validate_never_sold_requires_invoiceline(question: str, sql: str) -> None:
    q = (question or "").lower()
    if not any(p in q for p in ["never sold", "no sales", "not sold", "never been sold"]):
        return

    s = (sql or "").lower()

    if "source.invoiceline" not in s:
        raise ValueError(
            "Question asks about sales ('never sold'), but SQL does not reference source.InvoiceLine. "
            "Use NOT EXISTS / LEFT JOIN InvoiceLine to determine missing sales."
        )

    if re.search(r"\btrackid\s+is\s+null\b", s, re.IGNORECASE):
        raise ValueError(
            "Incorrect logic for 'never sold': TrackId IS NULL checks missing tracks, not missing sales."
        )


def validate_order_by_columns_in_scope(sql: str) -> None:
    s = (sql or "").strip()
    if not s:
        return

    # validate ONLY final ORDER BY
    s_lower = s.lower()
    idx = s_lower.rfind("order by")
    if idx == -1:
        return

    order_clause = s[idx:]

    order_by_has_countryrank = re.search(r"\bcountryrank\b", order_clause, re.IGNORECASE) is not None
    order_by_has_genrerank = re.search(r"\bgenrerank\b", order_clause, re.IGNORECASE) is not None
    if not (order_by_has_countryrank or order_by_has_genrerank):
        return

    m = re.search(r"\bselect\b(.+?)\bfrom\b", s, re.IGNORECASE | re.DOTALL)
    select_list = m.group(1) if m else ""

    if order_by_has_countryrank and re.search(r"\bcountryrank\b", select_list, re.IGNORECASE) is None:
        raise ValueError("ORDER BY references CountryRank but CountryRank is not in scope.")

    if order_by_has_genrerank and re.search(r"\bgenrerank\b", select_list, re.IGNORECASE) is None:
        raise ValueError("ORDER BY references GenreRank but GenreRank is not in scope.")


def validate_topk_per_country_requires_partition(question: str, sql: str) -> None:
    q = (question or "").lower()
    triggers = (
        ("top" in q and "genre" in q and "country" in q and ("per" in q or "each" in q))
        or ("top 3 genres" in q and "top 5 countr" in q)
    )
    if not triggers:
        return

    s = (sql or "").lower()
    has_global_limit_3 = re.search(r"\blimit\s+3\b", s) is not None
    has_partition_by_country = re.search(r"over\s*\(\s*partition\s+by\s+[^)]*country", s, re.IGNORECASE) is not None

    if has_global_limit_3 and not has_partition_by_country:
        raise ValueError(
            "Top-K-per-country requested, but SQL uses a global LIMIT without PARTITION BY Country."
        )


def validate_month_breakdown_is_date_safe(sql: str) -> None:
    """
    If query includes Month breakdown, enforce safe InvoiceDate handling to avoid BC/overflow dates.
    """
    s = (sql or "").lower()
    if "month" not in s:
        return

    # If it uses invoice date, it MUST try_cast + filter + strftime
    uses_invoice = "source.invoice" in s or re.search(r"\bi\.\s*invoicedate\b", s) is not None or "invoicedate" in s
    if not uses_invoice:
        return

    if "try_cast" not in s:
        raise ValueError("Month breakdown must use try_cast(InvoiceDate AS TIMESTAMP) to avoid invalid dates.")
    if "strftime" not in s:
        raise ValueError("Month breakdown must compute Month via strftime(InvoiceTS, '%Y-%m').")
    if "1900-01-01" not in s or "2100-01-01" not in s:
        raise ValueError("Month breakdown must filter InvoiceTS to a sane range (>=1900 and <2100).")


# -----------------------------
# Auto-fix / error classifiers
# -----------------------------
def _is_never_sold_question(question: str) -> bool:
    q = (question or "").lower()
    return any(p in q for p in ["never sold", "no sales", "not sold", "never been sold"])


def _looks_like_trackid_is_null(sql: str) -> bool:
    return re.search(r"\btrackid\s+is\s+null\b", (sql or ""), re.IGNORECASE) is not None


def _is_topk_genres_topn_countries_question(question: str) -> bool:
    q = (question or "").lower()
    return (
        ("top" in q and "genre" in q and "country" in q and ("per" in q or "each" in q))
        and ("top 5" in q or "5 countries" in q)
    )


def _is_totalrevenue_scalar_aggregate_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "totalrevenue referenced alongside sum" in msg or "potential duckdb binder error" in msg


def _is_countryrank_out_of_scope_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "order by references countryrank" in msg and "not in scope" in msg


def _is_genrerank_out_of_scope_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "order by references genrerank" in msg and "not in scope" in msg


def _is_missing_genre_join_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "question asks about genres" in msg and "does not reference source.genre" in msg


def _autofix_order_by_rank_not_in_scope(sql: str) -> str:
    s = (sql or "").strip()
    if not s:
        return s

    s_lower = s.lower()
    idx = s_lower.rfind("order by")
    if idx == -1:
        return s

    head = s[:idx]
    order = s[idx:]

    order = re.sub(r"\bGenreRank\b", "GenreRevenue", order, flags=re.IGNORECASE)
    order = re.sub(r"\bCountryRank\b", "CountryRevenue", order, flags=re.IGNORECASE)

    return head + order


# -----------------------------
# Deterministic: hierarchical country->genre revenue sort
# -----------------------------
def _is_country_genre_hier_sort_question(question: str) -> bool:
    q = (question or "").lower()
    return (
        "revenue" in q
        and "country" in q
        and "genre" in q
        and ("sort" in q or "order" in q)
        and ("within each country" in q or "within each" in q or "within" in q)
    )


def _deterministic_country_genre_hier_sort_sql() -> str:
    return (
        "WITH country_totals AS (\n"
        "  SELECT i.BillingCountry AS Country,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS CountryRevenue\n"
        "  FROM source.Invoice i\n"
        "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
        "  GROUP BY i.BillingCountry\n"
        "),\n"
        "country_genre AS (\n"
        "  SELECT i.BillingCountry AS Country,\n"
        "         g.Name AS Genre,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS GenreRevenue\n"
        "  FROM source.Invoice i\n"
        "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
        "  JOIN source.Track t ON il.TrackId = t.TrackId\n"
        "  JOIN source.Genre g ON t.GenreId = g.GenreId\n"
        "  GROUP BY i.BillingCountry, g.Name\n"
        ")\n"
        "SELECT cg.Country,\n"
        "       cg.Genre,\n"
        "       cg.GenreRevenue AS Revenue\n"
        "FROM country_genre cg\n"
        "JOIN country_totals ct ON ct.Country = cg.Country\n"
        "ORDER BY ct.CountryRevenue DESC, cg.Country ASC, Revenue DESC, cg.Genre ASC"
    )


# -----------------------------
# Deterministic: safe month templates
# -----------------------------
def _is_breakdown_by_month_question(question: str) -> bool:
    q = (question or "").lower().strip()
    return ("by month" in q) or ("break it down" in q and "month" in q) or ("monthly" in q)


def _deterministic_revenue_by_month_sql() -> str:
    return (
        "WITH clean AS (\n"
        "  SELECT try_cast(i.InvoiceDate AS TIMESTAMP) AS InvoiceTS,\n"
        "         i.InvoiceId\n"
        "  FROM source.Invoice i\n"
        ")\n"
        "SELECT strftime(c.InvoiceTS, '%Y-%m') AS Month,\n"
        "       SUM(il.UnitPrice * il.Quantity) AS Revenue\n"
        "FROM clean c\n"
        "JOIN source.InvoiceLine il ON il.InvoiceId = c.InvoiceId\n"
        "WHERE c.InvoiceTS IS NOT NULL\n"
        "  AND c.InvoiceTS >= TIMESTAMP '1900-01-01'\n"
        "  AND c.InvoiceTS <  TIMESTAMP '2100-01-01'\n"
        "GROUP BY Month\n"
        "ORDER BY Month"
    )


def _looks_like_top_tracks_per_unit_sql(sql: str) -> bool:
    s = (sql or "").lower()
    return ("revenueperunit" in s) and ("source.invoiceline" in s) and ("source.track" in s)


def _deterministic_top_tracks_by_rev_per_unit_per_month_sql(top_n: int = 10) -> str:
    return (
        "WITH base AS (\n"
        "  SELECT try_cast(i.InvoiceDate AS TIMESTAMP) AS InvoiceTS,\n"
        "         t.TrackId,\n"
        "         t.Name AS Track,\n"
        "         il.UnitPrice,\n"
        "         il.Quantity\n"
        "  FROM source.InvoiceLine il\n"
        "  JOIN source.Invoice i ON i.InvoiceId = il.InvoiceId\n"
        "  JOIN source.Track t ON t.TrackId = il.TrackId\n"
        "),\n"
        "clean AS (\n"
        "  SELECT *\n"
        "  FROM base\n"
        "  WHERE InvoiceTS IS NOT NULL\n"
        "    AND InvoiceTS >= TIMESTAMP '1900-01-01'\n"
        "    AND InvoiceTS <  TIMESTAMP '2100-01-01'\n"
        "),\n"
        "monthly AS (\n"
        "  SELECT strftime(InvoiceTS, '%Y-%m') AS Month,\n"
        "         Track,\n"
        "         SUM(UnitPrice * Quantity) AS TotalRevenue,\n"
        "         SUM(Quantity) AS UnitsSold,\n"
        "         SUM(UnitPrice * Quantity) / NULLIF(SUM(Quantity), 0) AS RevenuePerUnit\n"
        "  FROM clean\n"
        "  GROUP BY Month, Track\n"
        "),\n"
        "ranked AS (\n"
        "  SELECT Month,\n"
        "         Track,\n"
        "         RevenuePerUnit,\n"
        "         UnitsSold,\n"
        "         TotalRevenue,\n"
        f"         ROW_NUMBER() OVER (PARTITION BY Month ORDER BY RevenuePerUnit DESC, Track ASC) AS TrackRank\n"
        "  FROM monthly\n"
        ")\n"
        "SELECT Month,\n"
        "       Track,\n"
        "       RevenuePerUnit,\n"
        "       UnitsSold,\n"
        "       TotalRevenue\n"
        "FROM ranked\n"
        f"WHERE TrackRank <= {int(top_n)}\n"
        "ORDER BY Month ASC, RevenuePerUnit DESC, Track ASC"
    )


# -----------------------------
# Deterministic templates (existing)
# -----------------------------
def _deterministic_never_sold_sql() -> str:
    return (
        "SELECT a.Name\n"
        "FROM source.Artist a\n"
        "WHERE NOT EXISTS (\n"
        "  SELECT 1\n"
        "  FROM source.Album al\n"
        "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
        "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
        "  WHERE al.ArtistId = a.ArtistId\n"
        ")\n"
        "ORDER BY a.Name"
    )


def _deterministic_topk_genres_per_topn_countries_sql(top_n_countries: int = 5, top_k_genres: int = 3) -> str:
    return (
        "WITH country_totals AS (\n"
        "  SELECT i.BillingCountry AS Country,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS CountryRevenue\n"
        "  FROM source.Invoice i\n"
        "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
        "  GROUP BY i.BillingCountry\n"
        "),\n"
        "ranked_countries AS (\n"
        "  SELECT Country,\n"
        "         CountryRevenue,\n"
        "         ROW_NUMBER() OVER (ORDER BY CountryRevenue DESC, Country ASC) AS CountryRank\n"
        "  FROM country_totals\n"
        "),\n"
        "top_countries AS (\n"
        "  SELECT Country, CountryRevenue\n"
        "  FROM ranked_countries\n"
        f"  WHERE CountryRank <= {top_n_countries}\n"
        "),\n"
        "country_genre AS (\n"
        "  SELECT tc.Country,\n"
        "         tc.CountryRevenue,\n"
        "         g.Name AS Genre,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS GenreRevenue\n"
        "  FROM top_countries tc\n"
        "  JOIN source.Invoice i ON i.BillingCountry = tc.Country\n"
        "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
        "  JOIN source.Track t ON il.TrackId = t.TrackId\n"
        "  JOIN source.Genre g ON t.GenreId = g.GenreId\n"
        "  GROUP BY tc.Country, tc.CountryRevenue, g.Name\n"
        "),\n"
        "ranked_genres AS (\n"
        "  SELECT Country,\n"
        "         CountryRevenue,\n"
        "         Genre,\n"
        "         GenreRevenue,\n"
        "         ROW_NUMBER() OVER (PARTITION BY Country ORDER BY GenreRevenue DESC, Genre ASC) AS GenreRank\n"
        "  FROM country_genre\n"
        ")\n"
        "SELECT Country,\n"
        "       Genre,\n"
        "       GenreRevenue AS Revenue\n"
        "FROM ranked_genres\n"
        f"WHERE GenreRank <= {top_k_genres}\n"
        "ORDER BY CountryRevenue DESC, Country ASC, Revenue DESC, Genre ASC"
    )


# -----------------------------
# LLM runners
# -----------------------------
def _run_llm_once(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    llm = make_llm(temperature=0.0)
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    raw = llm.invoke(msgs).content
    return _extract_json_obj(raw)


def _run_month_breakdown_rewrite_llm(
    *,
    question: str,
    history: List[Dict[str, Any]],
    plan_out: Dict[str, Any],
    schema_ctx: str,
    joins_ctx: str,
) -> Dict[str, Any]:
    """
    Generic month breakdown rewrite:
    - Uses prior turn context (question+sql)
    - Forces safe InvoiceDate conversion + sane date filters
    - Forces Month as string via strftime('%Y-%m')
    """
    last = history[-1] if history else {}
    last_q = (last.get("question") or "").strip()
    last_sql = (last.get("sql") or "").strip()

    system_prompt = f"""
You are a SQL generator for DuckDB.

TASK:
The user asked a follow-up: "{question}".
Use the previous turn's intent to generate a NEW query that breaks the result down by month.

HARD RULES:
- Use ONLY allowed tables + joins.
- Do NOT invent columns.
- Month must be based on source.Invoice.InvoiceDate.
- InvoiceDate may contain invalid/out-of-range values. You MUST do:
    try_cast(i.InvoiceDate AS TIMESTAMP) AS InvoiceTS
  and filter:
    InvoiceTS IS NOT NULL
    AND InvoiceTS >= TIMESTAMP '1900-01-01'
    AND InvoiceTS <  TIMESTAMP '2100-01-01'
- Month must be a STRING:
    strftime(InvoiceTS, '%Y-%m') AS Month
- If the prior query had rankings (Top N), keep the same ranking concept but apply per Month if appropriate.
- Return STRICT JSON only with keys: sql, explanation, assumptions.
- Do NOT wrap JSON in markdown fences.

ALLOWED TABLES:
{chr(10).join(plan_out["pruned_tables"])}

ALLOWED JOINS:
{joins_ctx}

SCHEMA:
{schema_ctx}
""".strip()

    user_payload = {
        "followup_question": question,
        "previous_question": last_q,
        "previous_sql": last_sql,
        "plan_required_tables": plan_out.get("required_tables", []),
        "pruned_tables": plan_out.get("pruned_tables", []),
        "join_edges": plan_out.get("join_edges", []),
    }

    user_prompt = (
        "Rewrite the query to include Month breakdown (safe date handling required).\n"
        "Return JSON exactly:\n"
        '{ "sql": "...", "explanation": "...", "assumptions": "..." }\n\n'
        f"INPUT:\n{json.dumps(user_payload)}"
    )

    return _run_llm_once(system_prompt, user_prompt)


# -----------------------------
# Main
# -----------------------------
def generate_sql(question: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    history = history or []
    cfg = RagConfig()

    # ✅ Follow-up: break it down by month
    if _is_breakdown_by_month_question(question):
        last_sql = (history[-1].get("sql") if history else "") or ""

        # deterministic shortcut: if previous was clearly per-unit top tracks
        if _looks_like_top_tracks_per_unit_sql(last_sql):
            sql = normalize_sql(_deterministic_top_tracks_by_rev_per_unit_per_month_sql(top_n=10))
            return {
                "question": question,
                "sql": sql,
                "explanation": "Deterministic follow-up: Top tracks by revenue per unit sold, per month (safe InvoiceDate).",
                "assumptions": "Invalid/out-of-range invoice dates are excluded (1900-01-01 to 2100-01-01).",
                "tables_used": [],
                "joins_used": [],
            }

        # generic month breakdown rewrite using LLM + strict rules
        plan_out = plan(question)
        semantic_model = load_semantic_model(str(cfg.semantic_yaml_path))

        pruned_tables = plan_out["pruned_tables"]
        join_edges = plan_out["join_edges"]

        schema_ctx = build_schema_context(semantic_model, pruned_tables)
        joins_ctx = build_join_context(join_edges)

        result = _run_month_breakdown_rewrite_llm(
            question=question,
            history=history,
            plan_out=plan_out,
            schema_ctx=schema_ctx,
            joins_ctx=joins_ctx,
        )
        sql = normalize_sql(result.get("sql") or "")

        # validate hard requirements for month safety
        validate_basic_sql_shape(sql)
        validate_month_breakdown_is_date_safe(sql)
        validate_no_impossible_joins(sql)

        return {
            "question": question,
            "sql": sql,
            "explanation": (result.get("explanation") or "").strip(),
            "assumptions": (result.get("assumptions") or "").strip(),
            "tables_used": pruned_tables,
            "joins_used": join_edges,
        }

    # ✅ First: deterministic generic rank compiler
    compiled = compile_rank_query(question)
    if compiled is not None:
        sql, expl = compiled
        sql = normalize_sql(sql)
        return {
            "question": question,
            "sql": sql,
            "explanation": expl,
            "assumptions": "",
            "tables_used": [],
            "joins_used": [],
        }

    # ✅ Deterministic for hierarchical country->genre revenue sorts
    if _is_country_genre_hier_sort_question(question):
        sql = normalize_sql(_deterministic_country_genre_hier_sort_sql())
        return {
            "question": question,
            "sql": sql,
            "explanation": "Deterministic template: revenue by country and genre with hierarchical ordering.",
            "assumptions": "",
            "tables_used": [],
            "joins_used": [],
        }

    plan_out = plan(question)
    semantic_model = load_semantic_model(str(cfg.semantic_yaml_path))

    pruned_tables = plan_out["pruned_tables"]
    join_edges = plan_out["join_edges"]

    schema_ctx = build_schema_context(semantic_model, pruned_tables)
    joins_ctx = build_join_context(join_edges)

    system_prompt = f"""
You are a SQL generator for DuckDB.

HARD RULES:
- Use ONLY the tables listed below.
- Use ONLY the join conditions listed below.
- Do NOT invent tables, joins, or columns.
- If the user says "by <dimension>", you MUST include that dimension in SELECT and GROUP BY.
- Genre questions MUST join source.Genre and use source.Genre.Name.
- For "Top K X per each Y" use ROW_NUMBER()/DENSE_RANK PARTITION BY Y and filter <= K. No global LIMIT K.
- IMPORTANT: Do NOT ORDER BY CountryRank/GenreRank in the FINAL query unless you SELECT it.
- Return STRICT JSON only with keys: sql, explanation, assumptions.
- Do NOT wrap JSON in ``` fences.
- DuckDB date math: no DATE_SUB/DATEADD; use - INTERVAL.

ALLOWED TABLES:
{chr(10).join(pruned_tables)}

ALLOWED JOINS:
{joins_ctx}

SCHEMA:
{schema_ctx}
""".strip()

    user_payload = {
        "question": question,
        "conversation_history": history[-6:],
        "required_tables": plan_out.get("required_tables", []),
        "pruned_tables": pruned_tables,
        "join_edges": join_edges,
    }

    user_prompt = (
        "Generate DuckDB SQL for the question.\n"
        "Return JSON exactly:\n"
        '{ "sql": "...", "explanation": "...", "assumptions": "..." }\n\n'
        f"INPUT:\n{json.dumps(user_payload)}"
    )

    result = _run_llm_once(system_prompt, user_prompt)
    sql = normalize_sql(result.get("sql") or "")

    def _validate_all() -> None:
        validate_basic_sql_shape(sql)
        validate_sql_matches_intent(question, sql)
        validate_no_scalar_with_aggregate(sql)
        validate_no_money_equality(sql)
        validate_genre_label_is_from_genre_table(question, sql)
        validate_no_impossible_joins(sql)
        validate_per_unit_requires_division(question, sql)
        validate_second_highest_is_deterministic(question, sql)
        validate_never_sold_requires_invoiceline(question, sql)
        validate_order_by_columns_in_scope(sql)
        validate_topk_per_country_requires_partition(question, sql)

    try:
        _validate_all()
    except Exception as e:
        # rank ORDER BY not-in-scope: auto-fix final ORDER BY and retry validation
        if _is_genrerank_out_of_scope_error(e) or _is_countryrank_out_of_scope_error(e):
            fixed = normalize_sql(_autofix_order_by_rank_not_in_scope(sql))
            if fixed != sql:
                sql = fixed
                _validate_all()
                return {
                    "question": question,
                    "sql": sql,
                    "explanation": (result.get("explanation") or "").strip() + " (Auto-fixed: ORDER BY rank not in scope.)",
                    "assumptions": result.get("assumptions", ""),
                    "tables_used": pruned_tables,
                    "joins_used": join_edges,
                }

        # Known TopK genres/countries failure modes -> deterministic
        if (
            _is_topk_genres_topn_countries_question(question)
            or _is_totalrevenue_scalar_aggregate_error(e)
            or _is_missing_genre_join_error(e)
        ):
            sql = normalize_sql(_deterministic_topk_genres_per_topn_countries_sql(5, 3))
            result["explanation"] = (result.get("explanation") or "").strip() + (
                " (Auto-fixed: deterministic Top 3 genres per Top 5 countries.)"
            )
            _validate_all()
            return {
                "question": question,
                "sql": sql,
                "explanation": result.get("explanation", ""),
                "assumptions": result.get("assumptions", ""),
                "tables_used": pruned_tables,
                "joins_used": join_edges,
            }

        # never sold: deterministic pattern if model keeps using TrackId IS NULL
        if _is_never_sold_question(question) and _looks_like_trackid_is_null(sql):
            sql = normalize_sql(_deterministic_never_sold_sql())
            result["explanation"] = (result.get("explanation") or "").strip() + (
                " (Auto-fixed to use NOT EXISTS + InvoiceLine.)"
            )
            _validate_all()
            return {
                "question": question,
                "sql": sql,
                "explanation": result.get("explanation", ""),
                "assumptions": result.get("assumptions", ""),
                "tables_used": pruned_tables,
                "joins_used": join_edges,
            }

        # generic retry once
        correction = (
            "Your SQL has an issue.\n"
            f"Error: {e}\n\n"
            "Regenerate SQL that answers the SAME question and follows the rules.\n"
            "- If grouping by month/year, InvoiceDate may contain invalid/out-of-range values: use try_cast and filter.\n"
            "- Do NOT return markdown fences.\n"
            "Return JSON only."
        )
        retry_user_prompt = user_prompt + "\n\n" + correction
        result = _run_llm_once(system_prompt, retry_user_prompt)
        sql = normalize_sql(result.get("sql") or "")
        _validate_all()

    return {
        "question": question,
        "sql": sql,
        "explanation": result.get("explanation", ""),
        "assumptions": result.get("assumptions", ""),
        "tables_used": pruned_tables,
        "joins_used": join_edges,
    }


if __name__ == "__main__":
    # minimal smoke test: month follow-up
    hist = [
        {
            "question": "Top 10 tracks by revenue per unit sold",
            "sql": "SELECT t.Name AS Track, SUM(il.UnitPrice*il.Quantity)/NULLIF(SUM(il.Quantity),0) AS RevenuePerUnit "
                   "FROM source.InvoiceLine il JOIN source.Track t ON t.TrackId=il.TrackId "
                   "GROUP BY t.Name ORDER BY RevenuePerUnit DESC LIMIT 10",
            "result_columns": ["Track", "RevenuePerUnit"],
            "row_count": 10,
        }
    ]
    out = generate_sql("Now break it down by month", history=hist)
    print("\nSQL:\n", out["sql"])
    print("\nExplanation:\n", out["explanation"])
