from __future__ import annotations

import uuid
from typing import Any, Dict, List


def _mk_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def build_embedding_docs(semantic_model: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    """
    Input: semantic_model from load_semantic_model():
      { tables: [...], relationships: [...], raw: {...} }

    Output: list of docs suitable for embedding + LanceDB storage.
    Each doc includes a `text` field that we embed.
    """
    docs: List[Dict[str, Any]] = []

    tables = semantic_model.get("tables", []) or []
    rels = semantic_model.get("relationships", []) or []

    # ---- TABLE + COLUMN docs ----
    for t in tables:
        tname = t.get("name")
        if not tname:
            continue

        desc = t.get("description", "") or ""
        pk = t.get("primary_key", "") or ""
        grain = t.get("grain", "") or ""

        # Table doc text: dense but readable, helps retrieval.
        table_text = "\n".join([
            f"TABLE {tname}",
            f"Description: {desc}" if desc else "Description: (none)",
            f"Primary key: {pk}" if pk else "Primary key: (unknown)",
            f"Grain: {grain}" if grain else "Grain: (unspecified)",
            "Use cases: analytics, reporting, joins via foreign keys.",
        ])

        docs.append({
            "id": _mk_id("tbl"),
            "doc_type": "table",
            "source": source,
            "table": tname,
            "column": None,
            "text": table_text,
            "meta": {
                "primary_key": pk,
                "grain": grain,
                "description": desc,
            },
        })

        # Column docs
        for c in t.get("columns", []) or []:
            cname = c.get("name")
            if not cname:
                continue

            dtype = c.get("datatype", "") or ""
            cdesc = c.get("description", "") or ""
            synonyms = c.get("synonyms", []) or []
            role = c.get("role")

            col_text = "\n".join([
                f"COLUMN {tname}.{cname}",
                f"Type: {dtype}" if dtype else "Type: (unknown)",
                f"Role: {role}" if role else "Role: (unspecified)",
                f"Description: {cdesc}" if cdesc else "Description: (none)",
                f"Synonyms: {', '.join(synonyms)}" if synonyms else "Synonyms: (none)",
            ])

            docs.append({
                "id": _mk_id("col"),
                "doc_type": "column",
                "source": source,
                "table": tname,
                "column": cname,
                "text": col_text,
                "meta": {
                    "datatype": dtype,
                    "description": cdesc,
                    "synonyms": synonyms,
                    "role": role,
                    "is_pk": bool(c.get("is_pk", False)),
                    "is_fk": bool(c.get("is_fk", False)),
                },
            })

    # ---- RELATIONSHIP docs ----
    for r in rels:
        ft = r.get("from_table")
        fc = r.get("from_column")
        tt = r.get("to_table")
        tc = r.get("to_column")
        if not (ft and fc and tt and tc):
            continue

        join_type = r.get("join_type", "inner")
        rel_type = r.get("relationship_type", "")
        fanout = r.get("fanout_risk", "")
        preferred = bool(r.get("preferred", False))

        rel_text = "\n".join([
            "RELATIONSHIP (foreign key join)",
            f"From: {ft}.{fc}",
            f"To:   {tt}.{tc}",
            f"Join type default: {join_type}",
            f"Relationship type: {rel_type}" if rel_type else "Relationship type: (unspecified)",
            f"Fanout risk: {fanout}" if fanout else "Fanout risk: (unspecified)",
            f"Preferred: {preferred}",
            "Meaning: rows in From usually map to one row in To (dimension/lookup join).",
        ])

        docs.append({
            "id": _mk_id("rel"),
            "doc_type": "relationship",
            "source": source,
            "table": None,
            "column": None,
            "text": rel_text,
            "meta": {
                "from_table": ft,
                "from_column": fc,
                "to_table": tt,
                "to_column": tc,
                "join_type": join_type,
                "relationship_type": rel_type,
                "fanout_risk": fanout,
                "preferred": preferred,
                "manual": bool(r.get("manual", False)),
            },
        })

    return docs
