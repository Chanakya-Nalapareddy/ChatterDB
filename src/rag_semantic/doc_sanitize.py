from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _pk_to_str(pk: Any) -> str:
    if pk is None:
        return ""
    if isinstance(pk, list):
        return ",".join(str(x) for x in pk if x is not None)
    return str(pk)


def sanitize_docs_for_lancedb(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    PyArrow requires consistent column types across rows.
    We sanitize to make LanceDB ingestion stable.

    Strategy:
      - Keep top-level columns as simple scalars (str / None)
      - Convert meta dict into a JSON string (meta_json)
      - Ensure any mixed-type values (like primary_key list vs str) are normalized
    """
    out: List[Dict[str, Any]] = []

    for d in docs:
        meta = d.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"value": meta}

        # Normalize known problematic fields inside meta
        if "primary_key" in meta:
            meta["primary_key"] = _pk_to_str(meta.get("primary_key"))

        # Convert meta to JSON string so schema is stable
        meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)

        out.append({
            "id": str(d.get("id", "")),
            "doc_type": str(d.get("doc_type", "")),
            "source": str(d.get("source", "")),
            "table": d.get("table") if d.get("table") is None else str(d.get("table")),
            "column": d.get("column") if d.get("column") is None else str(d.get("column")),
            "text": str(d.get("text", "")),
            "vector": d.get("vector"),  # list[float]
            "meta_json": meta_json,
        })

    return out
