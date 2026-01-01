from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from src.rag_semantic.config import RagConfig
from src.rag_semantic.embedder_local import LocalEmbedder
from src.rag_semantic.lancedb_store import search_vectors


def _parse_meta_json(hit: Dict[str, Any]) -> Dict[str, Any]:
    meta_json = hit.get("meta_json") or "{}"
    try:
        return json.loads(meta_json)
    except Exception:
        return {}


def retrieve(
    question: str,
    k: int = 12,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "question": ...,
        "hits": [ {doc_type, text, table, column, meta, ...}, ... ],
        "tables": [...],
        "relationships": [...]
      }
    """
    cfg = RagConfig()
    cfg.validate()

    embedder = LocalEmbedder(cfg.embedding_model_name)
    qv = embedder.embed_query(question)

    hits = search_vectors(
        db_path=str(cfg.lancedb_path),
        table_name=cfg.lancedb_table,
        query_vector=qv,
        k=k,
    )

    # Parse meta_json into meta dict for each hit
    enriched_hits: List[Dict[str, Any]] = []
    for h in hits:
        meta = _parse_meta_json(h)
        hh = dict(h)
        hh["meta"] = meta
        enriched_hits.append(hh)

    # Derive candidate tables + relationships
    tables = set()
    relationships: List[Dict[str, Any]] = []

    for h in enriched_hits:
        dt = h.get("doc_type")
        if dt in ("table", "column") and h.get("table"):
            tables.add(h["table"])

        if dt == "relationship":
            meta = h.get("meta") or {}
            # relationship docs store keys under meta_json
            if meta.get("from_table") and meta.get("to_table"):
                relationships.append(meta)
                tables.add(meta["from_table"])
                tables.add(meta["to_table"])

    # de-dup relationships
    uniq = {}
    for r in relationships:
        sig = (r.get("from_table"), r.get("from_column"), r.get("to_table"), r.get("to_column"))
        uniq[sig] = r

    return {
        "question": question,
        "hits": enriched_hits,
        "tables": sorted(tables),
        "relationships": list(uniq.values()),
    }


if __name__ == "__main__":
    out = retrieve("albums and artists", k=10)
    print("Tables:", out["tables"])
    print("Relationships:", len(out["relationships"]))
    print("Top 3 hits:")
    for h in out["hits"][:3]:
        print("-", h["doc_type"], h.get("table"), h.get("column"))
