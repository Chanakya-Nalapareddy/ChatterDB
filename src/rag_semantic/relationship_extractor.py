from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _split_reference(ref: str) -> Optional[Tuple[str, str]]:
    """
    Your YAML uses references like: 'source.Artist.ArtistId'
    We interpret it as: table='source.Artist', column='ArtistId'
    """
    if not ref or not isinstance(ref, str):
        return None

    parts = [p for p in ref.split(".") if p]
    if len(parts) < 2:
        return None

    # last part is column; everything before is table
    col = parts[-1]
    table = ".".join(parts[:-1])
    return table, col


def extract_relationships_from_foreign_keys(raw_semantic: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract relationships from tables[].foreign_keys blocks.
    """
    rels: List[Dict[str, Any]] = []

    tables = raw_semantic.get("tables") or []
    for t in tables:
        table_name = t.get("name")  # in your YAML, tables have "name" like "source.Album"
        if not table_name:
            continue

        fk_list = t.get("foreign_keys") or []
        join_type_default = t.get("join_type_default") or "inner"
        relationship_type = t.get("relationship_type") or ""
        fanout_risk = t.get("fanout_risk") or ""
        preferred = bool(t.get("preferred", False))

        for fk in fk_list:
            from_col = fk.get("column")
            ref = fk.get("references") or fk.get("reference")
            parsed = _split_reference(ref)
            if not from_col or not parsed:
                continue

            to_table, to_col = parsed
            rels.append({
                "from_table": table_name,
                "from_column": from_col,
                "to_table": to_table,
                "to_column": to_col,
                "join_type": fk.get("join_type") or join_type_default,
                "relationship_type": fk.get("relationship_type") or relationship_type,
                "fanout_risk": fk.get("fanout_risk") or fanout_risk,
                "preferred": bool(fk.get("preferred", preferred)),
                "manual": bool(fk.get("manual", False)),
            })

    return rels


def extract_relationships(raw_semantic: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Unified extractor:
    - foreign keys (your primary structure)
    - optionally: top-level 'relationships' if present in other shapes
      (we keep it conservative and only add if it matches expected keys)
    """
    rels = extract_relationships_from_foreign_keys(raw_semantic)

    # Optional: parse top-level relationships if they exist with explicit from/to
    top_rels = raw_semantic.get("relationships") or []
    for r in top_rels:
        # your inspect showed keys: from, to (counts 11 each)
        # If it's shaped like:
        # { "from": "source.A.Col", "to": "source.B.Col", ... }
        f = r.get("from")
        t = r.get("to")
        pf = _split_reference(f) if isinstance(f, str) else None
        pt = _split_reference(t) if isinstance(t, str) else None
        if pf and pt:
            from_table, from_col = pf
            to_table, to_col = pt
            rels.append({
                "from_table": from_table,
                "from_column": from_col,
                "to_table": to_table,
                "to_column": to_col,
                "join_type": r.get("join_type") or r.get("join_type_default") or "inner",
                "relationship_type": r.get("relationship_type") or "",
                "fanout_risk": r.get("fanout_risk") or "",
                "preferred": bool(r.get("preferred", False)),
                "manual": bool(r.get("manual", False)),
            })

    # de-dup
    uniq = {}
    for rr in rels:
        sig = (rr["from_table"], rr["from_column"], rr["to_table"], rr["to_column"])
        uniq[sig] = rr
    return list(uniq.values())
