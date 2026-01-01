from __future__ import annotations

from typing import Any, Dict
from .semantic_loader import load_yaml, normalize_semantic_model
from .relationship_extractor import extract_relationships


def load_semantic_model(path: str) -> Dict[str, Any]:
    """
    Loads YAML and returns:
      - tables: normalized tables
      - relationships: extracted relationships (from foreign_keys + optional top-level)
      - raw: raw yaml (kept for debugging)
    """
    raw = load_yaml(path)
    tables, _rels_unused = normalize_semantic_model(raw)  # tables are correct already
    rels = extract_relationships(raw)
    return {"tables": tables, "relationships": rels, "raw": raw}
