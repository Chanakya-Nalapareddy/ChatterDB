from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class NormalizedColumn:
    name: str
    datatype: str = ""
    description: str = ""
    synonyms: List[str] = None
    is_pk: bool = False
    is_fk: bool = False
    role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "datatype": self.datatype,
            "description": self.description,
            "synonyms": self.synonyms or [],
            "is_pk": self.is_pk,
            "is_fk": self.is_fk,
            "role": self.role,
        }


@dataclass
class NormalizedTable:
    name: str
    description: str = ""
    primary_key: str = ""
    grain: str = ""
    columns: List[NormalizedColumn] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "primary_key": self.primary_key,
            "grain": self.grain,
            "columns": [c.to_dict() for c in (self.columns or [])],
        }


@dataclass
class NormalizedRelationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    join_type: str = "inner"
    cardinality: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "join_type": self.join_type,
            "cardinality": self.cardinality,
        }


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick_tables(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    # common options: tables / models
    return _as_list(root.get("tables") or root.get("models") or root.get("entities"))


def _pick_relationships(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    # common options: relationships / joins / relations
    return _as_list(root.get("relationships") or root.get("joins") or root.get("relations"))


def normalize_semantic_model(raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (tables, relationships) as lists of dicts with stable keys.
    """
    norm_tables: List[NormalizedTable] = []
    norm_rels: List[NormalizedRelationship] = []

    # --- tables ---
    for t in _pick_tables(raw):
        tname = t.get("name") or t.get("table") or t.get("table_name")
        if not tname:
            continue

        tdesc = t.get("description") or ""
        pk = t.get("primary_key") or t.get("pk") or ""
        grain = t.get("grain") or ""

        cols_raw = _as_list(t.get("columns") or t.get("fields") or [])
        cols: List[NormalizedColumn] = []

        for c in cols_raw:
            cname = c.get("name") or c.get("column") or c.get("field")
            if not cname:
                continue

            dtype = c.get("datatype") or c.get("data_type") or c.get("type") or ""
            cdesc = c.get("description") or ""
            syn = c.get("synonyms") or c.get("aliases") or []
            syn = syn if isinstance(syn, list) else [syn]

            cols.append(
                NormalizedColumn(
                    name=cname,
                    datatype=dtype,
                    description=cdesc,
                    synonyms=[s for s in syn if s],
                    is_pk=bool(c.get("is_pk", False)),
                    is_fk=bool(c.get("is_fk", False)),
                    role=c.get("role") or c.get("semantic_role"),
                )
            )

        norm_tables.append(
            NormalizedTable(
                name=tname,
                description=tdesc,
                primary_key=pk,
                grain=grain,
                columns=cols,
            )
        )

    # --- relationships ---
    for r in _pick_relationships(raw):
        ft = r.get("from_table") or r.get("from") or r.get("left_table")
        fc = r.get("from_column") or r.get("from_col") or r.get("left_column")
        tt = r.get("to_table") or r.get("to") or r.get("right_table")
        tc = r.get("to_column") or r.get("to_col") or r.get("right_column")
        if not (ft and fc and tt and tc):
            continue

        norm_rels.append(
            NormalizedRelationship(
                from_table=ft,
                from_column=fc,
                to_table=tt,
                to_column=tc,
                join_type=r.get("join_type") or r.get("type") or "inner",
                cardinality=r.get("cardinality") or r.get("card") or "",
            )
        )

    return [t.to_dict() for t in norm_tables], [r.to_dict() for r in norm_rels]


def load_and_normalize(path: str) -> Dict[str, Any]:
    raw = load_yaml(path)
    tables, rels = normalize_semantic_model(raw)
    return {"tables": tables, "relationships": rels}
