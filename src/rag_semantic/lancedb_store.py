from __future__ import annotations

from typing import Any, Dict, List, Optional

import lancedb


def connect(db_path: str):
    """
    Connect to LanceDB at the given path (folder).
    """
    return lancedb.connect(db_path)


def recreate_table(db_path: str, table_name: str, rows: List[Dict[str, Any]]):
    """
    Drops and recreates the LanceDB table. Best for early iteration so you
    don't accumulate duplicates as your semantic YAML changes.
    """
    db = connect(db_path)

    # Drop if exists
    if table_name in db.table_names():
        db.drop_table(table_name)

    # Create new
    tbl = db.create_table(table_name, data=rows)
    return tbl


def open_table(db_path: str, table_name: str):
    db = connect(db_path)
    return db.open_table(table_name)


def search_vectors(
    db_path: str,
    table_name: str,
    query_vector: List[float],
    k: int = 10,
    where: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Vector search in LanceDB.
    Optional `where` is a Lance filter expression, e.g.:
      where = "doc_type = 'table'"
    """
    tbl = open_table(db_path, table_name)

    q = tbl.search(query_vector)
    if where:
        q = q.where(where)

    return q.limit(k).to_list()
