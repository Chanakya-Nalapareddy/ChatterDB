from __future__ import annotations

from pathlib import Path
import duckdb

WAREHOUSE_DB = Path("data/warehouse/chatterdb.duckdb")

SOURCE_SCHEMA = "source"
CATALOG_SCHEMA = "catalog"


def main() -> None:
    if not WAREHOUSE_DB.exists():
        raise FileNotFoundError(f"Warehouse DB not found: {WAREHOUSE_DB}")

    con = duckdb.connect(str(WAREHOUSE_DB))
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_SCHEMA};")

    # Ensure catalog.columns exists (created by scan_warehouse.py)
    n_cols = con.execute(f"SELECT COUNT(*) FROM {CATALOG_SCHEMA}.columns;").fetchone()[0]
    if n_cols == 0:
        raise RuntimeError("catalog.columns is empty. Run scan_warehouse.py first.")

    # Load table->columns map from catalog
    rows = con.execute(f"""
        SELECT table_name, column_name
        FROM {CATALOG_SCHEMA}.columns
        WHERE table_schema = '{SOURCE_SCHEMA}'
        ORDER BY table_name, ordinal_position;
    """).fetchall()

    table_cols: dict[str, set[str]] = {}
    for t, c in rows:
        table_cols.setdefault(t, set()).add(c)

    tables = sorted(table_cols.keys())

    # 1) Infer primary keys: <TableName>Id pattern (works well for Chinook)
    pk_rows: list[tuple[str, str]] = []
    for t in tables:
        pk_candidate = f"{t}Id"
        if pk_candidate in table_cols[t]:
            pk_rows.append((t, pk_candidate))

    con.execute(f"""
        CREATE OR REPLACE TABLE {CATALOG_SCHEMA}.primary_keys (
            table_name VARCHAR,
            column_name VARCHAR
        );
    """)
    con.execute(f"DELETE FROM {CATALOG_SCHEMA}.primary_keys;")
    for r in pk_rows:
        con.execute(f"INSERT INTO {CATALOG_SCHEMA}.primary_keys VALUES (?, ?);", r)

    # 2) Infer foreign keys: column matches a known PK column name in another table
    pk_by_col = {col: t for (t, col) in pk_rows}  # e.g. "ArtistId" -> "Artist"

    fk_rows: list[tuple[str, str, str, str]] = []
    for child_table in tables:
        for col in sorted(table_cols[child_table]):
            parent_table = pk_by_col.get(col)
            if parent_table and parent_table != child_table:
                fk_rows.append((child_table, col, parent_table, col))

    con.execute(f"""
        CREATE OR REPLACE TABLE {CATALOG_SCHEMA}.foreign_keys (
            child_table VARCHAR,
            child_column VARCHAR,
            parent_table VARCHAR,
            parent_column VARCHAR
        );
    """)
    con.execute(f"DELETE FROM {CATALOG_SCHEMA}.foreign_keys;")
    for r in fk_rows:
        con.execute(f"INSERT INTO {CATALOG_SCHEMA}.foreign_keys VALUES (?, ?, ?, ?);", r)

    # Add known self-reference in Chinook: Employee.ReportsTo -> Employee.EmployeeId
    if "Employee" in table_cols and "ReportsTo" in table_cols["Employee"] and "EmployeeId" in table_cols["Employee"]:
        con.execute(
            f"INSERT INTO {CATALOG_SCHEMA}.foreign_keys VALUES (?, ?, ?, ?);",
            ("Employee", "ReportsTo", "Employee", "EmployeeId"),
        )

    print(f"✅ Inferred {len(pk_rows)} primary keys.")
    print(f"✅ Inferred {len(fk_rows) + 1} foreign keys (including Employee.ReportsTo).")

    con.close()


if __name__ == "__main__":
    main()
