from __future__ import annotations

from pathlib import Path
import duckdb

WAREHOUSE_DB = Path("data/warehouse/chatterdb.duckdb")

SOURCE_SCHEMA = "source"
CATALOG_SCHEMA = "catalog"
PROFILE_SCHEMA = "profile"


def main() -> None:
    if not WAREHOUSE_DB.exists():
        raise FileNotFoundError(f"Warehouse DB not found: {WAREHOUSE_DB}")

    con = duckdb.connect(str(WAREHOUSE_DB))

    # Create schemas for metadata
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_SCHEMA};")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {PROFILE_SCHEMA};")

    # -------------------------
    # 1) Catalog: tables/columns
    # -------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE {CATALOG_SCHEMA}.tables AS
        SELECT
            table_catalog,
            table_schema,
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = '{SOURCE_SCHEMA}'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {CATALOG_SCHEMA}.columns AS
        SELECT
            table_catalog,
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = '{SOURCE_SCHEMA}'
        ORDER BY table_name, ordinal_position;
    """)

    # -------------------------
    # 2) Profile: per-table + per-column stats
    # -------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE {PROFILE_SCHEMA}.table_stats AS
        SELECT
            t.table_name,
            CAST(NULL AS BIGINT) AS row_count
        FROM {CATALOG_SCHEMA}.tables t
        ORDER BY t.table_name;
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {PROFILE_SCHEMA}.column_stats AS
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            CAST(NULL AS BIGINT) AS row_count,
            CAST(NULL AS BIGINT) AS null_count,
            CAST(NULL AS BIGINT) AS approx_distinct
        FROM {CATALOG_SCHEMA}.columns c
        WHERE 1=0;
    """)

    tables = con.execute(
        f"SELECT table_name FROM {CATALOG_SCHEMA}.tables ORDER BY table_name;"
    ).fetchall()

    # Fill in table row counts
    for (t,) in tables:
        row_count = con.execute(
            f'SELECT COUNT(*) FROM {SOURCE_SCHEMA}."{t}";'
        ).fetchone()[0]
        con.execute(
            f"UPDATE {PROFILE_SCHEMA}.table_stats SET row_count=? WHERE table_name=?;",
            [row_count, t],
        )

    # Fill in per-column stats
    con.execute(f"DELETE FROM {PROFILE_SCHEMA}.column_stats;")
    for (t,) in tables:
        cols = con.execute(f"""
            SELECT column_name, data_type
            FROM {CATALOG_SCHEMA}.columns
            WHERE table_name = ?
            ORDER BY ordinal_position;
        """, [t]).fetchall()

        row_count = con.execute(f'SELECT COUNT(*) FROM {SOURCE_SCHEMA}."{t}";').fetchone()[0]

        for col_name, data_type in cols:
            null_count = con.execute(
                f'SELECT COUNT(*) FROM {SOURCE_SCHEMA}."{t}" WHERE "{col_name}" IS NULL;'
            ).fetchone()[0]

            approx_distinct = con.execute(
                f'SELECT approx_count_distinct("{col_name}") FROM {SOURCE_SCHEMA}."{t}";'
            ).fetchone()[0]

            con.execute(f"""
                INSERT INTO {PROFILE_SCHEMA}.column_stats
                (table_name, column_name, data_type, row_count, null_count, approx_distinct)
                VALUES (?, ?, ?, ?, ?, ?);
            """, [t, col_name, data_type, row_count, null_count, approx_distinct])

    # Output a tiny summary
    table_count = con.execute(f"SELECT COUNT(*) FROM {CATALOG_SCHEMA}.tables;").fetchone()[0]
    col_count = con.execute(f"SELECT COUNT(*) FROM {CATALOG_SCHEMA}.columns;").fetchone()[0]
    print(f"✅ Scan complete. Cataloged {table_count} tables and {col_count} columns.")
    print("Top 5 widest tables (by # columns):")
    widest = con.execute(f"""
        SELECT table_name, COUNT(*) AS column_count
        FROM {CATALOG_SCHEMA}.columns
        GROUP BY table_name
        ORDER BY column_count DESC, table_name
        LIMIT 5;
    """).fetchall()
    for t, n in widest:
        print(f"  {t}: {n} columns")

    con.close()


if __name__ == "__main__":
    main()
