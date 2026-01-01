from pathlib import Path
import duckdb

RAW_SQLITE = Path("data/raw/chinook.sqlite")
WAREHOUSE_DB = Path("data/warehouse/chatterdb.duckdb")

ATTACH_NAME = "chinook"     # the attached SQLite "catalog" name inside DuckDB
SOURCE_SCHEMA = "source"    # where we materialize tables in DuckDB


def main() -> None:
    if not RAW_SQLITE.exists():
        raise FileNotFoundError(f"Missing source DB: {RAW_SQLITE}")

    WAREHOUSE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE_DB))

    # Enable SQLite extension and attach the SQLite database file
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")
    con.execute(f"ATTACH '{RAW_SQLITE.as_posix()}' AS {ATTACH_NAME} (TYPE SQLITE);")

    # Target schema for copied tables
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SOURCE_SCHEMA};")

    # Discover tables from the attached SQLite database.
    # In DuckDB, ATTACH creates a catalog (table_catalog = ATTACH_NAME),
    # and SQLite tables typically live under schema 'main'.
    tables = con.execute(f"""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_catalog = '{ATTACH_NAME}'
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name;
    """).fetchall()

    if not tables:
        # Helpful debug info if something is still off
        debug = con.execute("""
            SELECT table_catalog, table_schema, table_name, table_type
            FROM information_schema.tables
            ORDER BY table_catalog, table_schema, table_name;
        """).fetchall()
        raise RuntimeError(f"No tables found in attached SQLite database. Found: {debug[:25]}")

    # Copy each table into DuckDB (materialize into SOURCE_SCHEMA)
    for schema, t in tables:
        con.execute(
            f'CREATE OR REPLACE TABLE {SOURCE_SCHEMA}."{t}" AS '
            f'SELECT * FROM {ATTACH_NAME}."{schema}"."{t}";'
        )

    print(f"✅ Imported {len(tables)} tables into {WAREHOUSE_DB}")
    for _, t in tables:
        n = con.execute(f'SELECT COUNT(*) FROM {SOURCE_SCHEMA}."{t}";').fetchone()[0]
        print(f"  {SOURCE_SCHEMA}.{t}: {n:,} rows")

    # Optional: detach the SQLite source
    con.execute(f"DETACH {ATTACH_NAME};")
    con.close()


if __name__ == "__main__":
    main()
