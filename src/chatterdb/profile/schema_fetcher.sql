
-- DUCKDB: TABLES + COLUMNS (run in DuckDB)
SELECT
  c.table_catalog,
  c.table_schema,
  c.table_name,
  t.table_type,
  c.ordinal_position,
  c.column_name,
  c.data_type,
  c.is_nullable,
  c.column_default
FROM information_schema.columns c
JOIN information_schema.tables t
  ON t.table_catalog = c.table_catalog
 AND t.table_schema  = c.table_schema
 AND t.table_name    = c.table_name
WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog')
ORDER BY c.table_schema, c.table_name, c.ordinal_position;
