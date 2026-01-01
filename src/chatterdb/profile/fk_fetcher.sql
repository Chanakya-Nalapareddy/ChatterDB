SELECT
  child_table,
  child_column,
  parent_table,
  parent_column
FROM catalog.foreign_keys
ORDER BY child_table, child_column, parent_table, parent_column;
