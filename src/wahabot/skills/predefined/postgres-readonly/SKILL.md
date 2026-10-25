# Postgres read-only lookup

Runs one `SELECT` or `WITH ... SELECT` per call through the optional
Postgres driver (`wahabot[postgres]`). Use a database user that has
only `SELECT` grants; the runtime also opens a read-only transaction
and applies a statement timeout.

SQL skills are operator-only in this version. Comments, `ATTACH`,
`PRAGMA`, mutating keywords, extra statements and multi-row/oversized
results are refused or capped. Store the DSN as the `POSTGRES_DSN`
secret; it is never returned to the model, web layer, logs or errors.

The safe check is a read-only `SELECT 1` probe before the skill can be
enabled; it does not prove that your own tables are reachable, so verify a
real lookup with the operator chat after enabling.
