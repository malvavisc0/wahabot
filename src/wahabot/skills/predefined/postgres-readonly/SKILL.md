# Postgres read-only lookup

Runs one `SELECT` or `WITH ... SELECT` per call through the optional
Postgres driver (`wahabot[postgres]`). Use a database user that has
only `SELECT` grants; the runtime also opens a read-only transaction
and applies a statement timeout.

SQL skills are operator-only in this version. Comments, `ATTACH`,
`PRAGMA`, mutating keywords, extra statements and multi-row/oversized
results are refused or capped. Store the DSN as the `POSTGRES_DSN`
secret; it is never returned to the model, web layer, logs or errors.

There is no safe automatic check for a SQL skill. Add one only when a
read-only health view exists, otherwise enable it after the operator
verifies the connection manually.
