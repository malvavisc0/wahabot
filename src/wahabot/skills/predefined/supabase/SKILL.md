# Supabase read lookup

Queries a Supabase project's REST API under the configured project
origin. Use the service key only for the tables this assistant needs,
and prefer a read-only row-level-security role.

Store the key as `SUPABASE_SERVICE_KEY`, replace the `base_url` project
origin, and point the safe check at a harmless table path. Customer
chats may read; customer and team-group writes wait for the operator's
yes.
