CREATE TABLE alert_intents (
    alert_key TEXT PRIMARY KEY,
    ticket_id INTEGER NOT NULL,
    session TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    text TEXT NOT NULL,
    state TEXT NOT NULL,
    sent_id TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX alert_intents_state_updated ON alert_intents(state, updated_at);
