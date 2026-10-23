CREATE TABLE contacts (
    id INTEGER PRIMARY KEY,
    chat_id TEXT NOT NULL UNIQUE,
    phone TEXT DEFAULT '',
    lid TEXT DEFAULT '',
    display_name TEXT DEFAULT '',
    language TEXT DEFAULT '',
    external_user_ref TEXT DEFAULT '',
    last_inbound_at INTEGER,
    paused_until INTEGER
);

CREATE TABLE tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    number INTEGER NOT NULL UNIQUE,
    contact_id INTEGER NOT NULL REFERENCES contacts(id),
    chat_id TEXT NOT NULL,
    state TEXT NOT NULL,
    urgent INTEGER NOT NULL DEFAULT 0,
    done_by TEXT,
    summary TEXT NOT NULL DEFAULT '',
    details_json TEXT NOT NULL DEFAULT '{}',
    missing_json TEXT NOT NULL DEFAULT '[]',
    external_user_ref TEXT DEFAULT '',
    alert_message_ids_json TEXT NOT NULL DEFAULT '[]',
    previous_ticket_id INTEGER REFERENCES tickets(id),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    done_at INTEGER,
    closed_at INTEGER,
    last_customer_at INTEGER
);

CREATE TABLE ticket_events (
    id INTEGER PRIMARY KEY,
    ticket_id INTEGER NOT NULL REFERENCES tickets(id),
    at INTEGER NOT NULL,
    actor TEXT NOT NULL,
    kind TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    message_ids_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE attachments (
    id INTEGER PRIMARY KEY,
    ticket_id INTEGER NOT NULL REFERENCES tickets(id),
    message_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    mime TEXT NOT NULL DEFAULT '',
    bytes INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    derived_text TEXT NOT NULL DEFAULT ''
);

CREATE TABLE translations (
    source_hash TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (source_hash, target_lang)
);

CREATE TABLE session_state (
    session TEXT PRIMARY KEY,
    paused_until INTEGER,
    setup_completed INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);

CREATE INDEX tickets_chat_state ON tickets(chat_id, state);
CREATE INDEX tickets_state_updated ON tickets(state, updated_at);
CREATE INDEX ticket_events_ticket_at ON ticket_events(ticket_id, at);
CREATE INDEX attachments_ticket ON attachments(ticket_id);
CREATE INDEX attachments_message ON attachments(message_id);
CREATE UNIQUE INDEX tickets_one_active_per_chat
    ON tickets(chat_id)
    WHERE state IN ('open', 'waiting_customer', 'needs_you');
