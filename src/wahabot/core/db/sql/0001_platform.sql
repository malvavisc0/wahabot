CREATE TABLE intake (
    message_id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    sender TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    record TEXT NOT NULL DEFAULT '',
    received_at REAL NOT NULL,
    state TEXT NOT NULL,
    outcome TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL,
    last_error TEXT,
    burst_key TEXT,
    addressed INTEGER NOT NULL DEFAULT 1,
    owner TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    handled_at REAL
);

CREATE INDEX intake_state_next_attempt ON intake(state, next_attempt_at);
CREATE INDEX intake_chat_received ON intake(chat_id, received_at);
CREATE INDEX intake_burst_key ON intake(burst_key);

CREATE TABLE outbound_intents (
    id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    intake_ids_json TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    transport_message_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL,
    last_error TEXT,
    operator_required INTEGER NOT NULL DEFAULT 0,
    notified_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    sent_at REAL
);

CREATE INDEX outbound_state_next_attempt ON outbound_intents(state, next_attempt_at);
CREATE INDEX outbound_chat_created ON outbound_intents(chat_id, created_at);
