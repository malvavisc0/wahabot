CREATE TABLE skills (
    name TEXT PRIMARY KEY,
    lifecycle TEXT NOT NULL DEFAULT 'draft'
        CHECK (lifecycle IN ('draft', 'configured')),
    enabled INTEGER NOT NULL DEFAULT 0,
    base_urls_json TEXT NOT NULL DEFAULT '[]',
    auth_type TEXT NOT NULL DEFAULT 'none',
    secret_ref TEXT NOT NULL DEFAULT '',
    customer_chats TEXT NOT NULL DEFAULT 'none'
        CHECK (customer_chats IN ('none', 'read', 'read_write')),
    writes_need_operator INTEGER NOT NULL DEFAULT 1,
    source_url TEXT NOT NULL DEFAULT '',
    instructions TEXT NOT NULL DEFAULT '',
    skill_md TEXT NOT NULL DEFAULT '',
    draft_stage TEXT NOT NULL DEFAULT 'connect',
    draft_inputs_json TEXT NOT NULL DEFAULT '{}',
    test_config_hash TEXT NOT NULL DEFAULT '',
    test_state TEXT NOT NULL DEFAULT 'untested',
    test_reason TEXT NOT NULL DEFAULT '',
    last_tested_at INTEGER,
    failure_count INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL DEFAULT 'system',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE secrets (
    name TEXT PRIMARY KEY,
    encrypted_value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE skill_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill TEXT NOT NULL,
    ticket_id INTEGER,
    chat_id TEXT NOT NULL DEFAULT '',
    started_by TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    status INTEGER,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    request_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE INDEX skill_calls_skill_at ON skill_calls(skill, created_at);
CREATE INDEX skill_calls_ticket_at ON skill_calls(ticket_id, created_at);

CREATE TABLE skill_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill TEXT NOT NULL,
    ticket_id INTEGER,
    chat_id TEXT NOT NULL DEFAULT '',
    request_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'approved', 'done', 'failed', 'refused')),
    operator_message_id TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    resolved_at INTEGER
);

CREATE INDEX skill_requests_state_at ON skill_requests(state, created_at);
CREATE INDEX skill_requests_message ON skill_requests(operator_message_id);
