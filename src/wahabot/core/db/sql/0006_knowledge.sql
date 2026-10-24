-- Phase 3: published answers, their FTS index, use ledger and reply
-- candidates; deferred operator notices; summary/night state; page counts.

CREATE TABLE answers (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    deleted_at INTEGER
);

CREATE VIRTUAL TABLE answers_fts USING fts5(
    question,
    answer,
    content='answers',
    content_rowid='id'
);

CREATE TRIGGER answers_ai AFTER INSERT ON answers BEGIN
    INSERT INTO answers_fts(rowid, question, answer)
    VALUES (new.id, new.question, new.answer);
END;

CREATE TRIGGER answers_ad AFTER DELETE ON answers BEGIN
    INSERT INTO answers_fts(answers_fts, rowid, question, answer)
    VALUES ('delete', old.id, old.question, old.answer);
END;

CREATE TRIGGER answers_au AFTER UPDATE ON answers BEGIN
    INSERT INTO answers_fts(answers_fts, rowid, question, answer)
    VALUES ('delete', old.id, old.question, old.answer);
    INSERT INTO answers_fts(rowid, question, answer)
    VALUES (new.id, new.question, new.answer);
END;

CREATE TABLE answer_uses (
    id INTEGER PRIMARY KEY,
    answer_id INTEGER NOT NULL REFERENCES answers(id),
    at INTEGER NOT NULL
);

CREATE INDEX answer_uses_answer_at ON answer_uses(answer_id, at);

CREATE TABLE answer_candidates (
    id INTEGER PRIMARY KEY,
    ticket_id INTEGER NOT NULL REFERENCES tickets(id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    saved_at INTEGER
);

CREATE INDEX answer_candidates_unsaved
    ON answer_candidates(saved_at, created_at);

CREATE TABLE deferred_notices (
    id INTEGER PRIMARY KEY,
    ticket_id INTEGER NOT NULL REFERENCES tickets(id),
    session TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    text TEXT NOT NULL,
    not_before INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    sent_at INTEGER,
    sent_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX deferred_notices_due ON deferred_notices(sent_at, not_before);

ALTER TABLE attachments ADD COLUMN page_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE session_state ADD COLUMN last_summary_date TEXT;
ALTER TABLE session_state ADD COLUMN summary_message_id TEXT NOT NULL DEFAULT '';
ALTER TABLE session_state ADD COLUMN night_contact_ready INTEGER NOT NULL DEFAULT 0;
