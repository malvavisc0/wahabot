ALTER TABLE intake ADD COLUMN notified_at REAL;

CREATE INDEX intake_operator_notice
    ON intake(state, outcome, notified_at);
