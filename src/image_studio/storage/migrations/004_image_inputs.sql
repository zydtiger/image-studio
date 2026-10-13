ALTER TABLE runs ADD COLUMN mode TEXT NOT NULL DEFAULT 'text-to-image';

CREATE TABLE inputs (
    input_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL
);

CREATE TABLE run_inputs (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    input_id TEXT NOT NULL REFERENCES inputs(input_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (run_id, position)
);
CREATE INDEX run_inputs_input ON run_inputs(input_id);
