-- Initial schema: registrations, downloads, runs, and per-image records.

CREATE TABLE registrations (
    id            TEXT PRIMARY KEY,
    repo_id       TEXT NOT NULL,
    commit_sha    TEXT NOT NULL,
    profile       TEXT NOT NULL,
    display_name  TEXT,
    status        TEXT NOT NULL DEFAULT 'ready',
    missing_files TEXT NOT NULL DEFAULT '[]',
    snapshot_path TEXT,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT
);

CREATE UNIQUE INDEX registrations_identity
    ON registrations (repo_id, profile, commit_sha);

CREATE TABLE downloads (
    id                 TEXT PRIMARY KEY,
    repo_id            TEXT NOT NULL,
    requested_revision TEXT,
    resolved_commit    TEXT,
    profile            TEXT NOT NULL,
    status             TEXT NOT NULL,
    error_code         TEXT,
    error_message      TEXT,
    bytes_done         INTEGER NOT NULL DEFAULT 0,
    bytes_total        INTEGER,
    files_done         INTEGER NOT NULL DEFAULT 0,
    files_total        INTEGER,
    created_at         TEXT NOT NULL,
    started_at         TEXT,
    finished_at        TEXT
);

CREATE INDEX downloads_status ON downloads (status, created_at);

CREATE TABLE runs (
    run_id              TEXT PRIMARY KEY,
    queue_seq           INTEGER NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    finished_at         TEXT,
    status              TEXT NOT NULL,
    favorite            INTEGER NOT NULL DEFAULT 0,
    trashed_at          TEXT,
    registration_id     TEXT NOT NULL,
    repo_id             TEXT NOT NULL,
    commit_sha          TEXT NOT NULL,
    profile             TEXT NOT NULL,
    dtype               TEXT NOT NULL,
    snapshot_path       TEXT NOT NULL,
    gpu_uuid            TEXT NOT NULL,
    gpu_name            TEXT NOT NULL,
    prompt              TEXT NOT NULL,
    negative_prompt     TEXT,
    width               INTEGER NOT NULL,
    height              INTEGER NOT NULL,
    steps               INTEGER NOT NULL,
    guidance            REAL NOT NULL,
    initial_seed        INTEGER NOT NULL,
    image_count         INTEGER NOT NULL,
    pipeline_class      TEXT,
    dependency_versions TEXT NOT NULL DEFAULT '{}',
    runtime_meta        TEXT NOT NULL DEFAULT '{}',
    error_code          TEXT,
    error_message       TEXT
);

CREATE INDEX runs_queue ON runs (queue_seq);
CREATE INDEX runs_status ON runs (status);
CREATE INDEX runs_registration ON runs (registration_id);

CREATE TABLE images (
    run_id       TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    artifact_id  TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    seed         INTEGER NOT NULL,
    status       TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    size_bytes   INTEGER,
    error        TEXT,
    completed_at TEXT,
    PRIMARY KEY (run_id, artifact_id)
);
