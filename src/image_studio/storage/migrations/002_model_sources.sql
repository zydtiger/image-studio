-- Preserve complete component provenance without changing legacy identities.
ALTER TABLE registrations ADD COLUMN sources TEXT NOT NULL DEFAULT '[]';
ALTER TABLE downloads ADD COLUMN sources TEXT NOT NULL DEFAULT '[]';
ALTER TABLE runs ADD COLUMN sources TEXT NOT NULL DEFAULT '[]';
