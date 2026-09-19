-- Cache locations belong to the current environment, not model provenance.
ALTER TABLE registrations DROP COLUMN snapshot_path;
ALTER TABLE runs DROP COLUMN snapshot_path;

UPDATE registrations SET sources = (
    SELECT json_group_array(json_remove(value, '$.snapshot_path'))
    FROM json_each(registrations.sources)
);
UPDATE downloads SET sources = (
    SELECT json_group_array(json_remove(value, '$.snapshot_path'))
    FROM json_each(downloads.sources)
);
UPDATE runs SET sources = (
    SELECT json_group_array(json_remove(value, '$.snapshot_path'))
    FROM json_each(runs.sources)
);
