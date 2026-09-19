"""Unit tests for the repository: migrations, run records, filters, reconciliation."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from image_studio import schemas as s
from image_studio.storage import database
from image_studio.storage.repository import NotFoundError, Repository, RunFilters


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    connection = database.connect(tmp_path / "app.sqlite")
    database.migrate(connection)
    return Repository(connection)


def _registration(repo: Repository) -> s.ModelRegistration:
    return repo.create_registration(
        repo_id="Tongyi-MAI/Z-Image",
        commit_sha="c" * 40,
        profile=s.ProfileId.Z_IMAGE,
        snapshot_path="/hf/snapshots/" + "c" * 40,
        display_name="Z-Image",
    )


def _spec(
    registration_id: str, *, run_id: str = "ab" * 16, created_offset_min=0
) -> s.FrozenRunSpec:
    return s.FrozenRunSpec(
        run_id=run_id,
        created_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC) + timedelta(minutes=created_offset_min),
        model=s.FrozenModel(
            registration_id=registration_id,
            repo_id="Tongyi-MAI/Z-Image",
            commit_sha="c" * 40,
            profile=s.ProfileId.Z_IMAGE,
            dtype="bfloat16",
            snapshot_path="/hf/snapshots/" + "c" * 40,
        ),
        gpu=s.FrozenGpu(uuid="GPU-fake-0001", name="Fake GPU A"),
        prompt="a prompt",
        negative_prompt=None,
        width=256,
        height=256,
        steps=50,
        guidance=4.0,
        image_count=2,
        seeds=(11, 12),
        artifact_ids=("image-001", "image-002"),
    )


def test_migrations_set_user_version(tmp_path: Path) -> None:
    connection = database.connect(tmp_path / "app.sqlite")
    assert database.migrate(connection) == 1
    assert database.schema_version(connection) == 1
    database.migrate(connection)  # idempotent
    assert database.schema_version(connection) == 1


def test_create_run_persists_queued_run_and_pending_images(repo: Repository) -> None:
    registration = _registration(repo)
    repo.create_run(_spec(registration.id))
    row = repo.get_run("ab" * 16)
    assert row["status"] == "queued"
    assert row["initial_seed"] == 11
    assert row["gpu_uuid"] == "GPU-fake-0001"
    images = repo.get_images("ab" * 16)
    assert [image["artifact_id"] for image in images] == ["image-001", "image-002"]
    assert all(image["status"] == "pending" for image in images)


def test_finish_semantics_partial_vs_failed(repo: Repository) -> None:
    registration = _registration(repo)
    repo.create_run(_spec(registration.id))
    repo.mark_run_running("ab" * 16)
    repo.complete_image("ab" * 16, "image-001", width=256, height=256, size_bytes=10)
    repo.finish_run(
        "ab" * 16,
        s.RunStatus.FAILED,
        error_code=s.ErrorCode.WORKER_ERROR,
        error_message="boom",
        cancel_pending=True,
    )
    row = repo.get_run("ab" * 16)
    assert row["status"] == "partial"
    images = {image["artifact_id"]: image["status"] for image in repo.get_images("ab" * 16)}
    assert images == {"image-001": "completed", "image-002": "cancelled"}

    repo.create_run(_spec(registration.id, run_id="cd" * 16))
    repo.finish_run("cd" * 16, s.RunStatus.FAILED, cancel_pending=True)
    assert repo.get_run("cd" * 16)["status"] == "failed"


def test_reconcile_marks_interrupted_and_paused(repo: Repository) -> None:
    registration = _registration(repo)
    repo.create_run(_spec(registration.id, run_id="aa" * 16))
    repo.create_run(_spec(registration.id, run_id="bb" * 16, created_offset_min=1))
    repo.mark_run_running("aa" * 16)
    interrupted, paused = repo.reconcile_runs()
    assert (interrupted, paused) == (1, 1)
    assert repo.get_run("aa" * 16)["status"] == "interrupted"
    assert repo.get_run("bb" * 16)["status"] == "paused"


def test_paused_specs_rebuild_frozen_handoff(repo: Repository) -> None:
    registration = _registration(repo)
    repo.create_run(_spec(registration.id, run_id="aa" * 16))
    repo.reconcile_runs()
    specs = repo.paused_specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.run_id == "aa" * 16
    assert spec.model.snapshot_path == "/hf/snapshots/" + "c" * 40
    assert spec.gpu.uuid == "GPU-fake-0001"
    assert spec.seeds == (11, 12)
    assert spec.artifact_ids == ("image-001", "image-002")


def test_filters_search_favorite_trashed(repo: Repository) -> None:
    registration = _registration(repo)
    for run_id, prompt in (("aa" * 16, "cat picture"), ("bb" * 16, "dog picture")):
        repo.create_run(_spec(registration.id, run_id=run_id))
        with repo._write() as conn:  # direct update: prompts come from the spec otherwise
            conn.execute("UPDATE runs SET prompt = ? WHERE run_id = ?", (prompt, run_id))
    repo.finish_run("aa" * 16, s.RunStatus.COMPLETED, cancel_pending=True)
    repo.set_run_favorite("aa" * 16, True)
    repo.set_run_trashed("bb" * 16, True)

    runs, total = repo.list_runs(RunFilters(q="cat"))
    assert total == 1 and runs[0].run_id == "aa" * 16
    runs, total = repo.list_runs(RunFilters(favorite=True))
    assert total == 1 and runs[0].favorite
    runs, total = repo.list_runs(RunFilters(trashed="only"))
    assert total == 1 and runs[0].run_id == "bb" * 16 and runs[0].trashed
    runs, total = repo.list_runs(RunFilters())
    assert total == 1 and runs[0].run_id == "aa" * 16


def test_filters_has_images_and_exclude_empty_cancelled(repo: Repository) -> None:
    registration = _registration(repo)
    # completed run with two images
    repo.create_run(_spec(registration.id, run_id="aa" * 16))
    repo.mark_run_running("aa" * 16)
    for artifact in ("image-001", "image-002"):
        repo.complete_image("aa" * 16, artifact, width=256, height=256, size_bytes=10)
    repo.finish_run("aa" * 16, s.RunStatus.COMPLETED)
    # cancelled run without images
    repo.create_run(_spec(registration.id, run_id="bb" * 16, created_offset_min=1))
    repo.finish_run("bb" * 16, s.RunStatus.CANCELLED, cancel_pending=True)
    # failed run without images
    repo.create_run(_spec(registration.id, run_id="cc" * 16, created_offset_min=2))
    repo.finish_run("cc" * 16, s.RunStatus.FAILED, cancel_pending=True)
    # active run with one saved image
    repo.create_run(_spec(registration.id, run_id="dd" * 16, created_offset_min=3))
    repo.mark_run_running("dd" * 16)
    repo.complete_image("dd" * 16, "image-001", width=256, height=256, size_bytes=10)

    # The unfiltered default still returns everything.
    runs, total = repo.list_runs(RunFilters())
    assert total == 4
    assert {run.run_id for run in runs} == {"aa" * 16, "bb" * 16, "cc" * 16, "dd" * 16}

    runs, total = repo.list_runs(RunFilters(exclude_empty_cancelled=True))
    assert total == 3
    assert {run.run_id for run in runs} == {"aa" * 16, "cc" * 16, "dd" * 16}

    runs, total = repo.list_runs(RunFilters(has_images=True))
    assert total == 2
    assert {run.run_id for run in runs} == {"aa" * 16, "dd" * 16}

    runs, total = repo.list_runs(RunFilters(has_images=False))
    assert total == 2
    assert {run.run_id for run in runs} == {"bb" * 16, "cc" * 16}

    # An explicit cancelled status filter overrides the default exclusion.
    runs, total = repo.list_runs(
        RunFilters(status=s.RunStatus.CANCELLED, exclude_empty_cancelled=True)
    )
    assert total == 1 and runs[0].run_id == "bb" * 16

    # Filters apply before LIMIT/OFFSET: paging moves over the filtered
    # set (newest first: cc then bb), not the raw table.
    runs, total = repo.list_runs(RunFilters(has_images=False, limit=1, offset=1))
    assert total == 2 and [run.run_id for run in runs] == ["bb" * 16]


def test_registration_lifecycle_and_guards(repo: Repository) -> None:
    registration = _registration(repo)
    repo.create_run(_spec(registration.id))
    assert repo.registration_in_active_use(registration.id)
    repo.finish_run("ab" * 16, s.RunStatus.COMPLETED, cancel_pending=True)
    assert not repo.registration_in_active_use(registration.id)
    duplicate = pytest.raises(ValueError)
    with duplicate:
        repo.create_registration(
            repo_id="Tongyi-MAI/Z-Image",
            commit_sha="c" * 40,
            profile=s.ProfileId.Z_IMAGE,
            snapshot_path="/x",
            display_name=None,
        )
    repo.delete_registration(registration.id)
    with pytest.raises(NotFoundError):
        repo.get_registration(registration.id)
    # history survives registration deletion
    assert repo.get_run("ab" * 16)["repo_id"] == "Tongyi-MAI/Z-Image"


def test_download_state_transitions(repo: Repository) -> None:
    job = repo.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=s.ProfileId.Z_IMAGE
    )
    assert job.status is s.DownloadStatus.QUEUED
    assert repo.claim_download(job.id)
    assert not repo.claim_download(job.id)  # already running
    repo.update_download_progress(job.id, resolved_commit="c" * 40, bytes_total=100, files_total=2)
    repo.update_download_progress(job.id, bytes_done=50, files_done=1)
    job = repo.get_download(job.id)
    assert job.progress.bytes_done == 50 and job.progress.files_total == 2
    repo.finish_download(job.id, s.DownloadStatus.COMPLETED)
    assert not repo.reset_download_for_retry(job.id)
    job2 = repo.create_download(repo_id="x/y", revision=None, profile=s.ProfileId.Z_IMAGE)
    repo.finish_download(
        job2.id,
        s.DownloadStatus.FAILED,
        error_code=s.ErrorCode.HUB_UNREACHABLE,
        error_message="network",
    )
    assert repo.reset_download_for_retry(job2.id)
    assert repo.get_download(job2.id).status is s.DownloadStatus.QUEUED


def test_migration_failure_rolls_back_completely_and_retries(tmp_path, monkeypatch):
    """A failing migration leaves no partial schema and can be retried."""
    from image_studio.storage import database

    connection = database.connect(tmp_path / "app.sqlite")
    original = database._migration_scripts
    failing = "CREATE TABLE partial_table(id INTEGER);\nINSERT INTO missing_table VALUES (1);\n"
    monkeypatch.setattr(database, "_migration_scripts", lambda: [(1, failing)])
    with pytest.raises(sqlite3.OperationalError):
        database.migrate(connection)
    leftovers = connection.execute(
        "SELECT name FROM sqlite_master WHERE name='partial_table'"
    ).fetchall()
    assert leftovers == [], "failed migration must not leave partial tables"
    assert database.schema_version(connection) == 0

    # restart/retry with the real scripts applies cleanly on the same file
    monkeypatch.setattr(database, "_migration_scripts", original)
    assert database.migrate(connection) == 1
    assert database.schema_version(connection) == 1
    assert (
        connection.execute("SELECT name FROM sqlite_master WHERE name='runs'").fetchone()
        is not None
    )
    connection.close()


def test_update_registration_conflict_is_atomic(tmp_path):
    from image_studio.storage import database

    connection = database.connect(tmp_path / "app.sqlite")
    database.migrate(connection)
    repo = Repository(connection)
    base = repo.create_registration(
        repo_id="Tongyi-MAI/Z-Image",
        commit_sha="c" * 40,
        profile=s.ProfileId.Z_IMAGE,
        snapshot_path="/x",
        display_name=None,
    )
    repo.create_registration(
        repo_id="Tongyi-MAI/Z-Image",
        commit_sha="c" * 40,
        profile=s.ProfileId.Z_IMAGE_TURBO,
        snapshot_path="/x",
        display_name=None,
    )
    with pytest.raises(ValueError, match="already exists"):
        repo.update_registration(
            base.id, display_name="New Name", profile=s.ProfileId.Z_IMAGE_TURBO
        )
    # no partial update survived the rollback
    unchanged = repo.get_registration(base.id)
    assert unchanged.profile is s.ProfileId.Z_IMAGE
    assert unchanged.display_name is None
    connection.close()
