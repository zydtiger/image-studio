"""Unit tests for the download engine using fake SDK responses.

Covers argument mapping (fixed-commit transfers, no local_dir), progress,
verification failure, retry reuse, and queued-only cancellation.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from image_studio.hub.downloads import DownloadEngine
from image_studio.schemas import DownloadStatus, ProfileId
from image_studio.storage import database
from image_studio.storage.repository import Repository
from image_studio.testing import FakeHub, FakeRepoSpec


@pytest.fixture
def setup(tmp_path: Path):
    hub = FakeHub(tmp_path / "hub")
    connection = database.connect(tmp_path / "app.sqlite")
    database.migrate(connection)
    repository = Repository(connection, hub_cache_dir=hub.cache_dir)
    engine = DownloadEngine(repository, hub.stack())
    engine.start()
    try:
        yield repository, engine, hub
    finally:
        engine.stop()


def _wait_status(repository: Repository, job_id: str, *statuses: DownloadStatus) -> object:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = repository.get_download(job_id)
        if job.status in statuses:
            return job
        time.sleep(0.01)
    return repository.get_download(job_id)


def test_download_resolves_commit_then_downloads_with_mapped_arguments(setup) -> None:
    repository, engine, hub = setup
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    engine.enqueue(job.id)
    finished = _wait_status(repository, job.id, DownloadStatus.COMPLETED, DownloadStatus.FAILED)
    assert finished.status is DownloadStatus.COMPLETED
    assert finished.resolved_commit == "a" * 40
    assert finished.progress.files_done == finished.progress.files_total
    assert finished.progress.bytes_done == finished.progress.bytes_total

    downloads = hub.api.calls["hf_hub_download"]
    assert downloads, "downloader was never invoked"
    for call in downloads:
        assert call["repo_id"] == "Tongyi-MAI/Z-Image"
        assert call["revision"] == "a" * 40  # fixed commit, never the branch name
        assert "local_dir" not in call
    listed = hub.api.calls["model_info"]
    assert listed and listed[0]["revision"] == "main"  # resolution happens first
    # only allowed patterns were fetched; no duplicate weight formats exist to fetch
    fetched = {call["filename"] for call in downloads}
    assert fetched == {
        name
        for name in hub.repos["Tongyi-MAI/Z-Image"].files
        if name.endswith((".json", ".txt", ".safetensors")) or name.startswith("tokenizer")
    }


def test_retry_reuses_cached_files(setup) -> None:
    repository, engine, hub = setup
    hub.api.logged_in_as = None  # not relevant; keep fakes inert
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision=None, profile=ProfileId.Z_IMAGE
    )
    repository.finish_download(job.id, DownloadStatus.FAILED, error_message="stale")
    assert repository.reset_download_for_retry(job.id)
    engine.enqueue(job.id)
    finished = _wait_status(repository, job.id, DownloadStatus.COMPLETED)
    assert finished.status is DownloadStatus.COMPLETED


def test_cancel_queued_only(setup) -> None:
    repository, engine, _hub = setup
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    # engine not given the id: the job stays queued
    cancelled = engine.cancel(job.id)
    assert cancelled.status is DownloadStatus.CANCELLED
    assert repository.list_registrations() == []
    running = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo", revision="main", profile=ProfileId.Z_IMAGE_TURBO
    )
    repository.claim_download(running.id)
    from image_studio.schemas import ImageStudioError

    with pytest.raises(ImageStudioError) as excinfo:
        engine.cancel(running.id)
    assert excinfo.value.code.value == "conflict"


def test_verification_failure_reports_missing_files(setup, tmp_path: Path) -> None:
    repository, engine, hub = setup
    # remove one required file from the catalog so the snapshot stays incomplete
    del hub.repos["Tongyi-MAI/Z-Image"].files["model_index.json"]
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    engine.enqueue(job.id)
    finished = _wait_status(repository, job.id, DownloadStatus.FAILED)
    assert finished.status is DownloadStatus.FAILED
    assert repository.list_registrations() == []
    assert finished.error.code.value == "cache_incomplete"
    assert "model_index.json" in finished.error.message


def test_bf16_layout_download_verifies(tmp_path: Path) -> None:
    """A bf16-only repository downloads its files and passes verification.

    The allow patterns already select bf16 shards and their
    ``*.index.bf16.json`` indexes, and post-download verification applies
    the shared variant-aware layout rules — no redownload, no rejection.
    """
    from tests.unit.test_snapshot_validation import _bf16, _complete_files

    hub = FakeHub(
        tmp_path / "hub",
        repos=[
            FakeRepoSpec(
                repo_id="test/z-image-bf16",
                files=_bf16(_complete_files()),
                sha="f" * 40,
            )
        ],
    )
    connection = database.connect(tmp_path / "app.sqlite")
    database.migrate(connection)
    repository = Repository(connection, hub_cache_dir=hub.cache_dir)
    engine = DownloadEngine(repository, hub.stack())
    engine.start()
    try:
        job = repository.create_download(
            repo_id="test/z-image-bf16", revision="main", profile=ProfileId.Z_IMAGE
        )
        engine.enqueue(job.id)
        finished = _wait_status(repository, job.id, DownloadStatus.COMPLETED, DownloadStatus.FAILED)
        assert finished.status is DownloadStatus.COMPLETED
        fetched = {call["filename"] for call in hub.api.calls["hf_hub_download"]}
        assert any("bf16" in name for name in fetched)
        assert "text_encoder/model.safetensors.index.bf16.json" in fetched
    finally:
        engine.stop()


def test_single_task_serialization(setup) -> None:
    repository, engine, hub = setup
    first = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    second = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo", revision="main", profile=ProfileId.Z_IMAGE_TURBO
    )
    engine.enqueue(first.id)
    engine.enqueue(second.id)
    _wait_status(repository, second.id, DownloadStatus.COMPLETED)
    assert repository.get_download(first.id).status is DownloadStatus.COMPLETED
    assert repository.get_download(second.id).status is DownloadStatus.COMPLETED


def test_typed_failure_does_not_starve_queue(setup, monkeypatch) -> None:
    """A gated transfer fails with its real code; the queue keeps serving."""
    import httpx
    from huggingface_hub.errors import GatedRepoError

    repository, engine, hub = setup

    def gated(repo_id, filename, revision):
        raise GatedRepoError(
            "gated",
            response=httpx.Response(
                403, request=httpx.Request("GET", f"https://huggingface.co/{repo_id}/{filename}")
            ),
        )

    hub_stack = engine._hub
    monkeypatch.setattr(hub_stack, "downloader", gated)
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    engine.enqueue(job.id)
    failed = _wait_status(repository, job.id, DownloadStatus.FAILED)
    assert failed.error.code.value == "gated_model"

    monkeypatch.undo()
    second = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    engine.enqueue(second.id)
    assert _wait_status(repository, second.id, DownloadStatus.COMPLETED).status is (
        DownloadStatus.COMPLETED
    )


def test_stop_prevents_new_jobs_and_survives_blocked_transfer(tmp_path) -> None:
    """Clean shutdown: no new claims, no repository use after stop, no crash."""
    import sqlite3
    import threading

    from image_studio.hub.client import HubStack
    from image_studio.hub.downloads import DownloadEngine
    from image_studio.storage import database

    hub = FakeHub(tmp_path / "hub")
    connection = database.connect(tmp_path / "app.sqlite")
    database.migrate(connection)
    repository = Repository(connection, hub_cache_dir=hub.cache_dir)
    gate = threading.Event()

    def blocked_downloader(repo_id, filename, revision):
        assert gate.wait(timeout=10), "test never released the transfer"
        raise sqlite3.ProgrammingError  # DB closed under us after stop

    stack = hub.stack()
    blocked_stack = HubStack(
        stack.api,
        file_lister=stack.file_lister,
        downloader=blocked_downloader,
        cache_dir=stack.cache_dir,
    )
    engine = DownloadEngine(repository, blocked_stack, stop_timeout=0.5)
    engine.start()
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    engine.enqueue(job.id)
    _wait_status(repository, job.id, DownloadStatus.RUNNING)

    unhandled: list[object] = []
    original_hook = threading.excepthook

    def capture(args):
        unhandled.append(args)

    threading.excepthook = capture
    try:
        engine.stop()  # bounded: returns while the transfer thread is blocked
        assert engine._thread is None
        # queued jobs must not start during/after stop
        queued = repository.create_download(
            repo_id="Tongyi-MAI/Z-Image-Turbo", revision="main", profile=ProfileId.Z_IMAGE_TURBO
        )
        engine.enqueue(queued.id)
        connection.close()  # caller owns the DB and closes it after stop
        gate.set()  # unblock the stuck transfer: it must exit quietly
        worker_thread = None
        for thread in threading.enumerate():
            if thread.name == "image-studio-downloads":
                worker_thread = thread
        if worker_thread is not None:
            worker_thread.join(timeout=5)
        time.sleep(0.2)
        assert not unhandled, f"worker thread died with an exception: {unhandled}"
    finally:
        threading.excepthook = original_hook
        gate.set()


def test_cancel_after_claim_conflicts_and_never_overwrites_running(setup, monkeypatch) -> None:
    """Deterministic interleaving: the worker claims the job between the
    cancel's status read and its write. The atomic conditional cancel must
    lose the race with a typed conflict, leaving the job running."""
    from image_studio.schemas import ImageStudioError

    repository, engine, _hub = setup
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    plain_get = repository.get_download
    interleaved = {"done": False}

    def claim_between_reads(job_id):
        # Hand the caller the pre-claim (queued) row and claim only after
        # the read: the first cancel read sees QUEUED, so the flow reaches
        # the conditional cancel and must lose the race with the claim.
        if not interleaved["done"]:
            interleaved["done"] = True
            old_row = plain_get(job_id)
            assert repository.claim_download(job_id)
            return old_row
        return plain_get(job_id)

    monkeypatch.setattr(repository, "get_download", claim_between_reads)
    with pytest.raises(ImageStudioError) as excinfo:
        engine.cancel(job.id)
    monkeypatch.undo()
    assert excinfo.value.code.value == "conflict"
    assert repository.get_download(job.id).status is DownloadStatus.RUNNING


def test_queued_cancel_is_atomic_and_normal_path_still_cancels(setup) -> None:
    repository, engine, _hub = setup
    job = repository.create_download(
        repo_id="Tongyi-MAI/Z-Image", revision="main", profile=ProfileId.Z_IMAGE
    )
    cancelled = engine.cancel(job.id)
    assert cancelled.status is DownloadStatus.CANCELLED
    assert repository.list_registrations() == []
    assert repository.cancel_queued_download(job.id) is False  # not queued anymore


def test_repeated_downloads_reuse_registration_and_preserve_name(setup):
    repository, engine, hub = setup
    for attempt in range(2):
        job = repository.create_download(
            repo_id="Tongyi-MAI/Z-Image-Turbo", revision="main", profile=ProfileId.Z_IMAGE_TURBO
        )
        engine.enqueue(job.id)
        assert (
            _wait_status(repository, job.id, DownloadStatus.COMPLETED, DownloadStatus.FAILED).status
            is DownloadStatus.COMPLETED
        )
        registrations = repository.list_registrations()
        assert len(registrations) == 1
        registration = registrations[0]
        assert registration.commit_sha == "b" * 40
        assert registration.profile is ProfileId.Z_IMAGE_TURBO
        if attempt == 0:
            original_id = registration.id
            repository.update_registration(registration.id, display_name="My Turbo", profile=None)
        else:
            assert registration.id == original_id
            assert registration.display_name == "My Turbo"
