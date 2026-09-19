"""Single-task download queue with fixed-commit, file-filtered transfers.

A job resolves its revision to a commit hash before any transfer, downloads
only the files selected for the profile into the SDK-resolved cache (never
``local_dir``), reports measured per-file progress, and verifies the
snapshot afterwards. Retries reuse already cached files automatically.

Failure semantics: SDK failures are translated to typed contract errors
(gated/auth/revision/not-found/unreachable); a failing job never kills the
worker thread and never starves the queue. During ``stop`` the engine stops
claiming queued jobs and stops touching the repository, so a caller closing
the database afterwards cannot crash a live transfer thread; a job left
``running`` is reconciled as interrupted on the next startup.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
from pathlib import Path

from image_studio.hub.cache import problem_codes, snapshot_for_commit, snapshot_problems
from image_studio.hub.client import HubStack, translate_hub_error
from image_studio.hub.compatibility import filter_files
from image_studio.schemas import (
    DownloadJob,
    DownloadStatus,
    ErrorCode,
    ImageStudioError,
)
from image_studio.storage.repository import Repository

logger = logging.getLogger(__name__)

_STOP = object()


class _ShuttingDown(Exception):
    """Internal: the engine is stopping; drop repository work immediately."""


class DownloadEngine:
    def __init__(
        self, repository: Repository, hub: HubStack, *, stop_timeout: float = 10.0
    ) -> None:
        self._repository = repository
        self._hub = hub
        self._cache_dir = hub.cache_dir
        self._stop_timeout = stop_timeout
        self._queue: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._worker, name="image-studio-downloads", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop claiming work and join the worker; bounded by ``stop_timeout``.

        An in-flight transfer may outlive the join (for example a hung
        network call); it abandons all repository work when it observes the
        stopping flag, so a caller may close the database right after this
        returns without crashing that thread.
        """
        self._stopping.set()
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=self._stop_timeout)
            self._thread = None

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def retry(self, job_id: str) -> None:
        if not self._repository.reset_download_for_retry(job_id):
            job = self._repository.get_download(job_id)
            raise ImageStudioError(
                ErrorCode.CONFLICT,
                f"download job is {job.status.value} and cannot be retried",
            )
        self.enqueue(job_id)

    def cancel(self, job_id: str) -> DownloadJob:
        # The read only decides which branch to attempt; the atomic
        # conditional UPDATE decides the outcome, so a job the worker
        # claimed between read and write can never be cancelled over.
        job = self._repository.get_download(job_id)
        if job.status == DownloadStatus.QUEUED:
            if not self._repository.cancel_queued_download(job_id):
                current = self._repository.get_download(job_id)
                raise ImageStudioError(
                    ErrorCode.CONFLICT,
                    f"download job is already {current.status.value}",
                )
            return self._repository.get_download(job_id)
        if job.status == DownloadStatus.RUNNING:
            raise ImageStudioError(ErrorCode.CONFLICT, "an active download cannot be cancelled")
        raise ImageStudioError(ErrorCode.CONFLICT, f"download job is already {job.status.value}")

    # ----- worker ---------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP or self._stopping.is_set():
                    return
                self._run_job(item)
            except _ShuttingDown:
                return
            except Exception:  # noqa: BLE001 - one job never kills the queue
                logger.exception("download worker iteration failed")
            finally:
                self._queue.task_done()

    def _persist(self, operation, *args, **kwargs):
        """Run one repository mutation; abandon all work while shutting down."""
        if self._stopping.is_set():
            raise _ShuttingDown
        try:
            return operation(*args, **kwargs)
        except sqlite3.ProgrammingError:
            # The database was closed under us (shutdown race): stop quietly.
            self._stopping.set()
            raise _ShuttingDown from None

    def _run_job(self, job_id: str) -> None:
        if self._stopping.is_set():
            return
        if not self._persist(self._repository.claim_download, job_id):
            return  # cancelled while queued
        job = self._persist(self._repository.get_download, job_id)
        try:
            if job.sources:
                self._transfer_sources(job)
            else:
                self._transfer(job_id, job.repo_id, job.requested_revision, job.resolved_commit)
        except _ShuttingDown:
            raise
        except ImageStudioError as exc:
            self._persist(
                self._repository.finish_download,
                job_id,
                DownloadStatus.FAILED,
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception as exc:
            translated = translate_hub_error(exc)
            self._persist(
                self._repository.finish_download,
                job_id,
                DownloadStatus.FAILED,
                error_code=translated.code,
                error_message=translated.message,
            )

    def _transfer_sources(self, job) -> None:
        from image_studio.hub.anima import model_problems

        selected = []
        for source in job.sources:
            commit, listing = self._hub.file_lister(source.repo_id, source.commit_sha)
            if commit != source.commit_sha:
                raise ImageStudioError(ErrorCode.REVISION_NOT_FOUND, "download revision mismatch")
            sizes = dict(listing)
            for name in source.files:
                if name not in sizes:
                    raise ImageStudioError(
                        ErrorCode.CACHE_INCOMPLETE, f"{source.repo_id}/{name} is missing"
                    )
                selected.append((source, name, sizes[name]))
        self._persist(
            self._repository.update_download_progress,
            job.id,
            bytes_done=0,
            files_done=0,
            files_total=len(selected),
            bytes_total=sum(size or 0 for _, _, size in selected),
        )
        done = 0
        for count, (source, name, size) in enumerate(selected, 1):
            if self._stopping.is_set():
                raise _ShuttingDown
            path = self._hub.downloader(source.repo_id, name, source.commit_sha)
            done += size if size is not None else _file_size(path)
            self._persist(
                self._repository.update_download_progress, job.id, bytes_done=done, files_done=count
            )
        problems = model_problems(
            Path(job.sources[0].snapshot_path), job.repo_id, job.profile, job.sources
        )
        if problems:
            raise ImageStudioError(
                problem_codes(problems), "; ".join(problem.detail for problem in problems)
            )
        self._persist(self._repository.complete_download, job.id)

    def _transfer(
        self,
        job_id: str,
        repo_id: str,
        requested_revision: str | None,
        resolved_commit: str | None,
    ) -> None:
        revision = resolved_commit or requested_revision
        commit, files = self._hub.file_lister(repo_id, revision)
        if self._stopping.is_set():
            raise _ShuttingDown
        if resolved_commit is None and commit:
            self._persist(self._repository.update_download_progress, job_id, resolved_commit=commit)
            revision = commit
        allowed = set(filter_files([name for name, _ in files]))
        selected = [(name, size) for name, size in files if name in allowed]
        if not selected:
            raise RuntimeError("no required files found for the selected profile")
        bytes_total = sum(size or 0 for _, size in selected)
        self._persist(
            self._repository.update_download_progress,
            job_id,
            bytes_done=0,
            bytes_total=bytes_total,
            files_done=0,
            files_total=len(selected),
        )

        bytes_done = 0
        for index, (name, size) in enumerate(selected, start=1):
            path = self._hub.downloader(repo_id, name, revision)
            bytes_done += size if size is not None else _file_size(path)
            self._persist(
                self._repository.update_download_progress,
                job_id,
                bytes_done=bytes_done,
                files_done=index,
            )

        if commit is None:
            raise RuntimeError("revision could not be resolved to a commit hash")
        snapshot = snapshot_for_commit(repo_id, commit, self._cache_dir)
        if snapshot is None:
            raise RuntimeError("downloaded snapshot not found in cache")
        problems = snapshot_problems(snapshot.path)
        if problems:
            self._persist(
                self._repository.finish_download,
                job_id,
                DownloadStatus.FAILED,
                error_code=problem_codes(problems),
                error_message="snapshot does not satisfy the Z-Image component manifest: "
                + "; ".join(problem.detail for problem in problems),
            )
            return
        self._persist(self._repository.complete_download, job_id)


def _file_size(path: str | None) -> int:
    return os.path.getsize(path) if path else 0
