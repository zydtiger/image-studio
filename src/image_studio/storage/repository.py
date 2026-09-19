"""Typed SQLite persistence for registrations, downloads, runs, and images.

All timestamps are stored as timezone-aware UTC ISO 8601 strings. A single
reentrant lock serializes access across the API threads and the runtime
event-dispatch thread sharing one connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from image_studio import schemas
from image_studio.hub.paths import resolve_sources, snapshot_path
from image_studio.schemas import (
    DownloadJob,
    DownloadStatus,
    ErrorCode,
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    ImageStatus,
    ProfileId,
    RegistrationStatus,
    RunStatus,
)

ACTIVE_RUN_STATUSES = (RunStatus.QUEUED.value, RunStatus.PAUSED.value, RunStatus.RUNNING.value)
TERMINAL_RUN_STATUSES = (
    RunStatus.COMPLETED.value,
    RunStatus.PARTIAL.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.INTERRUPTED.value,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class NotFoundError(Exception):
    """Internal marker for missing rows; the API layer maps it to 404."""


@dataclass(frozen=True)
class RunFilters:
    q: str | None = None
    status: RunStatus | None = None
    repo_id: str | None = None
    favorite: bool | None = None
    trashed: str = "exclude"  # exclude | only
    # None applies no image filter; True requires, False forbids, a
    # completed image.
    has_images: bool | None = None
    # Hides cancelled runs without completed images (the History default
    # view); an explicit cancelled status filter overrides it.
    exclude_empty_cancelled: bool = False
    limit: int = 50
    offset: int = 0


class Repository:
    def __init__(self, connection: sqlite3.Connection, *, hub_cache_dir: Path) -> None:
        self._connection = connection
        self._lock = threading.RLock()
        self.hub_cache_dir = hub_cache_dir

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    # ----- registrations ---------------------------------------------------

    def create_registration(
        self,
        *,
        repo_id: str,
        commit_sha: str,
        profile: ProfileId,
        display_name: str | None,
        sources: tuple[schemas.ModelSource, ...] = (),
    ) -> schemas.ModelRegistration:
        created = utc_now_iso()
        registration_id = _new_id()
        try:
            with self._write() as conn:
                conn.execute(
                    "INSERT INTO registrations (id, repo_id, commit_sha, profile,"
                    " display_name, status, missing_files, created_at, sources)"
                    " VALUES (?, ?, ?, ?, ?, 'ready', '[]', ?, ?)",
                    (
                        registration_id,
                        repo_id,
                        commit_sha,
                        profile.value,
                        display_name,
                        created,
                        _dump_sources(sources),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"a registration for {repo_id} at commit {commit_sha[:12]}"
                f" with profile {profile.value} already exists"
            ) from exc
        return self.get_registration(registration_id)

    def get_registration(self, registration_id: str) -> schemas.ModelRegistration:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM registrations WHERE id = ?", (registration_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown registration {registration_id}")
        return _row_to_registration(row, self.hub_cache_dir)

    def list_registrations(self) -> list[schemas.ModelRegistration]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM registrations ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_registration(row, self.hub_cache_dir) for row in rows]

    def update_registration(
        self,
        registration_id: str,
        *,
        display_name: str | None = None,
        profile: ProfileId | None = None,
    ) -> schemas.ModelRegistration:
        if display_name is None and profile is None:
            return self.get_registration(registration_id)
        try:
            with self._write() as conn:
                if display_name is not None:
                    conn.execute(
                        "UPDATE registrations SET display_name = ? WHERE id = ?",
                        (display_name, registration_id),
                    )
                if profile is not None:
                    conn.execute(
                        "UPDATE registrations SET profile = ? WHERE id = ?",
                        (profile.value, registration_id),
                    )
        except sqlite3.IntegrityError as exc:
            # The write context rolled the transaction back: no partial
            # display_name/profile update survives a conflicting change.
            raise ValueError(
                "a registration for this repository and profile already exists"
            ) from exc
        return self.get_registration(registration_id)

    def delete_registration(self, registration_id: str) -> None:
        with self._write() as conn:
            cursor = conn.execute("DELETE FROM registrations WHERE id = ?", (registration_id,))
            if cursor.rowcount == 0:
                raise NotFoundError(f"unknown registration {registration_id}")

    def mark_registration_used(self, registration_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE registrations SET last_used_at = ? WHERE id = ?",
                (utc_now_iso(), registration_id),
            )

    def set_registration_status(
        self, registration_id: str, status: RegistrationStatus, missing_files: list[str]
    ) -> None:
        import json as _json

        with self._write() as conn:
            conn.execute(
                "UPDATE registrations SET status = ?, missing_files = ? WHERE id = ?",
                (status.value, _json.dumps(missing_files), registration_id),
            )

    def registration_in_active_use(self, registration_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM runs WHERE registration_id = ? AND status IN (?, ?, ?) LIMIT 1",
                (registration_id, *ACTIVE_RUN_STATUSES),
            ).fetchone()
        return row is not None

    # ----- downloads ---------------------------------------------------------

    def create_download(
        self,
        *,
        repo_id: str,
        revision: str | None,
        profile: ProfileId,
        sources: tuple[schemas.ModelSource, ...] = (),
    ) -> DownloadJob:
        job_id = _new_id()
        created = utc_now_iso()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO downloads (id, repo_id, requested_revision, profile,"
                " status, created_at, sources, resolved_commit)"
                " VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)",
                (
                    job_id,
                    repo_id,
                    revision,
                    profile.value,
                    created,
                    _dump_sources(sources),
                    sources[0].commit_sha if sources else None,
                ),
            )
        return self.get_download(job_id)

    def get_download(self, job_id: str) -> DownloadJob:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM downloads WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown download job {job_id}")
        return _row_to_download(row, self.hub_cache_dir)

    def list_downloads(self) -> list[DownloadJob]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM downloads ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_download(row, self.hub_cache_dir) for row in rows]

    def claim_download(self, job_id: str) -> bool:
        """Atomically move a queued job to running; False if it left the queue."""
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE downloads SET status = 'running', started_at = ?"
                " WHERE id = ? AND status = 'queued'",
                (utc_now_iso(), job_id),
            )
            return cursor.rowcount == 1

    def update_download_progress(
        self,
        job_id: str,
        *,
        resolved_commit: str | None = None,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
        files_done: int | None = None,
        files_total: int | None = None,
    ) -> None:
        assignments = []
        values: list[object] = []
        if resolved_commit is not None:
            assignments.append("resolved_commit = ?")
            values.append(resolved_commit)
        for column, value in (
            ("bytes_done", bytes_done),
            ("bytes_total", bytes_total),
            ("files_done", files_done),
            ("files_total", files_total),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if not assignments:
            return
        values.append(job_id)
        with self._write() as conn:
            conn.execute(f"UPDATE downloads SET {', '.join(assignments)} WHERE id = ?", values)

    def complete_download(self, job_id: str) -> DownloadJob:
        """Publish a verified download and its registration in one transaction."""
        with self._write() as conn:
            job = conn.execute("SELECT * FROM downloads WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise NotFoundError(f"unknown download job {job_id}")
            if job["status"] != "running" or not job["resolved_commit"]:
                raise ValueError("only a running download with a fixed commit can complete")
            now = utc_now_iso()
            conn.execute(
                "INSERT INTO registrations (id, repo_id, commit_sha, profile,"
                " status, missing_files, created_at, sources)"
                " VALUES (?, ?, ?, ?, 'ready', '[]', ?, ?)"
                " ON CONFLICT (repo_id, profile, commit_sha) DO UPDATE SET"
                " status = 'ready', missing_files = '[]',"
                " sources = excluded.sources",
                (
                    _new_id(),
                    job["repo_id"],
                    job["resolved_commit"],
                    job["profile"],
                    now,
                    job["sources"],
                ),
            )
            conn.execute(
                "UPDATE downloads SET status = 'completed', error_code = NULL,"
                " error_message = NULL, finished_at = ? WHERE id = ?",
                (now, job_id),
            )
        return self.get_download(job_id)

    def finish_download(
        self,
        job_id: str,
        status: DownloadStatus,
        *,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> DownloadJob:
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE downloads SET status = ?, error_code = ?, error_message = ?,"
                " finished_at = ? WHERE id = ?",
                (
                    status.value,
                    error_code.value if error_code else None,
                    error_message,
                    utc_now_iso(),
                    job_id,
                ),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"unknown download job {job_id}")
        return self.get_download(job_id)

    def reset_download_for_retry(self, job_id: str) -> bool:
        """Move a failed or cancelled job back to queued, keeping resolved commit."""
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE downloads SET status = 'queued', error_code = NULL,"
                " error_message = NULL, finished_at = NULL, started_at = NULL,"
                " bytes_done = 0, files_done = 0 WHERE id = ?"
                " AND status IN ('failed', 'cancelled')",
                (job_id,),
            )
            return cursor.rowcount == 1

    def cancel_queued_download(self, job_id: str) -> bool:
        """Atomically cancel a queued job; False if it left the queue first.

        The conditional UPDATE races safely with ``claim_download``: exactly
        one of the two transitions (queued -> running, queued -> cancelled)
        wins, so a cancellation can never overwrite a job the worker already
        claimed.
        """
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE downloads SET status = 'cancelled',"
                " error_message = 'cancelled while queued', finished_at = ?"
                " WHERE id = ? AND status = 'queued'",
                (utc_now_iso(), job_id),
            )
            return cursor.rowcount == 1

    def reconcile_downloads(self) -> int:
        """Fail jobs stranded in queued/running state by a previous shutdown."""
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE downloads SET status = 'failed', error_code = ?,"
                " error_message = ?, finished_at = ? WHERE status IN ('queued', 'running')",
                (
                    ErrorCode.INTERNAL.value,
                    "interrupted by server shutdown",
                    utc_now_iso(),
                ),
            )
            return cursor.rowcount

    # ----- runs and images ----------------------------------------------------

    def create_run(self, spec: FrozenRunSpec) -> None:
        created = spec.created_at.isoformat()
        with self._write() as conn:
            queue_seq = conn.execute("SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM runs").fetchone()[
                0
            ]
            conn.execute(
                "INSERT INTO runs (run_id, queue_seq, created_at, status,"
                " registration_id, repo_id, commit_sha, profile, dtype,"
                " gpu_uuid, gpu_name, prompt, negative_prompt, width, height, steps,"
                " guidance, initial_seed, image_count, sources)"
                " VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.run_id,
                    queue_seq,
                    created,
                    spec.model.registration_id,
                    spec.model.repo_id,
                    spec.model.commit_sha,
                    spec.model.profile.value,
                    spec.model.dtype,
                    spec.gpu.uuid,
                    spec.gpu.name,
                    spec.prompt,
                    spec.negative_prompt,
                    spec.width,
                    spec.height,
                    spec.steps,
                    spec.guidance,
                    spec.seeds[0],
                    spec.image_count,
                    _dump_sources(spec.model.sources),
                ),
            )
            conn.executemany(
                "INSERT INTO images (run_id, artifact_id, idx, seed, status)"
                " VALUES (?, ?, ?, ?, 'pending')",
                [
                    (spec.run_id, artifact_id, index, seed)
                    for index, (artifact_id, seed) in enumerate(
                        zip(spec.artifact_ids, spec.seeds, strict=True), start=1
                    )
                ],
            )

    def get_run(self, run_id: str) -> dict[str, object]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown run {run_id}")
        return dict(row)

    def get_images(self, run_id: str) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM images WHERE run_id = ? ORDER BY idx", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def run_summary(self, run_id: str) -> schemas.RunSummary:
        return _row_to_summary(self.get_run(run_id), self.completed_count(run_id))

    def completed_count(self, run_id: str) -> int:
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM images WHERE run_id = ? AND status = 'completed'",
                (run_id,),
            ).fetchone()[0]

    def list_runs(self, filters: RunFilters) -> tuple[list[schemas.RunSummary], int]:
        clauses: list[str] = []
        values: list[object] = []
        if filters.trashed == "exclude":
            clauses.append("trashed_at IS NULL")
        elif filters.trashed == "only":
            clauses.append("trashed_at IS NOT NULL")
        if filters.q:
            clauses.append("prompt LIKE ? ESCAPE '\\'")
            values.append(f"%{_escape_like(filters.q)}%")
        if filters.status is not None:
            clauses.append("status = ?")
            values.append(filters.status.value)
        if filters.repo_id is not None:
            clauses.append("repo_id = ?")
            values.append(filters.repo_id)
        if filters.favorite is not None:
            clauses.append("favorite = ?")
            values.append(1 if filters.favorite else 0)
        has_completed_image = (
            "EXISTS (SELECT 1 FROM images WHERE images.run_id = runs.run_id"
            " AND images.status = 'completed')"
        )
        if filters.has_images is not None:
            clauses.append(
                has_completed_image if filters.has_images else f"NOT {has_completed_image}"
            )
        if filters.exclude_empty_cancelled and filters.status is None:
            # An explicit status filter wins: asking for cancelled runs
            # must expose their empty cancellations too.
            clauses.append(f"(status != 'cancelled' OR {has_completed_image})")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            total = self._connection.execute(
                f"SELECT COUNT(*) FROM runs{where}", values
            ).fetchone()[0]
            rows = self._connection.execute(
                f"SELECT * FROM runs{where} ORDER BY queue_seq DESC LIMIT ? OFFSET ?",
                [*values, filters.limit, filters.offset],
            ).fetchall()
            if not rows:
                return [], total
            placeholders = ",".join("?" for _ in rows)
            counts = {
                row["run_id"]: row["n"]
                for row in self._connection.execute(
                    "SELECT run_id, COUNT(*) AS n FROM images"
                    f" WHERE run_id IN ({placeholders}) AND status = 'completed'"
                    " GROUP BY run_id",
                    [row["run_id"] for row in rows],
                )
            }
        return [_row_to_summary(row, counts.get(row["run_id"], 0)) for row in rows], total

    def mark_run_running(self, run_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE runs SET status = 'running', started_at = ? WHERE run_id = ?",
                (utc_now_iso(), run_id),
            )

    def complete_image(
        self,
        run_id: str,
        artifact_id: str,
        *,
        width: int,
        height: int,
        size_bytes: int,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE images SET status = 'completed', width = ?, height = ?,"
                " size_bytes = ?, completed_at = ? WHERE run_id = ? AND artifact_id = ?",
                (width, height, size_bytes, utc_now_iso(), run_id, artifact_id),
            )

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
        cancel_pending: bool = False,
    ) -> bool:
        """Apply a terminal transition; never overwrites a terminal status.

        Failed or cancelled runs with at least one completed image become
        ``partial`` so finished images are retained.
        """
        if status in (RunStatus.FAILED, RunStatus.CANCELLED):
            completed = self.completed_count(run_id)
            if completed > 0:
                status = RunStatus.PARTIAL
        with self._write() as conn:
            if cancel_pending:
                conn.execute(
                    "UPDATE images SET status = 'cancelled'"
                    " WHERE run_id = ? AND status = 'pending'",
                    (run_id,),
                )
            cursor = conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error_code = ?,"
                " error_message = ? WHERE run_id = ?"
                " AND status IN ('queued', 'paused', 'running')",
                (
                    status.value,
                    utc_now_iso(),
                    error_code.value if error_code else None,
                    error_message,
                    run_id,
                ),
            )
            return cursor.rowcount > 0

    def set_run_favorite(self, run_id: str, favorite: bool) -> schemas.RunSummary:
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE runs SET favorite = ? WHERE run_id = ?", (1 if favorite else 0, run_id)
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"unknown run {run_id}")
        return self.run_summary(run_id)

    def set_run_trashed(self, run_id: str, trashed: bool) -> schemas.RunSummary:
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE runs SET trashed_at = ? WHERE run_id = ?",
                (utc_now_iso() if trashed else None, run_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"unknown run {run_id}")
        return self.run_summary(run_id)

    def run_queue_position(self, run_id: str) -> int | None:
        """1-based position among queued/paused runs ordered by queue_seq."""
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None or row["status"] not in (RunStatus.QUEUED.value, RunStatus.PAUSED.value):
                return None
            return (
                self._connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE queue_seq <"
                    " (SELECT queue_seq FROM runs WHERE run_id = ?)"
                    " AND status IN ('queued', 'paused')",
                    (run_id,),
                ).fetchone()[0]
                + 1
            )

    def pending_runs(self) -> list[schemas.RunSummary]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM runs WHERE status IN ('queued', 'paused') ORDER BY queue_seq"
            ).fetchall()
        return [_row_to_summary(row, 0) for row in rows]

    def run_ids_with_status(self, status: RunStatus) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id FROM runs WHERE status = ? ORDER BY queue_seq", (status.value,)
            ).fetchall()
        return [row["run_id"] for row in rows]

    def paused_specs(self) -> list[FrozenRunSpec]:
        """Rebuild frozen specs for paused runs, in original submission order."""
        with self._lock:
            run_rows = self._connection.execute(
                "SELECT * FROM runs WHERE status = 'paused' ORDER BY queue_seq"
            ).fetchall()
            specs: list[FrozenRunSpec] = []
            for row in run_rows:
                image_rows = self._connection.execute(
                    "SELECT artifact_id, seed FROM images WHERE run_id = ?"
                    " AND status = 'pending' ORDER BY idx",
                    (row["run_id"],),
                ).fetchall()
                if not image_rows:
                    continue
                specs.append(
                    FrozenRunSpec(
                        run_id=row["run_id"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        model=FrozenModel(
                            registration_id=row["registration_id"],
                            repo_id=row["repo_id"],
                            commit_sha=row["commit_sha"],
                            profile=ProfileId(row["profile"]),
                            dtype=row["dtype"],
                            snapshot_path=str(
                                snapshot_path(self.hub_cache_dir, row["repo_id"], row["commit_sha"])
                            ),
                            sources=resolve_sources(self.hub_cache_dir, json.loads(row["sources"])),
                        ),
                        gpu=FrozenGpu(uuid=row["gpu_uuid"], name=row["gpu_name"]),
                        prompt=row["prompt"],
                        negative_prompt=row["negative_prompt"],
                        width=row["width"],
                        height=row["height"],
                        steps=row["steps"],
                        guidance=row["guidance"],
                        image_count=len(image_rows),
                        seeds=tuple(image_row["seed"] for image_row in image_rows),
                        artifact_ids=tuple(image_row["artifact_id"] for image_row in image_rows),
                    )
                )
        return specs

    def mark_paused_run_queued(self, run_id: str) -> bool:
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = 'queued' WHERE run_id = ? AND status = 'paused'",
                (run_id,),
            )
            return cursor.rowcount == 1

    def fail_run(self, run_id: str, code: ErrorCode, message: str) -> None:
        self.finish_run(
            run_id, RunStatus.FAILED, error_code=code, error_message=message, cancel_pending=True
        )

    def reconcile_runs(self) -> tuple[int, int]:
        """After a restart: running -> interrupted, queued -> paused."""
        with self._write() as conn:
            interrupted = conn.execute(
                "UPDATE runs SET status = 'interrupted', finished_at = ?,"
                " error_code = ?, error_message = 'interrupted by server shutdown'"
                " WHERE status = 'running'",
                (utc_now_iso(), ErrorCode.INTERNAL.value),
            ).rowcount
            paused = conn.execute(
                "UPDATE runs SET status = 'paused' WHERE status = 'queued'"
            ).rowcount
        return interrupted, paused

    def update_run_runtime_metadata(
        self,
        run_id: str,
        *,
        pipeline_class: str | None = None,
        dependency_versions: dict[str, str] | None = None,
        runtime_meta: dict[str, object] | None = None,
    ) -> None:
        """Record worker-reported runtime identity; omitted fields are kept."""
        assignments = []
        values: list[object] = []
        if pipeline_class is not None:
            assignments.append("pipeline_class = ?")
            values.append(pipeline_class)
        if dependency_versions is not None:
            assignments.append("dependency_versions = ?")
            values.append(json.dumps(dependency_versions))
        if runtime_meta is not None:
            assignments.append("runtime_meta = ?")
            values.append(json.dumps(runtime_meta))
        if not assignments:
            return
        values.append(run_id)
        with self._write() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?", values)


# ----- row conversion helpers ------------------------------------------------


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_registration(row: sqlite3.Row, cache_dir: Path) -> schemas.ModelRegistration:
    return schemas.ModelRegistration(
        id=row["id"],
        repo_id=row["repo_id"],
        commit_sha=row["commit_sha"],
        profile=ProfileId(row["profile"]),
        sources=resolve_sources(cache_dir, json.loads(row["sources"])),
        display_name=row["display_name"],
        status=RegistrationStatus(row["status"]),
        missing_files=json.loads(row["missing_files"]),
        snapshot_path=str(snapshot_path(cache_dir, row["repo_id"], row["commit_sha"])),
        created_at=row["created_at"],
        last_used_at=_parse_ts(row["last_used_at"]),
    )


def _row_to_download(row: sqlite3.Row, cache_dir: Path) -> DownloadJob:
    error = None
    if row["error_code"]:
        error = schemas.ErrorInfo(
            code=ErrorCode(row["error_code"]), message=row["error_message"] or ""
        )
    return DownloadJob(
        id=row["id"],
        repo_id=row["repo_id"],
        requested_revision=row["requested_revision"],
        sources=resolve_sources(cache_dir, json.loads(row["sources"])),
        resolved_commit=row["resolved_commit"],
        profile=ProfileId(row["profile"]),
        status=DownloadStatus(row["status"]),
        error=error,
        progress=schemas.DownloadProgress(
            bytes_done=row["bytes_done"],
            bytes_total=row["bytes_total"],
            files_done=row["files_done"],
            files_total=row["files_total"],
        ),
        created_at=row["created_at"],
        started_at=_parse_ts(row["started_at"]),
        finished_at=_parse_ts(row["finished_at"]),
    )


def _row_to_summary(row: sqlite3.Row, completed_count: int) -> schemas.RunSummary:
    preview = None
    if completed_count > 0:
        preview = schemas.artifact_id(min(completed_count, row["image_count"]))
    return schemas.RunSummary(
        run_id=row["run_id"],
        created_at=_parse_ts(row["created_at"]),
        status=RunStatus(row["status"]),
        favorite=bool(row["favorite"]),
        trashed=row["trashed_at"] is not None,
        prompt=row["prompt"],
        negative_prompt=row["negative_prompt"],
        repo_id=row["repo_id"],
        profile=ProfileId(row["profile"]),
        image_count=row["image_count"],
        completed_count=completed_count,
        preview_artifact_id=preview,
    )


def run_detail_row(
    row: dict[str, object],
    images: list[dict[str, object]],
    *,
    cache_dir: Path,
    queue_position: int | None = None,
    progress: schemas.RunProgressSnapshot | None = None,
) -> schemas.RunDetail:
    """Build a RunDetail from raw rows; shared by coordinator and API layer."""
    error = None
    if row.get("error_code"):
        error = schemas.ErrorInfo(
            code=ErrorCode(row["error_code"]), message=str(row.get("error_message") or "")
        )
    gpu = (
        FrozenGpu(uuid=str(row["gpu_uuid"]), name=str(row["gpu_name"]))
        if row.get("gpu_uuid")
        else None
    )
    return schemas.RunDetail(
        run_id=str(row["run_id"]),
        created_at=_parse_ts(str(row["created_at"])),
        started_at=_parse_ts(row.get("started_at")),
        finished_at=_parse_ts(row.get("finished_at")),
        status=RunStatus(str(row["status"])),
        favorite=bool(row.get("favorite")),
        trashed=row.get("trashed_at") is not None,
        registration_id=str(row["registration_id"]),
        repo_id=str(row["repo_id"]),
        commit_sha=str(row["commit_sha"]),
        profile=ProfileId(str(row["profile"])),
        sources=resolve_sources(cache_dir, json.loads(row.get("sources") or "[]")),
        dtype=str(row["dtype"]),
        gpu=gpu,
        prompt=str(row["prompt"]),
        negative_prompt=row.get("negative_prompt"),
        width=int(row["width"]),
        height=int(row["height"]),
        steps=int(row["steps"]),
        guidance=float(row["guidance"]),
        initial_seed=int(row["initial_seed"]),
        image_count=int(row["image_count"]),
        pipeline_class=row.get("pipeline_class"),
        dependency_versions=json.loads(row.get("dependency_versions") or "{}"),
        runtime_meta=json.loads(row.get("runtime_meta") or "{}"),
        queue_position=queue_position,
        progress=progress,
        error=error,
        images=[
            schemas.ArtifactView(
                artifact_id=str(image["artifact_id"]),
                index=int(image["idx"]),
                seed=int(image["seed"]),
                status=ImageStatus(str(image["status"])),
                width=image.get("width"),
                height=image.get("height"),
                size_bytes=image.get("size_bytes"),
                error=image.get("error"),
            )
            for image in images
        ],
    )


def _dump_sources(sources: tuple[schemas.ModelSource, ...]) -> str:
    return json.dumps(
        [source.model_dump(mode="json", exclude={"snapshot_path"}) for source in sources]
    )
