"""Backend runtime coordinator: submission pipeline and event persistence.

The coordinator owns the contract's normalization pipeline (ids, timestamps,
profile normalization, seed resolution, freeze) before persistence and
before ``Runtime.submit``, and implements ``EventSink`` so the runtime never
touches SQLite or the artifact tree. Storage failures during event handling
fail the affected run instead of reporting success.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path

from image_studio import schemas
from image_studio.hub.anima import model_problems
from image_studio.hub.cache import problem_codes
from image_studio.schemas import (
    ErrorCode,
    GenerationRequest,
    ImageStudioError,
    RunStatus,
)
from image_studio.storage.artifacts import ArtifactStore
from image_studio.storage.repository import (
    NotFoundError,
    Repository,
    RunFilters,
    run_detail_row,
)

logger = logging.getLogger(__name__)


class RuntimeCoordinator:
    def __init__(
        self,
        repository: Repository,
        artifacts: ArtifactStore,
        runtime: schemas.Runtime,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts
        self._runtime = runtime
        self._progress: dict[str, schemas.RunProgressSnapshot] = {}
        self._queue_paused = False
        # Serializes (persist + runtime.submit) against resume_queue so the
        # queue_seq order of accepted requests always matches dispatch order.
        # Never held while handling runtime events: the dispatch thread calls
        # the event sink without this lock, so no deadlock is possible.
        self._submission_lock = threading.Lock()
        runtime.attach(self)

    # ----- submission pipeline ------------------------------------------------

    def submit(self, request: GenerationRequest) -> schemas.RunDetail:
        with self._submission_lock:
            if self._queue_paused:
                raise ImageStudioError(
                    ErrorCode.CONFLICT,
                    "generation queue is paused after a restart; resume it before "
                    "submitting new runs",
                )
            return self._submit_locked(request)

    def _submit_locked(self, request: GenerationRequest) -> schemas.RunDetail:
        registration = self._get_registration(request.registration_id)
        if registration.snapshot_path:
            # Always revalidate the registered fixed snapshot; the cached
            # status is display state only, so a repaired snapshot (same
            # fixed commit, no auto-update) becomes ready again and a broken
            # one keeps the same typed code across retries.
            found = model_problems(
                Path(registration.snapshot_path),
                registration.repo_id,
                registration.profile,
                registration.sources,
            )
            if found:
                problems = [problem.detail for problem in found]
                self._repository.set_registration_status(
                    registration.id,
                    schemas.RegistrationStatus.MISSING_FILES,
                    problems,
                )
                raise ImageStudioError(
                    problem_codes(found),
                    "registration is missing required files; download or repair it first",
                    {"missing_files": problems},
                )
            if registration.status != schemas.RegistrationStatus.READY:
                self._repository.set_registration_status(
                    registration.id, schemas.RegistrationStatus.READY, []
                )
        elif registration.status != schemas.RegistrationStatus.READY:
            raise ImageStudioError(
                ErrorCode.CACHE_INCOMPLETE,
                "registration is missing required files; download or repair it first",
                {"missing_files": list(registration.missing_files)},
            )
        # Contract §2 ordering: resolve the GPU before applying profile
        # defaults and profile-dependent validation.
        gpus = {gpu.uuid: gpu for gpu in self._runtime.list_gpus()}
        gpu = gpus.get(request.gpu_uuid)
        if gpu is None:
            raise ImageStudioError(
                ErrorCode.VALIDATION,
                f"unknown gpu_uuid {request.gpu_uuid!r}",
                {"field": "gpu_uuid", "available": sorted(gpus)},
            )
        profile = schemas.PROFILES[registration.profile]
        schemas.validate_generation(request, profile)

        seeds = schemas.resolve_seeds(request.seed, request.count)
        spec = schemas.FrozenRunSpec(
            run_id=uuid.uuid4().hex,
            created_at=schemas.utc_now(),
            model=schemas.FrozenModel(
                registration_id=registration.id,
                repo_id=registration.repo_id,
                commit_sha=registration.commit_sha,
                profile=registration.profile,
                dtype=profile.dtype,
                snapshot_path=registration.snapshot_path or "",
                sources=registration.sources,
            ),
            gpu=schemas.FrozenGpu(uuid=gpu.uuid, name=gpu.name),
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            steps=request.steps if request.steps is not None else profile.default_steps,
            guidance=(
                request.guidance if request.guidance is not None else profile.guidance_default
            ),
            image_count=request.count,
            seeds=seeds,
            artifact_ids=schemas.plan_artifact_ids(request.count),
        )
        self._repository.create_run(spec)
        self._repository.mark_registration_used(registration.id)
        try:
            self._runtime.submit(spec)
        except Exception as exc:
            logger.exception("runtime rejected submission for run %s", spec.run_id)
            self._repository.fail_run(spec.run_id, ErrorCode.INTERNAL, str(exc))
            raise
        return self.get_run(spec.run_id)

    # ----- queries ----------------------------------------------------------------

    def get_run(self, run_id: str) -> schemas.RunDetail:
        row = self._repository.get_run(run_id)
        images = self._repository.get_images(run_id)
        return run_detail_row(
            row,
            images,
            cache_dir=self._repository.hub_cache_dir,
            queue_position=self._repository.run_queue_position(run_id),
            progress=self._progress.get(run_id),
        )

    def list_runs(
        self,
        *,
        q: str | None = None,
        status: RunStatus | None = None,
        repo_id: str | None = None,
        favorite: bool | None = None,
        has_images: bool | None = None,
        exclude_empty_cancelled: bool = False,
        trashed: str = "exclude",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[schemas.RunSummary], int]:
        return self._repository.list_runs(
            RunFilters(
                q=q,
                status=status,
                repo_id=repo_id,
                favorite=favorite,
                has_images=has_images,
                exclude_empty_cancelled=exclude_empty_cancelled,
                trashed=trashed,
                limit=limit,
                offset=offset,
            )
        )

    def queue_state(self) -> schemas.QueueState:
        return schemas.QueueState(
            paused=self._queue_paused, pending=self._repository.pending_runs()
        )

    # ----- actions -----------------------------------------------------------------

    def cancel(self, run_id: str) -> schemas.RunDetail:
        row = self._repository.get_run(run_id)
        status = RunStatus(row["status"])
        if status in (RunStatus.QUEUED, RunStatus.RUNNING):
            result = self._runtime.cancel(run_id)
            if result.outcome is schemas.CancelOutcome.REMOVED_FROM_QUEUE:
                self._finish_terminal(
                    run_id, RunStatus.CANCELLED, error_message="cancelled while queued"
                )
            return self.get_run(run_id)
        if status is RunStatus.PAUSED:
            self._finish_terminal(
                run_id, RunStatus.CANCELLED, error_message="cancelled while paused"
            )
            return self.get_run(run_id)
        return self.get_run(run_id)

    def favorite(self, run_id: str, favorite: bool) -> schemas.RunSummary:
        return self._repository.set_run_favorite(run_id, favorite)

    def trash(self, run_id: str) -> schemas.RunSummary:
        row = self._repository.get_run(run_id)
        if row["status"] in ("queued", "paused", "running"):
            raise ImageStudioError(ErrorCode.CONFLICT, "cannot trash a run that is still active")
        if row["trashed_at"] is not None:
            raise ImageStudioError(ErrorCode.CONFLICT, "run is already in trash")
        try:
            self._artifacts.trash_run(_created_at(row), run_id)
        except FileNotFoundError:
            pass  # interrupted runs may have no files on disk yet
        return self._repository.set_run_trashed(run_id, True)

    def restore(self, run_id: str) -> schemas.RunSummary:
        row = self._repository.get_run(run_id)
        if row["trashed_at"] is None:
            raise ImageStudioError(ErrorCode.CONFLICT, "run is not in trash")
        try:
            self._artifacts.restore_run(_created_at(row), run_id)
        except FileNotFoundError:
            pass  # interrupted runs may have no files on disk
        return self._repository.set_run_trashed(run_id, False)

    def eject(self) -> None:
        self._runtime.eject()

    def runtime_status(self) -> schemas.RuntimeStatus:
        return self._runtime.status()

    def list_gpus(self) -> list[schemas.GpuInfo]:
        return self._runtime.list_gpus()

    def shutdown(self) -> None:
        self._runtime.shutdown()

    # ----- paused queue across restarts ------------------------------------------

    def mark_queue_paused(self) -> None:
        self._queue_paused = True

    def resume_queue(self) -> schemas.QueueState:
        # Hold the submission lock so the backlog is persisted-then-dispatched
        # in queue_seq order before any newly accepted submission can reach
        # the runtime; new submissions were rejected while paused.
        with self._submission_lock:
            resumed = 0
            for spec in self._repository.paused_specs():
                if not self._repository.mark_paused_run_queued(spec.run_id):
                    continue
                try:
                    problems = model_problems(
                        Path(spec.model.snapshot_path),
                        spec.model.repo_id,
                        spec.model.profile,
                        spec.model.sources,
                    )
                    if problems:
                        raise ImageStudioError(
                            problem_codes(problems),
                            "; ".join(problem.detail for problem in problems),
                        )
                    self._runtime.submit(spec)
                    resumed += 1
                except ImageStudioError as exc:
                    self._finish_terminal(
                        spec.run_id,
                        RunStatus.FAILED,
                        error_code=exc.code,
                        error_message=exc.message,
                    )
                except Exception as exc:
                    logger.exception("failed to resume run %s", spec.run_id)
                    self._repository.fail_run(spec.run_id, ErrorCode.INTERNAL, str(exc))
            self._queue_paused = False
        return schemas.QueueState(paused=False, pending=self._repository.pending_runs())

    # ----- EventSink ----------------------------------------------------------------

    def on_event(self, event: schemas.RuntimeEvent) -> None:
        handler = getattr(self, f"_on_{event.event}")
        handler(event)

    def _on_worker_state_changed(self, event: schemas.WorkerStateChanged) -> None:
        logger.debug("worker state -> %s (%s)", event.state, event.reason)

    def _on_run_started(self, event: schemas.RunStarted) -> None:
        self._repository.mark_run_running(event.run_id)
        if event.pipeline_class is not None or event.dependency_versions:
            self._repository.update_run_runtime_metadata(
                event.run_id,
                pipeline_class=event.pipeline_class,
                dependency_versions=event.dependency_versions,
            )

    def _on_run_progress(self, event: schemas.RunProgress) -> None:
        self._progress[event.run_id] = schemas.RunProgressSnapshot(
            image_index=event.image_index, step=event.step, total_steps=event.total_steps
        )

    def _on_image_completed(self, event: schemas.ImageCompleted) -> None:
        try:
            row = self._repository.get_run(event.run_id)
            size = self._artifacts.write_image(
                _created_at(row), event.run_id, event.artifact_id, event.png
            )
            self._repository.complete_image(
                event.run_id,
                event.artifact_id,
                width=event.width,
                height=event.height,
                size_bytes=size,
            )
            self._write_metadata(event.run_id)
        except Exception:
            logger.exception("durable write failed for %s", event.run_id)
            self._fail_run_for_storage(event.run_id)

    def _on_run_completed(self, event: schemas.RunCompleted) -> None:
        self._progress.pop(event.run_id, None)
        self._finish_terminal(event.run_id, RunStatus.COMPLETED)

    def _on_run_failed(self, event: schemas.RunFailed) -> None:
        self._progress.pop(event.run_id, None)
        self._finish_terminal(
            event.run_id,
            RunStatus.FAILED,
            error_code=event.error.code,
            error_message=event.error.message,
        )

    def _on_run_cancelled(self, event: schemas.RunCancelled) -> None:
        self._progress.pop(event.run_id, None)
        self._finish_terminal(event.run_id, RunStatus.CANCELLED, error_message="cancelled")

    def rewrite_metadata_for_interrupted(self) -> None:
        """Best-effort metadata parity for runs interrupted by a shutdown."""
        for run_id in self._repository.run_ids_with_status(RunStatus.INTERRUPTED):
            self._try_metadata(run_id)

    # ----- helpers --------------------------------------------------------------------

    def _finish_terminal(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> None:
        """Persist one terminal with metadata parity and bounded failures.

        Success is file-first: ``completed`` is recorded in the database only
        after the terminal metadata write succeeds, so a storage failure at
        the terminal can never be observed as success — the run flips to a
        visible ``storage_error`` terminal (``partial`` when images were
        retained) with one bounded metadata retry. Non-success terminals
        persist first and then rewrite metadata once; a failing write there
        never creates false success.

        A late terminal event arriving after the run is already durably
        terminal (for example a worker ``RunCompleted`` following an image
        sink storage failure that already terminalized the run as
        ``partial``/``storage_error``) can never override the durable state
        or advertise a contradicting metadata file: the metadata is rewritten
        from the authoritative database row only. Every database mutation is
        bounded: failures are logged and left to startup reconciliation
        instead of propagating into the runtime's dispatch thread.
        """
        try:
            row = self._repository.get_run(run_id)
        except Exception:  # noqa: BLE001 - bounded read
            logger.exception("terminal read failed for run %s", run_id)
            return
        current = RunStatus(row["status"])
        if current not in (RunStatus.QUEUED, RunStatus.PAUSED, RunStatus.RUNNING):
            # Durable terminal already exists: the late event's claimed
            # status is irrelevant; refresh metadata from the row only.
            self._try_metadata(run_id)
            return
        if status is RunStatus.COMPLETED:
            if self._try_metadata(run_id, status=RunStatus.COMPLETED):
                self._bounded_repository(self._repository.finish_run, run_id, RunStatus.COMPLETED)
                return
            logger.error("terminal metadata write failed for run %s", run_id)
            self._bounded_repository(
                self._repository.fail_run,
                run_id,
                ErrorCode.STORAGE_ERROR,
                "terminal metadata write failed; success was not recorded",
            )
            self._try_metadata(run_id)  # one bounded retry with the final state
            return
        self._bounded_repository(
            self._repository.finish_run,
            run_id,
            status,
            error_code=error_code,
            error_message=error_message,
            cancel_pending=True,
        )
        self._try_metadata(run_id)

    def _try_metadata(self, run_id: str, *, status: RunStatus | None = None) -> bool:
        """One bounded metadata write attempt; never raises into the runtime."""
        try:
            self._write_metadata(run_id, status=status)
            return True
        except Exception:  # noqa: BLE001 - bounded by the single attempt
            logger.exception("metadata write failed for run %s", run_id)
            return False

    def _bounded_repository(self, operation, *args, **kwargs) -> None:
        """Run one database mutation without letting failures escape the sink."""
        try:
            operation(*args, **kwargs)
        except Exception:  # noqa: BLE001 - startup reconciliation closes the gap
            logger.exception("database write failed for %s", args or operation)

    def _fail_run_for_storage(self, run_id: str) -> None:
        """A storage failure must fail the task, never report success."""
        try:
            self._repository.fail_run(
                run_id, ErrorCode.STORAGE_ERROR, "durable artifact write failed"
            )
        except Exception:
            logger.exception("could not persist storage failure for %s", run_id)
        try:
            self._runtime.cancel(run_id)
        except Exception:
            logger.exception("could not request cancellation for %s", run_id)

    def _write_metadata(self, run_id: str, *, status: RunStatus | None = None) -> None:
        row = self._repository.get_run(run_id)
        images = self._repository.get_images(run_id)
        payload_status = status.value if status is not None else row["status"]
        error_code = row.get("error_code") if status is None else None
        error_message = row.get("error_message") if status is None else None
        payload = {
            "run_id": run_id,
            "created_at": row["created_at"],
            "status": payload_status,
            "model": {
                "registration_id": row["registration_id"],
                "repo_id": row["repo_id"],
                "commit_sha": row["commit_sha"],
                "profile": row["profile"],
                "dtype": row["dtype"],
                "sources": json.loads(row["sources"]),
            },
            "gpu": {"uuid": row["gpu_uuid"], "name": row["gpu_name"]},
            "parameters": {
                "prompt": row["prompt"],
                "negative_prompt": row["negative_prompt"],
                "width": row["width"],
                "height": row["height"],
                "steps": row["steps"],
                "guidance": row["guidance"],
                "initial_seed": row["initial_seed"],
                "image_count": row["image_count"],
            },
            "images": [
                {
                    "artifact_id": image["artifact_id"],
                    "seed": image["seed"],
                    "status": image["status"],
                    "width": image["width"],
                    "height": image["height"],
                    "size_bytes": image["size_bytes"],
                    "completed_at": image["completed_at"],
                }
                for image in images
            ],
            "runtime": {
                "pipeline_class": row["pipeline_class"],
                "dependency_versions": json.loads(row["dependency_versions"] or "{}"),
                "runtime_meta": json.loads(row["runtime_meta"] or "{}"),
            },
        }
        if error_code:
            payload["error"] = {"code": error_code, "message": error_message}
        self._artifacts.write_metadata(_created_at(row), run_id, payload)

    def _get_registration(self, registration_id: str) -> schemas.ModelRegistration:
        try:
            return self._repository.get_registration(registration_id)
        except NotFoundError as exc:
            raise ImageStudioError(
                ErrorCode.NOT_FOUND, f"unknown registration {registration_id}"
            ) from exc


def _created_at(row: dict[str, object]) -> datetime:
    value = row["created_at"]
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
