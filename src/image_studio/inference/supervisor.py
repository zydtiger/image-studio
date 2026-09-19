"""Global FIFO generation supervisor: the public ``schemas.Runtime``.

Architecture:

- One dispatch thread owns the worker lifecycle and all sink delivery.
  Events are delivered strictly serialized; the next event is delivered
  only after the previous handler returns. Delivering one
  ``image_completed`` synchronously and only then acknowledging the worker
  is the backpressure that guarantees durable persistence of an image
  completes before further inference proceeds.
- The API-facing methods (``attach``, ``list_gpus``, ``status``,
  ``submit``, ``cancel``, ``eject``, ``shutdown``) only touch guarded
  state and hand work to the dispatch thread; all of them except
  ``eject``/``shutdown`` are non-blocking.
- The worker is a spawned child process that exclusively holds the
  pipeline and CUDA. Reuse requires an identical
  ``(repo_id, commit_sha, profile, dtype, gpu_uuid)`` key; any change is a
  serialized replacement — the current task finishes, the previous worker
  fully exits (join confirmed), and only then the replacement loads.
  Workers never overlap. There is no idle timeout: the model stays
  resident until an explicit Eject, a replacement, or shutdown.

Eject and task dispatch are serialized under the same supervisor lock:
``eject`` records a request that the dispatch thread honors only from an
idle state, and a run claim transitions the state under the same lock, so
an eject either wins before the next run starts or is rejected with
``RuntimeConflictError`` while loading/generating/switching/ejecting.

Sink exception semantics (contract section 4): a raise during an active
run fails that run — remaining images stop, ``run_failed`` is emitted, and
no success is ever reported afterwards; a negative image acknowledgement
tells the worker to abort. If ``run_failed`` delivery itself raises, the
failure is recorded in ``status.last_error`` and serving continues.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any

from image_studio.inference.gpus import list_nvidia_gpus
from image_studio.inference.profiles import worker_identity
from image_studio.inference.protocol import (
    CmdCancel,
    CmdImageAck,
    CmdRun,
    CmdShutdown,
    FakeScript,
    MsgFault,
    MsgImageCompleted,
    MsgProgress,
    MsgReady,
    MsgRunFinished,
    RunTask,
    WorkerLaunch,
)
from image_studio.inference.worker import worker_main
from image_studio.schemas import (
    CancelOutcome,
    CancelResult,
    ErrorCode,
    ErrorInfo,
    EventSink,
    FrozenGpu,
    FrozenRunSpec,
    GpuInfo,
    ImageCompleted,
    ImageStudioError,
    ResidentModel,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunProgress,
    RunStarted,
    RuntimeConflictError,
    RuntimeStatus,
    WorkerState,
    WorkerStateChanged,
    utc_now,
)

GpuProvider = Callable[[], list[GpuInfo]]
SpawnTarget = Callable[..., None]

#: Sentinel returned by :meth:`_get_worker_event` when the child is gone
#: and no buffered messages remain.
_DEAD = object()


class _Stopping(Exception):
    """Internal: shutdown was requested; abandon the current activity."""


class _WorkerFault(Exception):
    def __init__(self, error: ErrorInfo) -> None:
        super().__init__(error.message)
        self.error = error


class _WorkerDied(Exception):
    """The child exited without a terminal report (crash or kill)."""


@dataclass
class _WorkerHandle:
    process: multiprocessing.process.BaseProcess
    cmd_conn: multiprocessing.connection.Connection
    event_q: multiprocessing.Queue  # type: ignore[type-arg]
    launch: WorkerLaunch
    ready: MsgReady | None = field(default=None)


class InferenceSupervisor:
    """The single global FIFO generation supervisor (``schemas.Runtime``)."""

    implementation = "real"

    def __init__(
        self,
        *,
        gpu_provider: GpuProvider = list_nvidia_gpus,
        spawn_target: SpawnTarget = worker_main,
        fake_script: FakeScript | None = None,
        poll_interval: float = 0.2,
        stop_grace: float = 30.0,
        join_grace: float = 10.0,
        action_timeout: float = 120.0,
        thread_name: str = "image-studio-inference",
    ) -> None:
        self._gpu_provider = gpu_provider
        self._spawn_target = spawn_target
        # Test hook: attached to launches when a fake spawn target runs;
        # the real worker ignores it. Production keeps None.
        self._fake_script = fake_script
        self._poll_interval = poll_interval
        self._stop_grace = stop_grace
        self._join_grace = join_grace
        self._action_timeout = action_timeout
        self._ctx = multiprocessing.get_context("spawn")

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._sink: EventSink | None = None
        self._queue: deque[FrozenRunSpec] = deque()
        self._pending_terminal: deque[str] = deque()
        self._state = WorkerState.UNLOADED
        self._resident: ResidentModel | None = None
        self._current_run_id: str | None = None
        self._last_error: ErrorInfo | None = None
        self._cancel_requested: set[str] = set()
        self._stopping = False
        self._stopped = False
        self._shutdown_future: Future[None] | None = None
        self._eject_future: Future[None] | None = None

        # Dispatch-thread-owned state (mutated only on that thread).
        self._worker: _WorkerHandle | None = None

        self._thread = threading.Thread(target=self._dispatch_loop, name=thread_name, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # schemas.Runtime surface
    # ------------------------------------------------------------------

    def attach(self, sink: EventSink) -> None:
        with self._lock:
            if self._stopped or self._stopping:
                raise ImageStudioError(
                    ErrorCode.INTERNAL, "cannot attach: runtime is shutting down"
                )
            if self._sink is not None:
                raise ImageStudioError(ErrorCode.INTERNAL, "an event sink is already attached")
            self._sink = sink

    def list_gpus(self) -> list[GpuInfo]:
        return list(self._gpu_provider())

    def status(self) -> RuntimeStatus:
        with self._lock:
            return RuntimeStatus(
                implementation=self.implementation,
                state=self._state,
                resident=self._resident,
                current_run_id=self._current_run_id,
                queue_depth=len(self._queue),
                last_error=self._last_error,
            )

    def submit(self, spec: FrozenRunSpec) -> None:
        self._check_spec(spec)
        with self._lock:
            if self._stopping or self._stopped:
                raise ImageStudioError(
                    ErrorCode.INTERNAL, "cannot submit: runtime is shutting down"
                )
            if self._sink is None:
                raise ImageStudioError(
                    ErrorCode.INTERNAL, "cannot submit before a sink is attached"
                )
            self._queue.append(spec)
            self._cond.notify_all()

    def cancel(self, run_id: str) -> CancelResult:
        with self._lock:
            for index, spec in enumerate(self._queue):
                if spec.run_id == run_id:
                    del self._queue[index]
                    self._pending_terminal.append(run_id)
                    self._cond.notify_all()
                    return CancelResult(outcome=CancelOutcome.REMOVED_FROM_QUEUE)
            if run_id == self._current_run_id and self._state in (
                WorkerState.LOADING,
                WorkerState.GENERATING,
                WorkerState.SWITCHING,
            ):
                self._cancel_requested.add(run_id)
                self._cond.notify_all()
                return CancelResult(outcome=CancelOutcome.CANCELLING)
            return CancelResult(outcome=CancelOutcome.ALREADY_FINISHED)

    def eject(self) -> None:
        with self._lock:
            if self._stopping or self._stopped:
                raise RuntimeConflictError("cannot eject: runtime is shutting down")
            if self._state in (
                WorkerState.LOADING,
                WorkerState.GENERATING,
                WorkerState.SWITCHING,
                WorkerState.EJECTING,
            ):
                raise RuntimeConflictError(f"cannot eject while the worker is {self._state.value}")
            if self._eject_future is not None:
                raise RuntimeConflictError("an eject is already in progress")
            future: Future[None] = Future()
            self._eject_future = future
            self._cond.notify_all()
        future.result(timeout=self._action_timeout + self._stop_grace)

    def shutdown(self) -> None:
        with self._lock:
            if self._stopped:
                return
            future = self._shutdown_future
            if future is None:
                self._stopping = True
                future = Future()
                self._shutdown_future = future
                self._cond.notify_all()
        future.result(timeout=self._action_timeout + self._stop_grace + self._join_grace)
        self._thread.join(timeout=self._join_grace)
        with self._lock:
            self._stopped = True

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while True:
            kind, payload = self._next_action()
            if kind == "stop":
                self._do_shutdown()
                return
            if kind == "eject":
                self._do_eject()
                continue
            if kind == "terminal":
                self._emit_terminal_cancelled(payload, 0)
                continue
            if kind == "dead-worker":
                self._handle_idle_worker_exit()
                continue
            spec, plan = payload
            try:
                self._execute_run(spec, plan)
            except _Stopping:
                continue
            except Exception as exc:  # noqa: BLE001 - keep serving
                self._record_internal_error(exc)
                # The worker protocol state is unknown: fail the run and
                # stop the worker for a consistent unloaded state.
                if self._worker is not None:
                    self._stop_worker("supervisor-error")
                self._emit_state(WorkerState.UNLOADED, reason="supervisor-error")
                self._emit_terminal_failed(
                    spec.run_id,
                    ErrorInfo(
                        code=ErrorCode.INTERNAL,
                        message=f"supervisor error: {type(exc).__name__}: {exc}",
                    ),
                    0,
                )
                self._finish_run(spec.run_id)

    def _next_action(self) -> tuple[str, Any]:
        with self._lock:
            while True:
                if self._pending_terminal:
                    return "terminal", self._pending_terminal.popleft()
                if self._eject_future is not None:
                    return "eject", None
                if self._stopping:
                    return "stop", None
                # An idle worker may have exited without reporting (external
                # kill, OOM): observe it before claiming work so the reuse
                # plan never targets a dead process and residency stays
                # truthful. This is liveness observation, not an idle
                # timeout — a healthy worker is never unloaded here.
                if self._worker is not None and not self._worker.process.is_alive():
                    return "dead-worker", None
                if self._queue:
                    spec = self._queue.popleft()
                    self._current_run_id = spec.run_id
                    identity = worker_identity(spec)
                    if self._worker is None:
                        self._state = WorkerState.LOADING
                        plan = "spawn"
                    elif self._worker.launch.identity == identity:
                        self._state = WorkerState.GENERATING
                        plan = "reuse"
                    else:
                        self._state = WorkerState.SWITCHING
                        plan = "replace"
                    return "run", (spec, plan)
                self._cond.wait(self._poll_interval)

    # ------------------------------------------------------------------
    # Run execution
    # ------------------------------------------------------------------

    def _execute_run(self, spec: FrozenRunSpec, plan: str) -> None:
        if plan == "replace":
            reason = self._replacement_reason(spec)
            self._emit_state(WorkerState.SWITCHING, reason=reason)
            self._stop_worker("replace")
        if plan in ("replace", "spawn"):
            self._emit_state(WorkerState.LOADING)
            try:
                self._spawn_and_wait_ready(spec)
            except _WorkerFault as fault:
                self._worker = None
                self._set_resident(None)
                self._emit_state(WorkerState.UNLOADED, error=fault.error, reason="load-failed")
                self._record_last_error(fault.error)
                self._emit_terminal_failed(spec.run_id, fault.error, 0)
                self._finish_run(spec.run_id)
                return
            except _WorkerDied as died:
                self._worker = None
                self._set_resident(None)
                error = ErrorInfo(
                    code=ErrorCode.WORKER_ERROR,
                    message=f"worker process died during load: {died}",
                )
                self._emit_state(WorkerState.UNLOADED, error=error, reason="crashed")
                self._record_last_error(error)
                self._emit_terminal_failed(spec.run_id, error, 0)
                self._finish_run(spec.run_id)
                return
            if self._cancel_flag(spec.run_id):
                # Cancelled while loading; the freshly resident worker is
                # healthy, so keep it and skip generation.
                self._emit_state(WorkerState.IDLE, reason="cancelled-during-load")
                self._emit_terminal_cancelled(spec.run_id, 0)
                self._finish_run(spec.run_id)
                return

        exc = self._emit_state(WorkerState.GENERATING)
        if exc is None:
            ready = self._worker.ready if self._worker is not None else None
            exc = self._emit(
                RunStarted(
                    run_id=spec.run_id,
                    started_at=utc_now(),
                    pipeline_class=ready.pipeline_class if ready is not None else None,
                    dependency_versions=dict(ready.versions) if ready is not None else {},
                )
            )
        if exc is not None:
            # CmdRun has not been sent yet: the worker never received this
            # run. It is healthy and idle, so fail the run, keep residency,
            # and keep serving the queue — never cancel-and-wait for a
            # terminal an idle worker cannot produce.
            self._pre_dispatch_sink_failure(spec, exc)
            return

        try:
            self._send(CmdRun(task=self._task_for(spec)))
        except _WorkerDied as died:
            self._handle_worker_death(spec.run_id, completed=0, detail=str(died))
            return
        self._run_to_terminal(spec)

    def _run_to_terminal(self, spec: FrozenRunSpec) -> None:
        run_id = spec.run_id
        completed = 0
        terminal_emitted = False
        cancel_sent = False
        while True:
            if self._shutdown_pending():
                # Server shutdown mid-run: abort and leave the run
                # non-terminal; startup reconciliation marks it
                # interrupted (contract section 7).
                self._try_send(CmdShutdown(reason="server-shutdown"))
                self._stop_worker("shutdown")
                raise _Stopping
            if not cancel_sent and self._cancel_flag(run_id):
                try:
                    self._send(CmdCancel(run_id=run_id))
                    cancel_sent = True
                except _WorkerDied as died:
                    self._handle_worker_death(run_id, completed, str(died))
                    return

            msg = self._get_worker_event(self._poll_interval)
            if msg is None:
                continue
            if msg is _DEAD:
                self._handle_worker_death(
                    run_id, completed, "no terminal report", emit_terminal=not terminal_emitted
                )
                return

            if isinstance(msg, MsgProgress):
                if not terminal_emitted:
                    exc = self._emit(
                        RunProgress(
                            run_id=run_id,
                            image_index=msg.image_index,
                            step=msg.step,
                            total_steps=msg.total_steps,
                        )
                    )
                    if exc is not None and not terminal_emitted:
                        terminal_emitted = True
                        self._sink_failure(run_id, exc, completed)
                        self._try_send(CmdCancel(run_id=run_id))
                continue

            if isinstance(msg, MsgImageCompleted):
                if terminal_emitted:
                    # Unblock the child; the run is already failed.
                    self._try_send(CmdImageAck(run_id=run_id, ok=False))
                    continue
                exc = self._emit(
                    ImageCompleted(
                        run_id=run_id,
                        artifact_id=msg.artifact_id,
                        index=msg.index,
                        seed=msg.seed,
                        width=msg.width,
                        height=msg.height,
                        png=msg.png,
                    )
                )
                if exc is not None:
                    terminal_emitted = True
                    self._sink_failure(run_id, exc, completed)
                    self._try_send(CmdImageAck(run_id=run_id, ok=False))
                    continue
                completed = msg.index
                try:
                    self._send(CmdImageAck(run_id=run_id, ok=True))
                except _WorkerDied as died:
                    self._handle_worker_death(run_id, completed, str(died))
                    return
                continue

            if isinstance(msg, MsgFault):
                self._worker_fault(run_id, msg, completed, emit_terminal=not terminal_emitted)
                return

            if isinstance(msg, MsgRunFinished):
                if not terminal_emitted:
                    if msg.status == "completed":
                        self._emit_terminal(
                            RunCompleted(
                                run_id=run_id,
                                completed_count=msg.completed_count,
                                finished_at=utc_now(),
                            )
                        )
                    elif msg.status == "cancelled":
                        self._emit_terminal_cancelled(run_id, msg.completed_count)
                    elif msg.status == "aborted":
                        # Aborts only follow a negative acknowledgement,
                        # which already emitted the terminal. Any other
                        # abort is a worker defect.
                        error = ErrorInfo(
                            code=ErrorCode.WORKER_ERROR,
                            message="worker aborted the run without cause",
                        )
                        self._emit_terminal_failed(run_id, error, msg.completed_count)
                    else:
                        error = msg.error or ErrorInfo(
                            code=ErrorCode.WORKER_ERROR, message=msg.status
                        )
                        self._emit_terminal_failed(run_id, error, msg.completed_count)
                self._worker_idle()
                self._finish_run(run_id)
                return

            if isinstance(msg, MsgReady):
                continue  # stale/duplicate; nothing to do

    # ------------------------------------------------------------------
    # Worker lifecycle helpers
    # ------------------------------------------------------------------

    def _launch_for(self, spec: FrozenRunSpec) -> WorkerLaunch:
        launch = WorkerLaunch(
            registration_id=spec.model.registration_id,
            repo_id=spec.model.repo_id,
            commit_sha=spec.model.commit_sha,
            profile=spec.model.profile,
            dtype=spec.model.dtype,
            snapshot_path=spec.model.snapshot_path,
            sources=spec.model.sources,
            gpu_uuid=spec.gpu.uuid,
            gpu_name=spec.gpu.name,
        )
        if self._fake_script is not None:
            return replace(launch, fake_script=self._fake_script)
        return launch

    def _task_for(self, spec: FrozenRunSpec) -> RunTask:
        return RunTask(
            run_id=spec.run_id,
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt,
            width=spec.width,
            height=spec.height,
            steps=spec.steps,
            guidance=spec.guidance,
            seeds=tuple(spec.seeds),
            artifact_ids=tuple(spec.artifact_ids),
        )

    def _replacement_reason(self, spec: FrozenRunSpec) -> str:
        assert self._worker is not None
        current = self._worker.launch
        if (
            current.repo_id != spec.model.repo_id
            or current.commit_sha != spec.model.commit_sha
            or current.profile != spec.model.profile
            or current.dtype != spec.model.dtype
        ):
            return "model-changed"
        return "gpu-changed"

    def _spawn_and_wait_ready(self, spec: FrozenRunSpec) -> None:
        launch = self._launch_for(spec)
        parent_conn, child_conn = self._ctx.Pipe()
        event_q = self._ctx.Queue()
        process = self._ctx.Process(
            target=self._spawn_target,
            args=(launch, child_conn, event_q),
            daemon=True,
            name=f"image-studio-worker-{spec.gpu.uuid}",
        )
        process.start()
        child_conn.close()
        self._worker = _WorkerHandle(
            process=process, cmd_conn=parent_conn, event_q=event_q, launch=launch
        )

        while True:
            if self._shutdown_pending():
                self._stop_worker("shutdown-during-load", grace=self._join_grace)
                raise _Stopping
            msg = self._get_worker_event(self._poll_interval)
            if msg is None:
                continue
            if msg is _DEAD:
                self._worker = None
                process.join(self._join_grace)  # reap the zombie
                exitcode = process.exitcode
                raise _WorkerDied(f"exitcode={exitcode}")
            if isinstance(msg, MsgReady):
                self._worker.ready = msg
                resident = ResidentModel(
                    registration_id=launch.registration_id,
                    repo_id=launch.repo_id,
                    commit_sha=launch.commit_sha,
                    profile=launch.profile,
                    dtype=launch.dtype,
                    gpu=FrozenGpu(uuid=launch.gpu_uuid, name=launch.gpu_name),
                    pipeline_class=msg.pipeline_class,
                    dependency_versions=dict(msg.versions),
                )
                self._set_resident(resident)
                with self._lock:
                    self._last_error = None
                return
            if isinstance(msg, MsgFault):
                self._reap_faulted_process(self._worker)
                self._worker = None
                raise _WorkerFault(msg.error)

    def _stop_worker(self, reason: str, *, grace: float | None = None) -> None:
        handle = self._worker
        if handle is None:
            return
        self._try_send(CmdShutdown(reason=reason))
        handle.process.join(grace if grace is not None else self._stop_grace)
        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(self._join_grace)
        if handle.process.is_alive():
            handle.process.kill()
            handle.process.join(5.0)
        self._worker = None
        self._set_resident(None)

    def _get_worker_event(self, timeout: float) -> Any:
        assert self._worker is not None
        try:
            return self._worker.event_q.get(timeout=timeout)
        except queue.Empty:
            if self._worker.process.is_alive():
                return None
        except (EOFError, OSError):
            pass
        try:
            return self._worker.event_q.get(timeout=0.05)
        except (queue.Empty, EOFError, OSError):
            return _DEAD

    def _send(self, cmd: Any) -> None:
        assert self._worker is not None
        try:
            self._worker.cmd_conn.send(cmd)
        except (BrokenPipeError, OSError) as exc:
            raise _WorkerDied(f"command pipe broken: {exc}") from None

    def _try_send(self, cmd: Any) -> None:
        try:
            self._send(cmd)
        except _WorkerDied:
            pass

    def _handle_idle_worker_exit(self) -> None:
        """Reap an idle worker that exited without reporting.

        External kills or OOM can take down even a healthy idle child. Only
        this supervisor's own process is reaped; residency is cleared with a
        visible worker_error and an unloaded event, and the next run spawns
        a fresh worker normally. No retry of any completed run, no idle
        timeout: a live worker is never unloaded by this path.
        """
        handle = self._worker
        if handle is None:
            return
        handle.process.join(self._join_grace)
        self._worker = None
        self._set_resident(None)
        error = ErrorInfo(
            code=ErrorCode.WORKER_ERROR,
            message=(
                f"worker process exited unexpectedly while idle "
                f"(exitcode={handle.process.exitcode})"
            ),
        )
        self._emit_state(WorkerState.UNLOADED, error=error, reason="idle-exit")
        self._record_last_error(error)

    def _handle_worker_death(
        self, run_id: str, completed: int, detail: str, *, emit_terminal: bool = True
    ) -> None:
        handle = self._worker
        if handle is not None:
            handle.process.join(self._join_grace)
        self._worker = None
        self._set_resident(None)
        error = ErrorInfo(
            code=ErrorCode.WORKER_ERROR,
            message=f"worker process exited unexpectedly ({detail})",
        )
        self._emit_state(WorkerState.UNLOADED, error=error, reason="crashed")
        self._record_last_error(error)
        if emit_terminal:
            self._emit_terminal_failed(run_id, error, completed)
        self._finish_run(run_id)

    def _worker_fault(
        self, run_id: str, msg: MsgFault, completed: int, *, emit_terminal: bool = True
    ) -> None:
        handle = self._worker
        if handle is not None:
            self._reap_faulted_process(handle)
        self._worker = None
        self._set_resident(None)
        self._emit_state(WorkerState.UNLOADED, error=msg.error, reason="worker-fault")
        self._record_last_error(msg.error)
        if emit_terminal:
            self._emit_terminal_failed(run_id, msg.error, completed)
        self._finish_run(run_id)

    def _reap_faulted_process(self, handle: _WorkerHandle) -> None:
        """Join a self-reporting faulted worker; escalate if it hangs."""

        handle.process.join(self._join_grace)
        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(self._join_grace)
        if handle.process.is_alive():
            handle.process.kill()
            handle.process.join(5.0)

    def _worker_idle(self) -> None:
        self._emit_state(WorkerState.IDLE)

    # ------------------------------------------------------------------
    # Event delivery and terminal handling
    # ------------------------------------------------------------------

    def _emit(self, event: Any) -> Exception | None:
        with self._lock:
            sink = self._sink
        if sink is None:
            return None
        try:
            sink.on_event(event)
            return None
        except Exception as exc:  # noqa: BLE001 - propagated to the caller
            return exc

    def _emit_state(
        self,
        state: WorkerState,
        *,
        reason: str | None = None,
        error: ErrorInfo | None = None,
    ) -> Exception | None:
        with self._lock:
            self._state = state
            resident = self._resident
        exc = self._emit(
            WorkerStateChanged(state=state, resident=resident, reason=reason, error=error)
        )
        if exc is not None:
            # Worker-state notifications are not run events: a sink raise
            # never fails the run (except the run-scoped generating state,
            # whose caller handles the failure). Record the delivery failure
            # visibly in status.last_error and keep serving.
            self._record_last_error(
                ErrorInfo(
                    code=ErrorCode.STORAGE_ERROR,
                    message=(f"event sink failed on worker_state_changed -> {state.value}: {exc}"),
                )
            )
        return exc

    def _emit_terminal(self, event: Any) -> None:
        exc = self._emit(event)
        if exc is not None:
            self._record_last_error(
                ErrorInfo(
                    code=ErrorCode.STORAGE_ERROR,
                    message=(
                        f"event sink failed on {event.event} for "
                        f"{getattr(event, 'run_id', '?')}: {exc}"
                    ),
                )
            )

    def _emit_terminal_cancelled(self, run_id: str, completed: int) -> None:
        self._emit_terminal(
            RunCancelled(run_id=run_id, completed_count=completed, finished_at=utc_now())
        )

    def _emit_terminal_failed(self, run_id: str, error: ErrorInfo, completed: int) -> None:
        self._emit_terminal(
            RunFailed(
                run_id=run_id,
                error=error,
                completed_count=completed,
                finished_at=utc_now(),
            )
        )

    def _sink_failure(self, run_id: str, exc: Exception, completed: int) -> None:
        """A sink raise during an active run fails that run (contract S4).

        The terminal is emitted here; remaining images stop via a negative
        acknowledgement or a cancel command, and the worker stays resident
        and reusable.
        """

        error = ErrorInfo(
            code=ErrorCode.STORAGE_ERROR,
            message=f"event sink failed during run {run_id}: {exc}",
        )
        self._record_last_error(error)
        self._emit_terminal_failed(run_id, error, completed)

    def _pre_dispatch_sink_failure(self, spec: FrozenRunSpec, exc: Exception) -> None:
        """Sink raised on generating/run_started, before ``CmdRun`` was sent.

        The worker never received this run: it is healthy and resident, and it
        cannot produce a ``MsgRunFinished`` for a run it does not know about.
        Fail the run with the storage error, return the worker to idle, clear
        the run claim, and keep serving the queue. Exactly-one-terminal
        enforcement guarantees no success can be reported afterwards.
        """

        error = ErrorInfo(
            code=ErrorCode.STORAGE_ERROR,
            message=f"event sink failed before dispatching run {spec.run_id}: {exc}",
        )
        self._record_last_error(error)
        self._emit_terminal_failed(spec.run_id, error, 0)
        self._worker_idle()
        self._finish_run(spec.run_id)

    # ------------------------------------------------------------------
    # Eject and shutdown execution
    # ------------------------------------------------------------------

    def _do_eject(self) -> None:
        with self._lock:
            future = self._eject_future
            self._eject_future = None
        try:
            if self._worker is not None:
                self._emit_state(WorkerState.EJECTING, reason="eject")
                self._stop_worker("eject")
            self._emit_state(WorkerState.UNLOADED, reason="eject")
            future.set_result(None)
        except Exception as exc:  # noqa: BLE001 - resolve the caller's future
            future.set_exception(exc)

    def _do_shutdown(self) -> None:
        with self._lock:
            eject_future = self._eject_future
            self._eject_future = None
            self._queue.clear()
            self._current_run_id = None
            shutdown_future = self._shutdown_future
        if eject_future is not None:
            eject_future.set_exception(RuntimeConflictError("runtime is shutting down"))
        try:
            if self._worker is not None:
                self._stop_worker("shutdown")
            self._emit_state(WorkerState.UNLOADED, reason="shutdown")
        finally:
            if shutdown_future is not None:
                shutdown_future.set_result(None)

    # ------------------------------------------------------------------
    # Small guarded helpers
    # ------------------------------------------------------------------

    def _check_spec(self, spec: FrozenRunSpec) -> None:
        problems: list[str] = []
        if len(spec.seeds) != spec.image_count:
            problems.append("seeds length does not match image_count")
        if len(spec.artifact_ids) != spec.image_count:
            problems.append("artifact_ids length does not match image_count")
        if not spec.prompt:
            problems.append("prompt must be non-empty")
        if spec.steps < 1:
            problems.append("steps must be >= 1")
        if problems:
            raise ImageStudioError(
                ErrorCode.INTERNAL,
                "malformed FrozenRunSpec: " + "; ".join(problems),
                details={"run_id": spec.run_id},
            )

    def _cancel_flag(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancel_requested

    def _finish_run(self, run_id: str) -> None:
        with self._lock:
            if self._current_run_id == run_id:
                self._current_run_id = None
            self._cancel_requested.discard(run_id)

    def _set_resident(self, resident: ResidentModel | None) -> None:
        with self._lock:
            self._resident = resident

    def _record_last_error(self, error: ErrorInfo) -> None:
        with self._lock:
            self._last_error = error

    def _record_internal_error(self, exc: Exception) -> None:
        self._record_last_error(
            ErrorInfo(
                code=ErrorCode.INTERNAL,
                message=f"supervisor error: {type(exc).__name__}: {exc}",
            )
        )

    def _shutdown_pending(self) -> bool:
        with self._lock:
            return self._stopping


def create_runtime(
    *,
    gpu_provider: GpuProvider = list_nvidia_gpus,
    spawn_target: SpawnTarget = worker_main,
    fake_script: FakeScript | None = None,
    poll_interval: float = 0.2,
    stop_grace: float = 30.0,
    join_grace: float = 10.0,
    action_timeout: float = 120.0,
) -> InferenceSupervisor:
    """Construct and start the real runtime for backend composition.

    The dispatch thread starts immediately and stops via ``shutdown()``.
    ``attach`` must be called once before the first ``submit``. The extra
    keyword arguments are injection points for spawn-process tests.
    """

    return InferenceSupervisor(
        gpu_provider=gpu_provider,
        spawn_target=spawn_target,
        fake_script=fake_script,
        poll_interval=poll_interval,
        stop_grace=stop_grace,
        join_grace=join_grace,
        action_timeout=action_timeout,
    )
