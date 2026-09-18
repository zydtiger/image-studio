"""CPU-only fake worker process for spawn-process lifecycle tests.

Implements exactly the :mod:`protocol` the real worker speaks — same
commands, same events, same acknowledgement backpressure — without torch,
diffusers, or any GPU state. ``FakeScript`` configures deterministic
latency and fault injection; an optional JSON-lines trace file records
lifecycle transitions with ``time.time_ns()`` stamps and the child pid so
tests can assert cross-process ordering (for example: the previous worker
fully exited before its replacement started loading).

This module is an internal test fixture of the inference area. Production
code never selects it; ``create_runtime(spawn_target=...)`` injections in
tests do.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time

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
from image_studio.schemas import ErrorCode, ErrorInfo

_PARENT_POLL_SECONDS = 0.2

#: Per-process run counter backing the ``crash_on_run``/``fault_on_run``
#: script hooks (each spawned worker starts at zero).
_run_ordinal = 0


class _Tracer:
    """Append-only JSONL trace; every record is flushed immediately."""

    def __init__(self, path: str | None) -> None:
        self._path = path

    def __call__(self, event: str, **fields: object) -> None:
        if not self._path:
            return
        record = {"t_ns": time.time_ns(), "pid": os.getpid(), "event": event}
        record.update(fields)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _parent_alive() -> bool:
    parent = multiprocessing.parent_process()
    return parent is None or parent.is_alive()


def _drain_pending_commands(
    cmd_conn: multiprocessing.connection.Connection,
    run_id: str,  # type: ignore[name-defined]
) -> tuple[bool, bool]:
    cancel_requested = False
    shutdown_requested = False
    while cmd_conn.poll(0):
        msg = cmd_conn.recv()
        if isinstance(msg, CmdCancel):
            cancel_requested = cancel_requested or msg.run_id == run_id
        elif isinstance(msg, CmdShutdown):
            shutdown_requested = True
    return cancel_requested, shutdown_requested


def _await_ack(
    cmd_conn: multiprocessing.connection.Connection,
    run_id: str,  # type: ignore[name-defined]
) -> tuple[bool, bool, bool]:
    cancel_requested = False
    while True:
        if not _parent_alive():
            sys.exit(0)
        if cmd_conn.poll(_PARENT_POLL_SECONDS):
            msg = cmd_conn.recv()
            if isinstance(msg, CmdImageAck) and msg.run_id == run_id:
                return msg.ok, cancel_requested, False
            if isinstance(msg, CmdCancel):
                cancel_requested = cancel_requested or msg.run_id == run_id
            elif isinstance(msg, CmdShutdown):
                return False, cancel_requested, True


def _fake_png(seed: int, width: int, height: int) -> bytes:
    import io

    from PIL import Image

    color = ((seed >> 16) & 0xFF, (seed >> 8) & 0xFF, seed & 0xFF)
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fake_worker_main(launch: WorkerLaunch, cmd_conn, event_q) -> None:  # type: ignore[no-untyped-def]
    script = launch.fake_script or FakeScript()
    trace = _Tracer(script.trace_path)

    trace("load_start", gpu=launch.gpu_uuid, repo=launch.repo_id)
    deadline = time.monotonic() + script.load_delay
    while time.monotonic() < deadline:
        if not _parent_alive():
            return
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if script.fail_load:
        trace("fault", stage="load")
        event_q.put(
            MsgFault(
                error=ErrorInfo(code=ErrorCode.WORKER_ERROR, message="fake: scripted load failure"),
                stage="load",
            )
        )
        return
    trace("load_end")
    event_q.put(
        MsgReady(
            pipeline_class="FakeZImagePipeline",
            device_name="fake-device",
            device_uuid=launch.gpu_uuid,
            uuid_verified=True,
            versions={"fake": "1"},
        )
    )

    while True:
        if not _parent_alive():
            trace("worker_exit", reason="parent-gone")
            return
        if not cmd_conn.poll(_PARENT_POLL_SECONDS):
            continue
        msg = cmd_conn.recv()
        if isinstance(msg, CmdShutdown):
            trace("shutdown", reason=msg.reason)
            break
        if isinstance(msg, CmdRun):
            if _execute_run(msg.task, script, cmd_conn, event_q, trace):
                break
    trace("worker_exit", reason="commanded")


def _execute_run(  # type: ignore[no-untyped-def]
    task: RunTask, script: FakeScript, cmd_conn, event_q, trace: _Tracer
) -> bool:
    """Mirror of the real run loop; True means stop serving (shutdown/fault)."""

    global _run_ordinal
    _run_ordinal += 1
    ordinal = _run_ordinal
    trace("run_start", run_id=task.run_id, ordinal=ordinal)
    completed = 0
    if script.crash_on_run == ordinal:
        trace("crash", run_id=task.run_id, ordinal=ordinal)
        os._exit(70)
    if script.fault_on_run == ordinal:
        trace("fault", stage="generate", run_id=task.run_id, ordinal=ordinal)
        event_q.put(
            MsgFault(
                error=ErrorInfo(code=ErrorCode.WORKER_ERROR, message="fake: scripted OOM fault"),
                stage="generate",
                run_id=task.run_id,
            )
        )
        return True
    for index, (seed, artifact_id) in enumerate(
        zip(task.seeds, task.artifact_ids, strict=True), start=1
    ):
        trace("image_start", run_id=task.run_id, index=index)
        if script.crash_during == index:
            trace("crash", run_id=task.run_id, index=index)
            os._exit(70)
        if script.fault_during == index:
            trace("fault", stage="generate", run_id=task.run_id, index=index)
            event_q.put(
                MsgFault(
                    error=ErrorInfo(
                        code=ErrorCode.WORKER_ERROR, message="fake: scripted OOM fault"
                    ),
                    stage="generate",
                    run_id=task.run_id,
                )
            )
            return True

        for step in range(task.steps):
            cancel_requested, shutdown_requested = _drain_pending_commands(cmd_conn, task.run_id)
            if shutdown_requested:
                trace("cancel", run_id=task.run_id, index=index, reason="shutdown")
                event_q.put(
                    MsgRunFinished(
                        run_id=task.run_id,
                        status="cancelled",
                        completed_count=completed,
                    )
                )
                return True
            time.sleep(script.step_delay)
            event_q.put(
                MsgProgress(
                    run_id=task.run_id,
                    image_index=index,
                    step=step + 1,
                    total_steps=task.steps,
                )
            )
            if cancel_requested:
                trace("cancel", run_id=task.run_id, index=index, reason="request")
                event_q.put(
                    MsgRunFinished(
                        run_id=task.run_id,
                        status="cancelled",
                        completed_count=completed,
                    )
                )
                return False

        png = _fake_png(seed, task.width, task.height)
        event_q.put(
            MsgImageCompleted(
                run_id=task.run_id,
                artifact_id=artifact_id,
                index=index,
                seed=seed,
                width=task.width,
                height=task.height,
                png=png,
            )
        )
        ack_ok, cancel_requested, shutdown_requested = _await_ack(cmd_conn, task.run_id)
        trace("ack", run_id=task.run_id, index=index, ok=ack_ok)
        if shutdown_requested:
            trace("cancel", run_id=task.run_id, index=index, reason="shutdown")
            event_q.put(
                MsgRunFinished(run_id=task.run_id, status="cancelled", completed_count=completed)
            )
            return True
        if not ack_ok:
            trace("abort_nack", run_id=task.run_id, index=index)
            event_q.put(
                MsgRunFinished(run_id=task.run_id, status="aborted", completed_count=completed)
            )
            return False
        completed = index
        trace("image_end", run_id=task.run_id, index=index)
        if cancel_requested:
            event_q.put(
                MsgRunFinished(run_id=task.run_id, status="cancelled", completed_count=completed)
            )
            return False

    event_q.put(MsgRunFinished(run_id=task.run_id, status="completed", completed_count=completed))
    return False
