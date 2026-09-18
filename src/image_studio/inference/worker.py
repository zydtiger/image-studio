"""Real inference worker: the only process that owns CUDA and the pipeline.

Spawned by the supervisor through the ``spawn`` start method so no CUDA
state is forked into children. The child pins ``CUDA_VISIBLE_DEVICES`` to
the selected GPU UUID *before* the first torch import (reliable UUID-based
device selection), loads strictly from the verified snapshot path, reports
``MsgReady``, and then serves run commands until shutdown, a fatal fault,
or parent-process death. It stays resident between runs: there is no idle
timeout and no automatic unload.

Cancellation is cooperative and checked only at safe inference boundaries:
inside the per-step callback and between images. After reporting an image
the worker blocks on ``CmdImageAck`` — the backpressure point that lets
durable persistence of one image complete before inference continues; a
negative acknowledgement aborts the remaining images of the run.
"""

from __future__ import annotations

import multiprocessing
import os
import sys

from image_studio.inference import z_image
from image_studio.inference.gpus import normalize_gpu_uuid
from image_studio.inference.protocol import (
    CmdCancel,
    CmdImageAck,
    CmdRun,
    CmdShutdown,
    MsgFault,
    MsgImageCompleted,
    MsgProgress,
    MsgReady,
    MsgRunFinished,
    RunTask,
    WorkerLaunch,
)
from image_studio.schemas import ErrorCode, ErrorInfo

_PARENT_POLL_SECONDS = 1.0


def _fault(
    event_q: multiprocessing.Queue,  # type: ignore[type-arg]
    code: ErrorCode,
    stage: str,
    exc: BaseException,
    run_id: str | None = None,
) -> None:
    event_q.put(
        MsgFault(
            error=ErrorInfo(code=code, message=f"{type(exc).__name__}: {exc}"),
            stage=stage,
            run_id=run_id,
        )
    )


def _parent_alive() -> bool:
    parent = multiprocessing.parent_process()
    return parent is None or parent.is_alive()


def _verify_device(launch: WorkerLaunch) -> tuple[str, str | None, bool]:
    """Confirm the child sees exactly the selected GPU.

    ``CUDA_VISIBLE_DEVICES`` was pinned to the UUID before torch was
    imported, so exactly one visible device is expected. When torch exposes
    the device UUID, it must match the requested one (normalized compare);
    otherwise visibility count alone is the evidence.
    """

    import torch

    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            f"expected exactly 1 visible CUDA device after pinning "
            f"CUDA_VISIBLE_DEVICES={launch.gpu_uuid!r}, saw {visible}"
        )
    props = torch.cuda.get_device_properties(0)
    raw_uuid = getattr(props, "uuid", None)
    device_name = getattr(props, "name", "cuda:0")
    if raw_uuid is not None:
        normalized = normalize_gpu_uuid(str(raw_uuid))
        if normalized != normalize_gpu_uuid(launch.gpu_uuid):
            raise RuntimeError(
                f"selected device UUID {normalized} does not match requested {launch.gpu_uuid!r}"
            )
        return device_name, str(raw_uuid), True
    return device_name, None, False


def _await_ack(
    cmd_conn: multiprocessing.connection.Connection,  # type: ignore[name-defined]
    run_id: str,
) -> tuple[bool, bool, bool]:
    """Block until the supervisor acknowledges the last reported image.

    Returns ``(ack_ok, cancel_requested, shutdown_requested)``. A cancel or
    shutdown command that overtakes the acknowledgement is consumed here and
    applied immediately afterwards. Waiting continues while the parent lives;
    if the parent is gone the worker exits (orphan prevention).
    """

    cancel_requested = False
    shutdown_requested = False
    while True:
        if not _parent_alive():
            sys.exit(0)
        if cmd_conn.poll(_PARENT_POLL_SECONDS):
            msg = cmd_conn.recv()
            if isinstance(msg, CmdImageAck) and msg.run_id == run_id:
                return msg.ok, cancel_requested, shutdown_requested
            if isinstance(msg, CmdCancel):
                cancel_requested = cancel_requested or msg.run_id == run_id
            elif isinstance(msg, CmdShutdown):
                # The supervisor is stopping; never wait for a further ack.
                return False, cancel_requested, True


def _drain_pending_commands(
    cmd_conn: multiprocessing.connection.Connection,  # type: ignore[name-defined]
    run_id: str,
) -> tuple[bool, bool]:
    """Non-blocking drain used at safe boundaries inside a run."""

    cancel_requested = False
    shutdown_requested = False
    while cmd_conn.poll(0):
        msg = cmd_conn.recv()
        if isinstance(msg, CmdCancel):
            cancel_requested = cancel_requested or msg.run_id == run_id
        elif isinstance(msg, CmdShutdown):
            shutdown_requested = True
    return cancel_requested, shutdown_requested


def worker_main(launch: WorkerLaunch, cmd_conn, event_q) -> None:  # type: ignore[no-untyped-def]
    # UUID-based device selection must precede the first torch import, which
    # happens inside z_image.load_pipeline below.
    os.environ["CUDA_VISIBLE_DEVICES"] = launch.gpu_uuid
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    pipeline = None
    try:
        try:
            pipeline, pipeline_class = z_image.load_pipeline(launch.snapshot_path, launch.dtype)
        except (FileNotFoundError, NotADirectoryError) as exc:
            _fault(event_q, ErrorCode.CACHE_INCOMPLETE, "load", exc)
            return
        try:
            device_name, device_uuid, uuid_verified = _verify_device(launch)
        except Exception as exc:  # noqa: BLE001 - reported as a worker fault
            _fault(event_q, ErrorCode.WORKER_ERROR, "device-select", exc)
            return
        event_q.put(
            MsgReady(
                pipeline_class=pipeline_class,
                device_name=device_name,
                device_uuid=device_uuid,
                uuid_verified=uuid_verified,
                versions=z_image.dependency_versions(),
            )
        )

        _serve_commands(pipeline, cmd_conn, event_q)
    finally:
        _release(pipeline, event_q)


def _serve_commands(pipeline: object, cmd_conn, event_q) -> None:  # type: ignore[no-untyped-def]
    shutdown_requested = False
    while not shutdown_requested:
        if not _parent_alive():
            return
        if not cmd_conn.poll(_PARENT_POLL_SECONDS):
            continue
        msg = cmd_conn.recv()
        if isinstance(msg, CmdShutdown):
            break
        if isinstance(msg, CmdCancel):
            # No active run; a stale cancel needs no action.
            continue
        if isinstance(msg, CmdRun):
            shutdown_requested = _execute_run(pipeline, msg.task, cmd_conn, event_q)


def _execute_run(pipeline: object, task: RunTask, cmd_conn, event_q) -> bool:  # type: ignore[no-untyped-def]
    """Run one task to a terminal report; returns True when shutdown is due."""

    cancel_requested = False
    completed = 0
    try:
        for index, (seed, artifact_id) in enumerate(
            zip(task.seeds, task.artifact_ids, strict=True), start=1
        ):
            cancelled, shutdown_requested, png = _generate_one_image(
                pipeline, task, index, seed, cmd_conn, event_q
            )
            if cancelled or shutdown_requested:
                break
            completed = index

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
            if shutdown_requested or not ack_ok:
                status = "cancelled" if shutdown_requested else "aborted"
                event_q.put(
                    MsgRunFinished(
                        run_id=task.run_id,
                        status=status,
                        completed_count=completed,
                    )
                )
                return shutdown_requested
            if cancel_requested:
                break
    except Exception as exc:  # noqa: BLE001 - reported as a worker fault
        _fault(event_q, ErrorCode.WORKER_ERROR, "generate", exc, run_id=task.run_id)
        return True

    if shutdown_requested:
        event_q.put(
            MsgRunFinished(run_id=task.run_id, status="cancelled", completed_count=completed)
        )
        return True
    status = "cancelled" if cancel_requested else "completed"
    event_q.put(MsgRunFinished(run_id=task.run_id, status=status, completed_count=completed))
    return False


def _generate_one_image(
    pipeline: object,
    task: RunTask,
    index: int,
    seed: int,
    cmd_conn,
    event_q,  # type: ignore[no-untyped-def]
) -> tuple[bool, bool, bytes]:
    """Generate one image, reporting per-step progress.

    Returns ``(cancel_requested, shutdown_requested, png_bytes)``; the
    flags reflect requests honored at safe inference boundaries. The
    per-step callback is the only place commands are heard mid-image; a
    cancel or shutdown request raises :class:`GenerationCancelled` out of
    the pipeline call.
    """

    from image_studio.inference.z_image import GenerationCancelled

    def on_step(step_index: int) -> None:
        cancel_requested, shutdown_requested = _drain_pending_commands(cmd_conn, task.run_id)
        if shutdown_requested:
            raise GenerationCancelled
        event_q.put(
            MsgProgress(
                run_id=task.run_id,
                image_index=index,
                step=step_index + 1,
                total_steps=task.steps,
            )
        )
        if cancel_requested:
            raise GenerationCancelled

    try:
        png = z_image.generate_image(
            pipeline,
            prompt=task.prompt,
            negative_prompt=task.negative_prompt,
            width=task.width,
            height=task.height,
            steps=task.steps,
            guidance=task.guidance,
            seed=seed,
            on_step=on_step,
        )
    except GenerationCancelled:
        cancel_requested, shutdown_requested = _drain_pending_commands(cmd_conn, task.run_id)
        return True, shutdown_requested, b""
    return False, False, png


def _release(pipeline: object, event_q) -> None:  # type: ignore[no-untyped-def]
    """Release the model and CUDA context on the way out."""

    try:
        if pipeline is not None:
            del pipeline
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup during exit
            pass
