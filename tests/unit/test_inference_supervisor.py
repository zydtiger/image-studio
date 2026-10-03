"""Supervisor semantics with real spawn processes running the fake worker.

These tests exercise the actual cross-process protocol — spawn, pipes,
event queue, acknowledgement backpressure, join-confirmed replacement —
against the CPU-only fake worker. No torch, no CUDA, no network.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

import pytest

from image_studio.inference import InferenceSupervisor
from image_studio.inference.fake import fake_worker_main
from image_studio.inference.protocol import FakeScript
from image_studio.schemas import (
    ErrorCode,
    ErrorInfo,
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    GpuInfo,
    ImageStudioError,
    ProfileId,
    RuntimeConflictError,
    WorkerState,
    plan_artifact_ids,
    utc_now,
)

GPU_A = GpuInfo(uuid="GPU-aaa", name="GPU A", index=0)
GPU_B = GpuInfo(uuid="GPU-bbb", name="GPU B", index=1)


def make_spec(
    run_id="run-1",
    *,
    repo="repo/a",
    commit="c1",
    profile=ProfileId.Z_IMAGE_TURBO,
    dtype="bfloat16",
    gpu=GPU_A,
    count=2,
    steps=3,
    seed=10,
) -> FrozenRunSpec:
    return FrozenRunSpec(
        run_id=run_id,
        created_at=utc_now(),
        model=FrozenModel(
            registration_id="reg-1",
            repo_id=repo,
            commit_sha=commit,
            profile=profile,
            dtype=dtype,
            snapshot_path="/nonexistent/snapshot",
        ),
        gpu=FrozenGpu(uuid=gpu.uuid, name=gpu.name),
        prompt="a cat",
        negative_prompt=None,
        width=256,
        height=256,
        steps=steps,
        guidance=0.0,
        image_count=count,
        seeds=tuple(seed + offset for offset in range(count)),
        artifact_ids=plan_artifact_ids(count),
    )


class RecordingSink:
    """Thread-safe event recorder with scripted per-event failures."""

    def __init__(self, fail_on=None):
        self.events = []
        self.terminals: dict[str, object] = {}
        self._lock = threading.Lock()
        self._fail_on = fail_on

    def on_event(self, event) -> None:
        with self._lock:
            self.events.append(event)
        if self._fail_on is not None and self._fail_on(event):
            raise RuntimeError(f"scripted sink failure on {event.event}")
        if event.event in ("run_completed", "run_failed", "run_cancelled"):
            with self._lock:
                self.terminals[event.run_id] = event

    def wait_terminal(self, run_id, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if run_id in self.terminals:
                    return self.terminals[run_id]
            time.sleep(0.01)
        with self._lock:
            names = [event.event for event in self.events]
        raise AssertionError(f"no terminal for {run_id}; saw events {names}")

    def of(self, *event_names):
        with self._lock:
            return [e for e in self.events if e.event in event_names]

    @property
    def names(self):
        with self._lock:
            return [event.event for event in self.events]


def make_runtime(
    sink,
    *,
    script=None,
    gpus=(GPU_A, GPU_B),
    **kwargs,
) -> InferenceSupervisor:
    runtime = InferenceSupervisor(
        gpu_provider=lambda: list(gpus),
        spawn_target=fake_worker_main,
        fake_script=script if script is not None else FakeScript(load_delay=0.02, step_delay=0.004),
        poll_interval=0.02,
        stop_grace=3.0,
        join_grace=2.0,
        action_timeout=30.0,
        **kwargs,
    )
    runtime.attach(sink)
    return runtime


def wait_for_state(runtime, state, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.status().state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"worker never reached {state}; now {runtime.status().state}")


def assert_pid_dead(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"pid {pid} is still alive (orphaned worker)")


def read_trace(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Event order, residency, and basic lifecycle
# ---------------------------------------------------------------------------


def test_basic_run_event_order_and_resident_status(tmp_path):
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=2, steps=3, seed=10))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_completed"
        assert terminal.completed_count == 2

        names = sink.names
        assert names[0] == "worker_state_changed"
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states[0] == WorkerState.LOADING
        assert states[1] == WorkerState.GENERATING, "a generating state must precede run_started"
        assert "run_started" in names

        images = sink.of("image_completed")
        assert [(image.artifact_id, image.seed) for image in images] == [
            ("image-001", 10),
            ("image-002", 11),
        ]
        assert all(image.png[:8] == b"\x89PNG\r\n\x1a\n" for image in images)

        progress = sink.of("run_progress")
        assert [(p.image_index, p.step, p.total_steps) for p in progress][:3] == [
            (1, 1, 3),
            (1, 2, 3),
            (1, 3, 3),
        ]

        status = runtime.status()
        assert status.implementation == "real"
        assert status.state == WorkerState.IDLE
        assert status.resident is not None
        assert status.resident.gpu == FrozenGpu(uuid=GPU_A.uuid, name=GPU_A.name)
        assert status.resident.repo_id == "repo/a"
        assert status.current_run_id is None
        assert status.queue_depth == 0
        assert status.last_error is None
    finally:
        runtime.shutdown()


def test_fifo_order_and_same_identity_reuse(tmp_path):
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        for index in range(3):
            runtime.submit(make_spec(f"run-{index}", count=1, steps=2))
        for index in range(3):
            sink.wait_terminal(f"run-{index}")

        terminals = [e for e in sink.events if e.event in TERMINAL_EVENTS]
        assert [event.run_id for event in terminals] == ["run-0", "run-1", "run-2"]

        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 1, "identical runs must reuse"
        assert states.count(WorkerState.SWITCHING) == 0
        assert states.count(WorkerState.GENERATING) == 3
    finally:
        runtime.shutdown()


TERMINAL_EVENTS = ("run_completed", "run_failed", "run_cancelled")


def test_replacement_finishes_and_exits_before_new_load(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.004, trace_path=trace)
    sink = RecordingSink()
    runtime = make_runtime(sink, script=script)
    try:
        runtime.submit(make_spec("run-a", repo="repo/a", count=1, steps=2))
        sink.wait_terminal("run-a")
        runtime.submit(make_spec("run-b", repo="repo/b", count=1, steps=2))
        sink.wait_terminal("run-b")

        switching = [e for e in sink.of("worker_state_changed") if e.state == WorkerState.SWITCHING]
        assert len(switching) == 1
        assert switching[0].reason == "model-changed"

        records = read_trace(trace)
        by_pid: dict[int, dict[str, int]] = {}
        for record in records:
            by_pid.setdefault(record["pid"], {})[record["event"]] = record["t_ns"]
        assert len(by_pid) == 2, "a replacement must spawn a second process"
        ordered = sorted(by_pid.items(), key=lambda item: item[1]["load_start"])
        old_pid, new_pid = ordered[0][0], ordered[1][0]
        assert by_pid[old_pid]["worker_exit"] <= by_pid[new_pid]["load_start"], (
            "the old worker must fully exit before the replacement loads"
        )
        assert_pid_dead(old_pid)
    finally:
        runtime.shutdown()


def test_gpu_change_is_a_full_replacement():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-a", gpu=GPU_A, count=1, steps=2))
        sink.wait_terminal("run-a")
        runtime.submit(make_spec("run-b", gpu=GPU_B, count=1, steps=2))
        sink.wait_terminal("run-b")
        switching = [e for e in sink.of("worker_state_changed") if e.state == WorkerState.SWITCHING]
        assert len(switching) == 1
        assert switching[0].reason == "gpu-changed"
        assert runtime.status().resident.gpu.uuid == GPU_B.uuid
    finally:
        runtime.shutdown()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_running_keeps_worker_resident():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=3, steps=60, seed=0))
        deadline = time.monotonic() + 15.0
        while not sink.of("image_completed"):
            assert time.monotonic() < deadline, "first image never completed"
            time.sleep(0.005)
        outcome = runtime.cancel("run-1")
        assert outcome.outcome.value == "cancelling"
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_cancelled"
        assert terminal.completed_count >= 1

        status = runtime.status()
        assert status.state == WorkerState.IDLE
        assert status.resident is not None, "healthy worker stays resident after cancel"

        runtime.submit(make_spec("run-2", count=1, steps=2))
        sink.wait_terminal("run-2")
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 1, "cancelled worker is reused"
    finally:
        runtime.shutdown()


def test_cancel_queued_run_removes_without_running():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1, steps=400))
        runtime.submit(make_spec("run-2", count=1, steps=2))
        outcome = runtime.cancel("run-2")
        assert outcome.outcome.value == "removed_from_queue"
        terminal = sink.wait_terminal("run-2")
        assert terminal.event == "run_cancelled"
        assert terminal.completed_count == 0

        runtime.cancel("run-1")
        sink.wait_terminal("run-1")
        image_events = sink.of("image_completed")
        assert image_events == [] or all(event.run_id == "run-1" for event in image_events), (
            "a queued-cancelled run must never generate images"
        )
    finally:
        runtime.shutdown()


def test_cancel_during_load_skips_generation():
    script = FakeScript(load_delay=1.0, step_delay=0.004)
    sink = RecordingSink()
    runtime = make_runtime(sink, script=script)
    try:
        runtime.submit(make_spec("run-1", count=2, steps=2))
        wait_for_state(runtime, WorkerState.LOADING)
        assert runtime.cancel("run-1").outcome.value == "cancelling"
        terminal = sink.wait_terminal("run-1", timeout=30.0)
        assert terminal.event == "run_cancelled"
        assert terminal.completed_count == 0
        assert runtime.status().state == WorkerState.IDLE
        assert runtime.status().resident is not None
    finally:
        runtime.shutdown()


def test_cancel_unknown_run():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        assert runtime.cancel("nope").outcome.value == "already_finished"
    finally:
        runtime.shutdown()


# ---------------------------------------------------------------------------
# Sink failures and backpressure
# ---------------------------------------------------------------------------


def test_sink_rejecting_image_fails_run_and_stops_remaining_images(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.004, trace_path=trace)
    sink = RecordingSink(fail_on=lambda event: event.event == "image_completed")
    runtime = make_runtime(sink, script=script)
    try:
        runtime.submit(make_spec("run-1", count=3, steps=2))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.STORAGE_ERROR
        assert terminal.completed_count == 0
        assert "run_completed" not in sink.names

        # The failure reaches the sink before the worker processes the nack.
        # A follow-up run on the same worker confirms that processing finished.
        runtime.submit(make_spec("run-2", count=1, steps=2))
        sink.wait_terminal("run-2")
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 1

        records = [record for record in read_trace(trace) if record.get("run_id") == "run-1"]
        starts = [r for r in records if r["event"] == "image_start"]
        assert [r["index"] for r in starts] == [1], (
            "the worker must not continue to the next image after a nack"
        )
        assert any(r["event"] == "abort_nack" for r in records)
    finally:
        runtime.shutdown()


def test_sink_failure_on_progress_fails_run(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.004, trace_path=trace)
    sink = RecordingSink(fail_on=lambda event: event.event == "run_progress" and event.step == 2)
    runtime = make_runtime(sink, script=script)
    try:
        runtime.submit(make_spec("run-1", count=2, steps=30))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.STORAGE_ERROR
        assert "run_completed" not in sink.names
        wait_for_state(runtime, WorkerState.IDLE)
    finally:
        runtime.shutdown()


def test_sink_failure_on_run_started_fails_run_and_keeps_serving():
    """A run_started sink raise must not wedge the dispatch thread.

    Regression: CmdRun is unsent at that point, so the worker is idle and
    can never answer a cancel with a terminal. The run fails with a storage
    error, the healthy worker stays resident and idle, the claim clears, the
    next queued run completes, and Eject succeeds afterwards.
    """
    sink = RecordingSink(
        fail_on=lambda event: event.event == "run_started" and event.run_id == "run-1"
    )
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.STORAGE_ERROR
        assert terminal.completed_count == 0
        assert not [e for e in sink.of("run_completed") if e.run_id == "run-1"]

        wait_for_state(runtime, WorkerState.IDLE)
        status = runtime.status()
        assert status.current_run_id is None
        assert status.resident is not None  # healthy worker stays loaded

        runtime.submit(make_spec("run-2", count=1))
        assert sink.wait_terminal("run-2").event == "run_completed"

        runtime.eject()  # idle worker: must succeed, not 409 forever
        assert runtime.status().state == WorkerState.UNLOADED
    finally:
        runtime.shutdown()


def test_sink_failure_on_generating_state_fails_run_and_keeps_serving():
    """A generating-state sink raise follows the same pre-dispatch path."""
    armed = {"generating": True}

    def fail_once(event):
        if (
            event.event == "worker_state_changed"
            and event.state == WorkerState.GENERATING
            and armed["generating"]
        ):
            armed["generating"] = False
            return True
        return False

    sink = RecordingSink(fail_on=fail_once)
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.STORAGE_ERROR

        wait_for_state(runtime, WorkerState.IDLE)
        assert runtime.status().current_run_id is None

        runtime.submit(make_spec("run-2", count=1))
        assert sink.wait_terminal("run-2").event == "run_completed"

        runtime.eject()
        assert runtime.status().state == WorkerState.UNLOADED
    finally:
        runtime.shutdown()


def test_loading_sink_failure_recorded_and_run_still_completes():
    """Worker-state notifications are non-run events: record and continue.

    Regression: the loading-state sink raise used to be silently discarded.
    It must never fail the run. The recording is observable while it happens
    (see the probe test below); a subsequent healthy load clears last_error
    per the existing rule, so the completed run sees no lingering error.
    """
    sink = RecordingSink(
        fail_on=lambda event: (
            event.event == "worker_state_changed" and event.state == WorkerState.LOADING
        )
    )
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1))
        assert sink.wait_terminal("run-1").event == "run_completed"
        status = runtime.status()
        assert status.state == WorkerState.IDLE
        assert status.last_error is None  # cleared by the healthy load
    finally:
        runtime.shutdown()


def test_state_notification_sink_failure_recorded_in_last_error():
    """A failed worker-state delivery surfaces in status.last_error."""
    sink = RecordingSink(
        fail_on=lambda event: (
            event.event == "worker_state_changed" and event.state == WorkerState.EJECTING
        )
    )
    runtime = make_runtime(sink)
    try:
        exc = runtime._emit_state(WorkerState.EJECTING, reason="probe")
        assert exc is not None
        status = runtime.status()
        assert status.state == WorkerState.EJECTING
        assert status.last_error is not None
        assert status.last_error.code == ErrorCode.STORAGE_ERROR
        assert "ejecting" in status.last_error.message
    finally:
        runtime.shutdown()


def test_idle_worker_exit_observed_and_next_run_spawns_fresh():
    """An idle worker dying externally must not leave phantom residency.

    Regression: the dispatch loop never observed an idle child exit, so
    status kept reporting a cached resident worker and the next same-model
    run took the reuse path onto a dead process and failed. The dispatch
    owner must reap only its own child, clear residency with a visible
    worker_error and an unloaded event, and serve the next run on a fresh
    worker. This is liveness observation, never an idle timeout.
    """
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1))
        assert sink.wait_terminal("run-1").event == "run_completed"
        wait_for_state(runtime, WorkerState.IDLE)

        handle = runtime._worker
        assert handle is not None
        os.kill(handle.process.pid, signal.SIGKILL)  # only our own child
        handle.process.join(timeout=5)

        wait_for_state(runtime, WorkerState.UNLOADED)
        status = runtime.status()
        assert status.resident is None
        assert status.last_error is not None
        assert status.last_error.code == ErrorCode.WORKER_ERROR
        assert "idle" in status.last_error.message
        exited = [
            e
            for e in sink.of("worker_state_changed")
            if e.state == WorkerState.UNLOADED and e.reason == "idle-exit"
        ]
        assert len(exited) == 1
        assert exited[0].error.code == ErrorCode.WORKER_ERROR

        # No phantom failure: the next same-model request spawns fresh.
        runtime.submit(make_spec("run-2", count=1))
        assert sink.wait_terminal("run-2").event == "run_completed"
        assert set(sink.terminals) == {"run-1", "run-2"}
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 2
    finally:
        runtime.shutdown()
        # shutdown after recovery still leaves no live worker of ours
        assert runtime._worker is None


# ---------------------------------------------------------------------------
# Faults and crashes
# ---------------------------------------------------------------------------


def test_worker_crash_mid_run_fails_run_and_recovers():
    sink = RecordingSink()
    runtime = make_runtime(
        sink, script=FakeScript(load_delay=0.02, step_delay=0.004, crash_during=1)
    )
    try:
        runtime.submit(make_spec("run-1", count=2, steps=2))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.WORKER_ERROR
        assert terminal.completed_count == 0

        status = runtime.status()
        assert status.state == WorkerState.UNLOADED
        assert status.resident is None
        assert status.last_error is not None
        assert status.last_error.code == ErrorCode.WORKER_ERROR

        runtime.submit(make_spec("run-2", count=1, steps=2))
        sink.wait_terminal("run-2")
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 2, "a fresh worker must spawn"
    finally:
        runtime.shutdown()


def test_scripted_oom_fault_mid_run():
    sink = RecordingSink()
    runtime = make_runtime(
        sink, script=FakeScript(load_delay=0.02, step_delay=0.004, fault_during=2)
    )
    try:
        runtime.submit(make_spec("run-1", count=3, steps=2))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.WORKER_ERROR
        assert terminal.completed_count == 1, "completed images are preserved"
        assert runtime.status().state == WorkerState.UNLOADED
        assert runtime.status().last_error is not None
    finally:
        runtime.shutdown()


def test_load_failure_fails_run():
    sink = RecordingSink()
    runtime = make_runtime(sink, script=FakeScript(load_delay=0.02, fail_load=True))
    try:
        runtime.submit(make_spec("run-1", count=1, steps=2))
        terminal = sink.wait_terminal("run-1")
        assert terminal.event == "run_failed"
        assert terminal.error.code == ErrorCode.WORKER_ERROR
        assert runtime.status().state == WorkerState.UNLOADED
        assert runtime.status().resident is None
    finally:
        runtime.shutdown()


# ---------------------------------------------------------------------------
# Eject
# ---------------------------------------------------------------------------


def test_eject_idle_unloads_and_busy_conflicts():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1, steps=2))
        sink.wait_terminal("run-1")
        assert runtime.status().state == WorkerState.IDLE

        runtime.submit(make_spec("run-2", count=2, steps=600))
        wait_for_state(runtime, WorkerState.GENERATING)
        with pytest.raises(RuntimeConflictError):
            runtime.eject()
        runtime.cancel("run-2")
        sink.wait_terminal("run-2")
        wait_for_state(runtime, WorkerState.IDLE)

        runtime.eject()
        status = runtime.status()
        assert status.state == WorkerState.UNLOADED
        assert status.resident is None

        runtime.submit(make_spec("run-3", count=1, steps=2))
        sink.wait_terminal("run-3")
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.EJECTING) == 1
        assert states.count(WorkerState.LOADING) == 2, "eject unloads for real"
    finally:
        runtime.shutdown()


def test_eject_during_loading_conflicts():
    sink = RecordingSink()
    runtime = make_runtime(sink, script=FakeScript(load_delay=1.5, step_delay=0.004))
    try:
        runtime.submit(make_spec("run-1", count=1, steps=2))
        wait_for_state(runtime, WorkerState.LOADING)
        with pytest.raises(RuntimeConflictError):
            runtime.eject()
        runtime.cancel("run-1")
        sink.wait_terminal("run-1", timeout=30.0)
    finally:
        runtime.shutdown()


# ---------------------------------------------------------------------------
# Guards and shutdown
# ---------------------------------------------------------------------------


def test_attach_and_submit_guards():
    from image_studio.inference import InferenceSupervisor as Supervisor

    runtime = Supervisor(
        gpu_provider=lambda: [],
        spawn_target=fake_worker_main,
        poll_interval=0.02,
    )
    try:
        with pytest.raises(ImageStudioError):
            runtime.submit(make_spec("run-1"))
        sink = RecordingSink()
        runtime.attach(sink)
        with pytest.raises(ImageStudioError):
            runtime.attach(RecordingSink())

        spec = make_spec("run-bad")
        object.__setattr__(spec, "seeds", (1,))
        with pytest.raises(ImageStudioError):
            runtime.submit(spec)
        assert runtime.status().queue_depth == 0
    finally:
        runtime.shutdown()


def test_shutdown_mid_run_leaves_no_terminal_and_no_orphans(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.01, trace_path=trace)
    sink = RecordingSink()
    runtime = make_runtime(sink, script=script)
    runtime.submit(make_spec("run-1", count=4, steps=400))
    wait_for_state(runtime, WorkerState.GENERATING)
    runtime.shutdown()

    status = runtime.status()
    assert status.state == WorkerState.UNLOADED
    assert "run-1" not in sink.terminals, "shutdown leaves the run non-terminal"

    pids = {record["pid"] for record in read_trace(trace)}
    for pid in pids:
        assert_pid_dead(pid)

    runtime.shutdown()  # idempotent
    with pytest.raises(ImageStudioError):
        runtime.submit(make_spec("run-2"))


def test_fault_after_terminal_does_not_double_emit():
    from image_studio.inference.protocol import MsgFault

    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        error = ErrorInfo(code=ErrorCode.WORKER_ERROR, message="late fault")
        # A run whose terminal was already delivered (storage failure path);
        # a late worker fault must clean up without a second terminal.
        runtime._worker_fault(
            "run-terminal-first",
            MsgFault(error=error, stage="generate"),
            1,
            emit_terminal=False,
        )
        # A fault for a run without a terminal still emits exactly one.
        runtime._worker_fault("run-fresh", MsgFault(error=error, stage="generate"), 0)

        assert "run-terminal-first" not in sink.terminals
        terminal = sink.wait_terminal("run-fresh")
        assert terminal.event == "run_failed"
        unloaded = [e for e in sink.of("worker_state_changed") if e.state == WorkerState.UNLOADED]
        assert len(unloaded) == 2
        assert runtime.status().state == WorkerState.UNLOADED
    finally:
        runtime.shutdown()


def test_no_torch_imported_after_fake_runs():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=1, steps=2))
        sink.wait_terminal("run-1")
    finally:
        runtime.shutdown()
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules
