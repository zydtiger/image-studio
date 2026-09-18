"""Runtime lifecycle integration: serialized replacements, crash recovery,
restart semantics, and eject/queue interplay — with real spawn processes
running the CPU-only fake worker.
"""

from __future__ import annotations

import json
import threading
import time

from image_studio.inference import InferenceSupervisor
from image_studio.inference.fake import fake_worker_main
from image_studio.inference.protocol import FakeScript
from image_studio.schemas import (
    ErrorCode,
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    GpuInfo,
    ProfileId,
    WorkerState,
    plan_artifact_ids,
    utc_now,
)

GPU_A = GpuInfo(uuid="GPU-aaa", name="GPU A", index=0)
TERMINAL_EVENTS = ("run_completed", "run_failed", "run_cancelled")


def make_spec(
    run_id,
    *,
    repo="repo/a",
    gpu=GPU_A,
    count=1,
    steps=2,
    seed=5,
) -> FrozenRunSpec:
    return FrozenRunSpec(
        run_id=run_id,
        created_at=utc_now(),
        model=FrozenModel(
            registration_id="reg-1",
            repo_id=repo,
            commit_sha="c1",
            profile=ProfileId.Z_IMAGE_TURBO,
            dtype="bfloat16",
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
    def __init__(self):
        self.events = []
        self.terminals = {}

    def on_event(self, event) -> None:
        self.events.append(event)
        if event.event in TERMINAL_EVENTS:
            self.terminals[event.run_id] = event

    def wait_terminal(self, run_id, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if run_id in self.terminals:
                return self.terminals[run_id]
            time.sleep(0.01)
        raise AssertionError(f"no terminal for {run_id}; saw {[e.event for e in self.events]}")

    def of(self, *names):
        return [e for e in self.events if e.event in names]


def make_runtime(sink, script=None):
    runtime = InferenceSupervisor(
        gpu_provider=lambda: [GPU_A],
        spawn_target=fake_worker_main,
        fake_script=script if script is not None else FakeScript(load_delay=0.02, step_delay=0.004),
        poll_interval=0.02,
        stop_grace=3.0,
        join_grace=2.0,
        action_timeout=60.0,
    )
    runtime.attach(sink)
    return runtime


def test_alternating_models_serialize_replacements(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.004, trace_path=trace)
    sink = RecordingSink()
    runtime = make_runtime(sink, script)
    try:
        plan = [("run-1", "repo/a"), ("run-2", "repo/b"), ("run-3", "repo/a"), ("run-4", "repo/b")]
        for run_id, repo in plan:
            runtime.submit(make_spec(run_id, repo=repo))
        for run_id, _repo in plan:
            terminal = sink.wait_terminal(run_id)
            assert terminal.event == "run_completed"

        started = [e.run_id for e in sink.of("run_started")]
        assert started == [run_id for run_id, _ in plan], "strict FIFO dispatch"

        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 4
        assert states.count(WorkerState.SWITCHING) == 3
        assert states.count(WorkerState.GENERATING) == 4

        records = [json.loads(line) for line in open(trace, encoding="utf-8")]
        loads = sorted((r["t_ns"], r["pid"]) for r in records if r["event"] == "load_start")
        exits = sorted((r["t_ns"], r["pid"]) for r in records if r["event"] == "worker_exit")
        assert len(loads) == 4
        # Every exit (except the live one) must precede the next load.
        for index in range(len(exits) - 1):
            assert exits[index][0] <= loads[index + 1][0], "workers must never overlap"
        live_pids = {pid for _, pid in loads} - {pid for _, pid in exits}
        assert len(live_pids) == 1, "exactly one worker stays resident"
    finally:
        runtime.shutdown()


def test_crash_mid_queue_fails_one_run_and_recovers_next(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.004, crash_on_run=2, trace_path=trace)
    sink = RecordingSink()
    runtime = make_runtime(sink, script)
    try:
        for run_id in ("run-1", "run-2", "run-3"):
            runtime.submit(make_spec(run_id))
        terminal_1 = sink.wait_terminal("run-1")
        terminal_2 = sink.wait_terminal("run-2")
        terminal_3 = sink.wait_terminal("run-3")
        assert terminal_1.event == "run_completed"
        assert terminal_2.event == "run_failed"
        assert terminal_2.error.code == ErrorCode.WORKER_ERROR
        assert terminal_3.event == "run_completed"

        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 2
        assert runtime.status().state == WorkerState.IDLE
        # The crash transition itself carries the error; a subsequent
        # healthy load clears last_error by design.
        unloaded = [
            e
            for e in sink.of("worker_state_changed")
            if e.state == WorkerState.UNLOADED and e.error is not None
        ]
        assert len(unloaded) == 1
        assert unloaded[0].error.code == ErrorCode.WORKER_ERROR
        assert unloaded[0].reason == "crashed"
    finally:
        runtime.shutdown()


def test_recreated_runtime_starts_unloaded_and_preloads_nothing():
    sink1 = RecordingSink()
    runtime1 = make_runtime(sink1)
    runtime1.submit(make_spec("run-1"))
    sink1.wait_terminal("run-1")
    runtime1.shutdown()

    sink2 = RecordingSink()
    runtime2 = make_runtime(sink2)
    try:
        status = runtime2.status()
        assert status.state == WorkerState.UNLOADED
        assert status.resident is None
        assert status.current_run_id is None
        assert status.queue_depth == 0

        runtime2.submit(make_spec("run-2"))
        terminal = sink2.wait_terminal("run-2")
        assert terminal.event == "run_completed"
        states = [e.state for e in sink2.of("worker_state_changed")]
        assert states[0] == WorkerState.LOADING, "no preload before work exists"
    finally:
        runtime2.shutdown()


def test_eject_succeeds_with_queued_work_and_queue_continues(tmp_path):
    """Idle Eject wins dispatch arbitration while the queue is non-empty.

    Deterministic by construction: the dispatch thread is held inside the
    idle-notification delivery after run-1 (gated sink; the supervisor sets
    its state to idle before emitting, so the guarded state is already
    IDLE while the sink blocks). run-2 sits queued in that window. The
    eject request is registered there, arbitration honors it before
    claiming run-2, the healthy worker is stopped, and run-2 is then served
    by a freshly spawned worker. No timing sleeps.
    """
    trace = str(tmp_path / "trace.jsonl")
    script = FakeScript(load_delay=0.02, step_delay=0.004, trace_path=trace)
    gate = threading.Event()
    held = {"armed": True}

    class GatedSink(RecordingSink):
        def on_event(self, event) -> None:
            super().on_event(event)
            if (
                held["armed"]
                and event.event == "worker_state_changed"
                and event.state == WorkerState.IDLE
            ):
                held["armed"] = False
                assert gate.wait(timeout=10), "test never released the gate"

    sink = GatedSink()
    runtime = make_runtime(sink, script)
    try:
        runtime.submit(make_spec("run-1"))
        runtime.submit(make_spec("run-2"))
        deadline = time.monotonic() + 10
        while held["armed"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not held["armed"], "run-1 idle notification never reached the sink"
        # The dispatch thread is blocked inside the idle delivery, so run-2
        # is deterministically still queued and the guarded state is IDLE.
        status = runtime.status()
        assert status.state == WorkerState.IDLE
        assert status.queue_depth == 1

        eject_outcome: dict[str, object] = {}

        def do_eject() -> None:
            try:
                runtime.eject()
            except Exception as exc:  # noqa: BLE001 - recorded for assertion
                eject_outcome["error"] = exc

        eject_thread = threading.Thread(target=do_eject)
        eject_thread.start()
        while runtime._eject_future is None and eject_thread.is_alive():
            time.sleep(0.005)
        gate.set()
        eject_thread.join(timeout=10)
        assert "error" not in eject_outcome, eject_outcome.get("error")

        terminal = sink.wait_terminal("run-2")
        assert terminal.event == "run_completed"
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.EJECTING) == 1
        assert states.count(WorkerState.LOADING) == 2

        records = [json.loads(line) for line in open(trace, encoding="utf-8")]
        loads = [(r["t_ns"], r["pid"]) for r in records if r["event"] == "load_start"]
        eject_exit = [
            (r["t_ns"], r["pid"])
            for r in records
            if r["event"] == "shutdown" and r.get("reason") == "eject"
        ]
        assert len(loads) == 2 and loads[0][1] != loads[1][1], "run-2 served by a fresh worker"
        assert len(eject_exit) == 1
        assert eject_exit[0][0] <= loads[1][0], "old worker exited before its replacement"
    finally:
        gate.set()
        runtime.shutdown()


def wait_state(runtime, state, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.status().state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"never reached {state}; now {runtime.status().state}")


def wait_idle(runtime):
    wait_state(runtime, WorkerState.IDLE)
