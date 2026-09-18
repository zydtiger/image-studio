"""Concurrency behavior of the runtime under threaded API pressure.

Deterministic FIFO ordering is asserted for submissions made from one
thread, while side threads hammer the non-blocking surface (``status``,
``cancel`` of unknown runs, ``list_gpus``) and busy-state ``eject``
attempts. Multi-threaded submits must all reach exactly one terminal each.
"""

from __future__ import annotations

import threading
import time

from image_studio.inference import InferenceSupervisor
from image_studio.inference.fake import fake_worker_main
from image_studio.inference.protocol import FakeScript
from image_studio.schemas import (
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    GpuInfo,
    ProfileId,
    RuntimeConflictError,
    WorkerState,
    plan_artifact_ids,
    utc_now,
)

GPU_A = GpuInfo(uuid="GPU-aaa", name="GPU A", index=0)
TERMINAL_EVENTS = ("run_completed", "run_failed", "run_cancelled")


def make_spec(run_id, *, repo="repo/a", count=1, steps=2, seed=5):
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
        gpu=FrozenGpu(uuid=GPU_A.uuid, name=GPU_A.name),
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
        self._lock = threading.Lock()

    def on_event(self, event) -> None:
        with self._lock:
            self.events.append(event)
            if event.event in TERMINAL_EVENTS:
                self.terminals[event.run_id] = event

    def wait_terminal(self, run_id, timeout=90.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if run_id in self.terminals:
                    return self.terminals[run_id]
            time.sleep(0.01)
        with self._lock:
            names = [event.event for event in self.events]
        raise AssertionError(f"no terminal for {run_id}; saw {names}")

    def of(self, *names):
        with self._lock:
            return [e for e in self.events if e.event in names]


def make_runtime(sink):
    runtime = InferenceSupervisor(
        gpu_provider=lambda: [GPU_A],
        spawn_target=fake_worker_main,
        fake_script=FakeScript(load_delay=0.02, step_delay=0.004),
        poll_interval=0.02,
        stop_grace=3.0,
        join_grace=2.0,
        action_timeout=60.0,
    )
    runtime.attach(sink)
    return runtime


def wait_state(runtime, state, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.status().state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"never reached {state}; now {runtime.status().state}")


def test_fifo_order_holds_under_status_hammering():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        order = [f"run-{index}" for index in range(8)]
        for index, run_id in enumerate(order):
            # Alternate identities every two runs to force replacements
            # while the queue keeps draining in order.
            repo = "repo/a" if index % 4 < 2 else "repo/b"
            runtime.submit(make_spec(run_id, repo=repo))

        stop = threading.Event()
        errors = []

        def hammer():
            while not stop.is_set():
                try:
                    runtime.status()
                    runtime.cancel("unknown-run")
                    runtime.list_gpus()
                except Exception as exc:  # noqa: BLE001 - record and fail
                    errors.append(exc)
                    return

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        try:
            for run_id in order:
                sink.wait_terminal(run_id)
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=5.0)

        assert not errors, errors
        started = [event.run_id for event in sink.of("run_started")]
        assert started == order, "dispatch order must equal submission order"
        terminals = [event.run_id for event in sink.of(*TERMINAL_EVENTS)]
        assert terminals == order
        assert runtime.status().queue_depth == 0
    finally:
        runtime.shutdown()


def test_busy_eject_conflicts_from_many_threads():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=2, steps=300))
        wait_state(runtime, WorkerState.GENERATING)

        conflicts = []
        barrier = threading.Barrier(6)

        def attempt_eject():
            barrier.wait()
            try:
                runtime.eject()
            except RuntimeConflictError:
                conflicts.append(True)

        threads = [threading.Thread(target=attempt_eject) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        assert len(conflicts) == 6, "every busy eject must be rejected"
        assert runtime.status().state == WorkerState.GENERATING

        runtime.cancel("run-1")
        sink.wait_terminal("run-1")
        wait_state(runtime, WorkerState.IDLE)
        runtime.eject()
        assert runtime.status().state == WorkerState.UNLOADED
    finally:
        runtime.shutdown()


def test_concurrent_submits_all_reach_exactly_one_terminal():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        run_ids = [f"run-{index}" for index in range(12)]
        start = threading.Barrier(4)

        def submit_slice(names):
            start.wait()
            for name in names:
                runtime.submit(make_spec(name))

        groups = [run_ids[index::4] for index in range(4)]
        threads = [threading.Thread(target=submit_slice, args=(group,)) for group in groups]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        for run_id in run_ids:
            terminal = sink.wait_terminal(run_id)
            assert terminal.event == "run_completed"
        terminals = sink.of(*TERMINAL_EVENTS)
        assert len(terminals) == len(run_ids), "exactly one terminal per run"
        assert len({event.run_id for event in terminals}) == len(run_ids)
        started = [event.run_id for event in sink.of("run_started")]
        assert sorted(started) == sorted(run_ids)
        assert runtime.status().queue_depth == 0
    finally:
        runtime.shutdown()


def test_cancel_storm_on_running_and_queued_runs():
    sink = RecordingSink()
    runtime = make_runtime(sink)
    try:
        runtime.submit(make_spec("run-1", count=4, steps=300))
        wait_state(runtime, WorkerState.GENERATING)
        for index in range(2, 6):
            runtime.submit(make_spec(f"run-{index}", steps=2))

        results = {
            run_id: runtime.cancel(f"run-{index}")
            for index, run_id in enumerate(["run-1", "run-2", "run-3", "run-4", "run-5"], start=1)
        }
        assert results["run-1"].outcome.value == "cancelling"
        for run_id in ("run-2", "run-3", "run-4", "run-5"):
            assert results[run_id].outcome.value == "removed_from_queue"

        for run_id in ("run-1", "run-2", "run-3", "run-4", "run-5"):
            terminal = sink.wait_terminal(run_id)
            assert terminal.event == "run_cancelled"
        assert runtime.status().state == WorkerState.IDLE
        assert runtime.status().queue_depth == 0
    finally:
        runtime.shutdown()
