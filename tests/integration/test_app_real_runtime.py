"""App-level integration with the real supervisor and a fake worker process.

The real ``create_runtime`` runs with its CPU-only fake spawn target, so the
full supervisor/worker protocol — spawn, ready handshake, acknowledgement
backpressure, cooperative cancel, replacement — is exercised end to end
through the public HTTP API without torch, weights, or GPUs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from image_studio.inference import create_runtime
from image_studio.inference.fake import fake_worker_main
from image_studio.inference.protocol import FakeScript
from image_studio.storage.artifacts import ArtifactStore
from image_studio.testing import FAKE_GPUS, FakeHub
from tests.conftest import build_harness

WORKER_IDENTITY = {"pipeline_class": "FakeZImagePipeline", "versions": {"fake": "1"}}


def _real_harness(env: dict[str, str], hub: FakeHub, script: FakeScript):
    runtime = create_runtime(
        gpu_provider=lambda: list(FAKE_GPUS),
        spawn_target=fake_worker_main,
        fake_script=script,
        poll_interval=0.05,
        stop_grace=5.0,
        join_grace=5.0,
        action_timeout=30.0,
    )
    return build_harness(env, runtime, hub)


def _wait_status(harness, run_id: str, *statuses: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    detail = {}
    while time.monotonic() < deadline:
        detail = harness.client.get(f"/api/generations/{run_id}").json()
        if detail["status"] in statuses:
            return detail
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach {statuses}: {detail}")


def test_full_generation_flow_records_worker_identity(xdg_env, tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    harness = _real_harness(xdg_env, hub, FakeScript(load_delay=0.05, step_delay=0.0))
    with harness.client:
        registration = harness.register_model()
        detail = harness.submit(registration["id"], count=2, seed=3, steps=1)
        finished = _wait_status(harness, detail["run_id"], "completed")
        assert finished["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]
        assert finished["dependency_versions"] == WORKER_IDENTITY["versions"]
        assert [image["seed"] for image in finished["images"]] == [3, 4]

        runtime = harness.client.get("/api/runtime").json()
        assert runtime["implementation"] == "real"
        assert runtime["resident"]["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]

        metadata = harness.client.get(f"/api/generations/{detail['run_id']}/metadata")
        payload = json.loads(metadata.content)
        assert payload["runtime"]["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]
        assert payload["runtime"]["dependency_versions"] == WORKER_IDENTITY["versions"]


def test_reused_worker_still_reports_identity_per_run(xdg_env, tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    hub = FakeHub(tmp_path / "hub")
    script = FakeScript(load_delay=0.05, step_delay=0.0, trace_path=str(trace))
    harness = _real_harness(xdg_env, hub, script)
    with harness.client:
        registration = harness.register_model()
        first = harness.submit(registration["id"], count=1, steps=1)
        _wait_status(harness, first["run_id"], "completed")
        second = harness.submit(registration["id"], count=1, steps=1)
        _wait_status(harness, second["run_id"], "completed")

        # exactly one worker load for two runs: identity reported per run anyway
        loads = [
            line
            for line in trace.read_text().splitlines()
            if json.loads(line)["event"] == "load_start"
        ]
        assert len(loads) == 1
        for run in (first, second):
            detail = harness.client.get(f"/api/generations/{run['run_id']}").json()
            assert detail["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]
            assert detail["dependency_versions"] == WORKER_IDENTITY["versions"]


def test_queue_cancel_terminal_without_run_started(xdg_env, tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    script = FakeScript(load_delay=0.05, step_delay=0.02)
    harness = _real_harness(xdg_env, hub, script)
    with harness.client:
        registration = harness.register_model()
        blocker = harness.submit(registration["id"], count=3, steps=100)
        _wait_status(harness, blocker["run_id"], "running")
        queued = harness.submit(registration["id"], count=1, steps=1)

        cancelled = harness.client.post(f"/api/generations/{queued['run_id']}/cancel")
        assert cancelled.status_code == 202
        # the supervisor delivers run_cancelled for a run that never started
        terminal = _wait_status(harness, queued["run_id"], "cancelled")
        assert terminal["started_at"] is None
        assert all(image["status"] == "cancelled" for image in terminal["images"])

        harness.client.post(f"/api/generations/{blocker['run_id']}/cancel")
        _wait_status(harness, blocker["run_id"], "cancelled", "partial", timeout=20)
        assert harness.client.get("/api/runtime").json()["resident"] is not None


def test_failed_run_keeps_worker_identity(xdg_env, tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    script = FakeScript(load_delay=0.05, step_delay=0.0, fault_during=2)
    harness = _real_harness(xdg_env, hub, script)
    with harness.client:
        registration = harness.register_model()
        detail = harness.submit(registration["id"], count=3, steps=1)
        finished = _wait_status(harness, detail["run_id"], "partial", "failed")
        assert finished["status"] == "partial"
        assert finished["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]
        assert finished["dependency_versions"] == WORKER_IDENTITY["versions"]
        assert finished["error"]["code"] == "worker_error"
        statuses = {image["artifact_id"]: image["status"] for image in finished["images"]}
        assert statuses["image-001"] == "completed"
        assert statuses["image-002"] == "cancelled"


def test_shutdown_mid_run_reconciles_on_restart(xdg_env, tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    script = FakeScript(load_delay=0.05, step_delay=0.02)
    harness = _real_harness(xdg_env, hub, script)
    with harness.client:
        registration = harness.register_model()
        active = harness.submit(registration["id"], count=3, steps=100)
        _wait_status(harness, active["run_id"], "running")
        # this one stays queued when the server shuts down mid-run
        queued = harness.submit(registration["id"], count=1, steps=1)

    restarted = _real_harness(xdg_env, hub, script)
    with restarted.client:
        interrupted = restarted.client.get(f"/api/generations/{active['run_id']}").json()
        assert interrupted["status"] == "interrupted"

        queue = restarted.client.get("/api/generations/queue").json()
        assert queue["paused"] is True
        assert [run["run_id"] for run in queue["pending"]] == [queued["run_id"]]

        runtime = restarted.client.get("/api/runtime").json()
        assert runtime["state"] == "unloaded"  # no preload after restart

        resumed = restarted.client.post("/api/generations/queue/resume").json()
        assert resumed["paused"] is False
        finished = _wait_status(restarted, queued["run_id"], "completed")
        assert finished["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]


def test_sink_failure_residue_and_frozen_request(
    xdg_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path / "hub")
    harness = _real_harness(xdg_env, hub, FakeScript(load_delay=0.05, step_delay=0.0))
    with harness.client:
        registration = harness.register_model()

        def failing(self, created_at, run_id, artifact, png):
            raise OSError("disk full")

        monkeypatch.setattr(ArtifactStore, "write_image", failing)
        detail = harness.submit(registration["id"], count=2, steps=1, prompt="kept frozen")
        failed = _wait_status(harness, detail["run_id"], "failed", timeout=20)
        monkeypatch.undo()

        # the frozen request survives the storage failure
        assert failed["prompt"] == "kept frozen"
        assert failed["width"] == 256 and failed["height"] == 256
        assert failed["steps"] == 1 and failed["repo_id"] == "Tongyi-MAI/Z-Image"
        assert failed["pipeline_class"] == WORKER_IDENTITY["pipeline_class"]
        # no artifact files were durably written for the failed run
        outputs = Path(harness.client.get("/api/system").json()["paths"]["outputs_dir"])
        assert not list(outputs.rglob(f"{detail['run_id']}/*.png"))

        # the queue keeps serving after a sink failure
        recovered = harness.submit(registration["id"], count=1, steps=1)
        _wait_status(harness, recovered["run_id"], "completed")
