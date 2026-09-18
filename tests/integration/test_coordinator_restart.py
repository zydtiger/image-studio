"""Integration tests for the coordinator: restart, storage failure, queue pause."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from image_studio.storage.artifacts import ArtifactStore
from image_studio.testing import FakeHub, FakeRuntime
from tests.conftest import build_harness


def _wait_status(harness, run_id: str, status: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    detail = {}
    while time.monotonic() < deadline:
        detail = harness.client.get(f"/api/generations/{run_id}").json()
        if detail["status"] == status:
            return detail
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached {status}: {detail}")


def test_restart_marks_interrupted_and_paused_then_resumes_fifo(
    xdg_env: dict[str, str], tmp_path: Path
) -> None:
    hub = FakeHub(tmp_path / "hub")
    # the first dispatched run blocks after its start event; later runs stay queued
    gated = FakeRuntime(gate_after_start=True)
    harness_a = build_harness(xdg_env, gated, hub)
    with harness_a.client:
        registration = harness_a.register_model()
        active = harness_a.submit(registration["id"], count=2, prompt="active")
        _wait_status(harness_a, active["run_id"], "running")
        first_paused = harness_a.submit(registration["id"], count=1, prompt="paused-1")
        second_paused = harness_a.submit(registration["id"], count=1, prompt="paused-2")

    runtime_b = FakeRuntime()
    harness_b = build_harness(xdg_env, runtime_b, hub)
    with harness_b.client:
        interrupted = harness_b.client.get(f"/api/generations/{active['run_id']}").json()
        assert interrupted["status"] == "interrupted"
        assert interrupted["error"]["code"] == "internal"

        queue = harness_b.client.get("/api/generations/queue").json()
        assert queue["paused"] is True
        assert [run["prompt"] for run in queue["pending"]] == ["paused-1", "paused-2"]

        resumed = harness_b.client.post("/api/generations/queue/resume").json()
        assert resumed["paused"] is False
        for run in (first_paused, second_paused):
            finished = harness_b.wait_terminal(run["run_id"])
            assert finished["status"] == "completed"
        # resumed runs keep their original frozen settings
        detail = harness_b.client.get(f"/api/generations/{first_paused['run_id']}").json()
        assert detail["initial_seed"] == first_paused["initial_seed"]
        assert detail["repo_id"] == "Tongyi-MAI/Z-Image"


def test_cancel_paused_run_after_restart(xdg_env: dict[str, str], tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    harness_a = build_harness(xdg_env, FakeRuntime(gate_after_start=True), hub)
    with harness_a.client:
        registration = harness_a.register_model()
        active = harness_a.submit(registration["id"], count=1)
        _wait_status(harness_a, active["run_id"], "running")
        paused_run = harness_a.submit(registration["id"], count=1)

    harness_b = build_harness(xdg_env, FakeRuntime(), hub)
    with harness_b.client:
        assert (
            harness_b.client.get(f"/api/generations/{paused_run['run_id']}").json()["status"]
            == "paused"
        )
        cancelled = harness_b.client.post(f"/api/generations/{paused_run['run_id']}/cancel")
        assert cancelled.status_code == 202
        assert cancelled.json()["status"] == "cancelled"


def test_storage_failure_fails_run_not_success(
    xdg_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
    with harness.client:
        registration = harness.register_model()

        def failing(self, created_at, run_id, artifact, png):
            raise OSError("disk full")

        # patch before submission so no image can complete durably first
        monkeypatch.setattr(ArtifactStore, "write_image", failing)
        detail = harness.submit(registration["id"], count=2)
        finished = harness.wait_terminal(detail["run_id"])
        assert finished["status"] == "failed"
        assert finished["error"]["code"] == "storage_error"
        assert all(image["status"] != "completed" for image in finished["images"])
        listing = harness.client.get(f"/api/generations/{detail['run_id']}/artifacts").json()
        assert all(image["status"] != "completed" for image in listing["artifacts"])
        assert harness.client.get("/api/runtime").json()["state"] in ("idle", "generating")


def test_eject_conflict_while_busy(xdg_env: dict[str, str], tmp_path: Path) -> None:
    runtime = FakeRuntime(image_delay=0.5)
    harness = build_harness(xdg_env, runtime, FakeHub(tmp_path / "hub"))
    with harness.client:
        registration = harness.register_model()
        detail = harness.submit(registration["id"], count=4)
        busy = harness.client.post("/api/runtime/eject")
        assert busy.status_code == 409
        assert busy.json()["error"]["code"] == "conflict"

        harness.client.post(f"/api/generations/{detail['run_id']}/cancel")
        harness.wait_terminal(detail["run_id"])
        idle = harness.client.post("/api/runtime/eject")
        assert idle.status_code == 204
        status = harness.client.get("/api/runtime").json()
        assert status["state"] == "unloaded"
        assert status["resident"] is None
        # eject while unloaded is an idempotent no-op, like the real runtime
        assert harness.client.post("/api/runtime/eject").status_code == 204


def test_resident_gpu_visible_after_run_completes(xdg_env: dict[str, str], tmp_path: Path) -> None:
    harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
    with harness.client:
        registration = harness.register_model()
        detail = harness.submit(registration["id"], count=1, gpu_uuid="GPU-fake-0002")
        harness.wait_terminal(detail["run_id"])
        status = harness.client.get("/api/runtime").json()
        assert status["current_run_id"] is None
        assert status["resident"]["gpu"]["uuid"] == "GPU-fake-0002"
