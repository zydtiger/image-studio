"""Paused-queue semantics, FIFO serialization, validators, and metadata parity."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from image_studio.storage.artifacts import ArtifactStore
from image_studio.testing import FakeHub, FakeRuntime
from tests.conftest import AppHarness, build_harness


@pytest.fixture
def paused_harness(xdg_env, tmp_path: Path) -> Iterator[AppHarness]:
    harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
    with harness.client:
        registration = harness.register_model()
        harness.client.app.state.app_state.coordinator.mark_queue_paused()
        harness.registration_id = registration["id"]
        yield harness


class TestPausedQueue:
    def test_submit_rejected_with_409_and_no_phantom_run(self, paused_harness) -> None:
        before = paused_harness.client.get("/api/generations").json()["total"]
        response = paused_harness.client.post(
            "/api/generations",
            json={
                "registration_id": paused_harness.registration_id,
                "gpu_uuid": "GPU-fake-0001",
                "prompt": "while paused",
                "width": 256,
                "height": 256,
            },
        )
        assert response.status_code == 409
        body = response.json()
        assert body["error"]["code"] == "conflict"
        assert "paused" in body["error"]["message"]
        assert paused_harness.client.get("/api/generations").json()["total"] == before

    def test_resume_accepts_again_and_backlog_completes(self, xdg_env, tmp_path: Path) -> None:
        harness = build_harness(
            xdg_env, FakeRuntime(gate_after_start=True), FakeHub(tmp_path / "hub")
        )
        with harness.client:
            registration = harness.register_model()
            active = harness.submit(registration["id"], count=1)
            import time

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (
                    harness.client.get(f"/api/generations/{active['run_id']}").json()["status"]
                    == "running"
                ):
                    break
                time.sleep(0.01)
            paused_run = harness.submit(registration["id"], count=1)  # queued behind gate

        resumed_harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
        with resumed_harness.client:
            assert (
                resumed_harness.client.get(f"/api/generations/{paused_run['run_id']}").json()[
                    "status"
                ]
                == "paused"
            )
            rejected = resumed_harness.client.post(
                "/api/generations",
                json={
                    "registration_id": registration["id"],
                    "gpu_uuid": "GPU-fake-0001",
                    "prompt": "still paused",
                    "width": 256,
                    "height": 256,
                },
            )
            assert rejected.status_code == 409

            resumed_harness.client.post("/api/generations/queue/resume")
            finished = resumed_harness.wait_terminal(paused_run["run_id"])
            assert finished["status"] == "completed"

            after = resumed_harness.client.post(
                "/api/generations",
                json={
                    "registration_id": registration["id"],
                    "gpu_uuid": "GPU-fake-0001",
                    "prompt": "after resume",
                    "width": 256,
                    "height": 256,
                },
            )
            assert after.status_code == 202
            assert resumed_harness.wait_terminal(after.json()["run_id"])["status"] == "completed"


class TestConcurrentSubmitOrder:
    def test_queue_seq_matches_dispatch_order(self, paused_harness) -> None:
        """Concurrent accepted submissions persist and dispatch in the same order."""
        runtime = paused_harness.runtime
        dispatch_order: list[str] = []
        original_submit = runtime.submit

        def recording_submit(spec):
            dispatch_order.append(spec.run_id)
            original_submit(spec)

        runtime.submit = recording_submit  # type: ignore[method-assign]
        paused_harness.client.post("/api/generations/queue/resume")

        accepted: list[str] = []
        errors: list[object] = []
        guard = threading.Lock()

        def worker() -> None:
            response = paused_harness.client.post(
                "/api/generations",
                json={
                    "registration_id": paused_harness.registration_id,
                    "gpu_uuid": "GPU-fake-0001",
                    "prompt": "concurrent",
                    "width": 256,
                    "height": 256,
                },
            )
            with guard:
                (accepted if response.status_code == 202 else errors).append(response.json())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        assert len(accepted) == 8

        # listing is ordered by queue_seq DESC; reverse for ascending seq
        listing = paused_harness.client.get("/api/generations", params={"limit": 200}).json()
        assert listing["total"] == 8
        seq_ascending = [run["run_id"] for run in reversed(listing["runs"])]
        assert sorted(dispatch_order) == sorted(seq_ascending)
        assert dispatch_order == seq_ascending, "dispatch order must match queue_seq order"


class TestHistoryValidators:
    def test_invalid_limit_and_trashed_rejected(self, harness) -> None:
        for params in ({"limit": -1}, {"limit": 0}, {"limit": 201}, {"trashed": "invalid"}):
            response = harness.client.get("/api/generations", params=params)
            assert response.status_code == 422, params
            assert response.json()["error"]["code"] == "validation"

    def test_valid_bounds_accepted(self, harness) -> None:
        response = harness.client.get("/api/generations", params={"limit": 1, "offset": 0})
        assert response.status_code == 200


class TestMetadataParity:
    def _metadata(self, harness, run_id: str) -> dict:
        response = harness.client.get(f"/api/generations/{run_id}/metadata")
        assert response.status_code == 200
        return json.loads(response.content)

    def test_completed_metadata_final(self, harness) -> None:
        registration = harness.register_model()
        detail = harness.submit(registration["id"], count=2, seed=9)
        finished = harness.wait_terminal(detail["run_id"])
        assert finished["status"] == "completed"
        payload = self._metadata(harness, detail["run_id"])
        assert payload["status"] == "completed"
        assert [image["seed"] for image in payload["images"]] == [9, 10]
        assert payload["runtime"]["pipeline_class"] == "FakeZImagePipeline"

    def test_partial_metadata_final(self, harness) -> None:
        harness.runtime.fail_image_at = 2
        registration = harness.register_model()
        detail = harness.submit(registration["id"], count=3)
        finished = harness.wait_terminal(detail["run_id"])
        assert finished["status"] == "partial"
        payload = self._metadata(harness, detail["run_id"])
        assert payload["status"] == "partial"
        statuses = {image["artifact_id"]: image["status"] for image in payload["images"]}
        assert statuses["image-001"] == "completed"
        assert statuses["image-003"] == "cancelled"

    def test_zero_image_cancelled_metadata_exists(self, harness) -> None:
        registration = harness.register_model()
        harness.runtime.image_delay = 0.05
        first = harness.submit(registration["id"], count=4)
        queued = harness.submit(registration["id"], count=1)
        harness.client.post(f"/api/generations/{queued['run_id']}/cancel")
        terminal = harness.wait_terminal(queued["run_id"])
        assert terminal["status"] == "cancelled"
        assert terminal["started_at"] is None
        payload = self._metadata(harness, queued["run_id"])
        assert payload["status"] == "cancelled"
        assert all(image["status"] == "cancelled" for image in payload["images"])
        harness.client.post(f"/api/generations/{first['run_id']}/cancel")
        harness.wait_terminal(first["run_id"])

    def test_interrupted_metadata_after_restart(self, xdg_env, tmp_path: Path) -> None:
        harness_a = build_harness(
            xdg_env, FakeRuntime(gate_after_start=True), FakeHub(tmp_path / "hub")
        )
        with harness_a.client:
            registration = harness_a.register_model()
            active = harness_a.submit(registration["id"], count=1)
            import time

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (
                    harness_a.client.get(f"/api/generations/{active['run_id']}").json()["status"]
                    == "running"
                ):
                    break
                time.sleep(0.01)

        harness_b = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
        with harness_b.client:
            detail = harness_b.client.get(f"/api/generations/{active['run_id']}").json()
            assert detail["status"] == "interrupted"
            payload = self._metadata(harness_b, active["run_id"])
            assert payload["status"] == "interrupted"

    def test_failed_metadata_keeps_frozen_request(
        self, xdg_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
        with harness.client:
            registration = harness.register_model()

            def failing(self, created_at, run_id, artifact, png):
                raise OSError("disk full")

            monkeypatch.setattr(ArtifactStore, "write_image", failing)
            detail = harness.submit(registration["id"], count=2, steps=1)
            finished = harness.wait_terminal(detail["run_id"], timeout=20)
            assert finished["status"] == "failed"
            assert finished["error"]["code"] == "storage_error"
            payload = self._metadata(harness, detail["run_id"])
            assert payload["status"] == "failed"
            assert payload["parameters"]["prompt"] == "a test prompt"


class TestTerminalMetadataStorageFailure:
    def _metadata(self, harness, run_id: str) -> dict:
        response = harness.client.get(f"/api/generations/{run_id}/metadata")
        assert response.status_code == 200
        return json.loads(response.content)

    def test_terminal_only_failure_never_reports_success(self, xdg_env, tmp_path: Path) -> None:
        """Per-image writes succeed; the terminal (completed) write fails.

        The run must end with a visible storage failure (partial, images
        retained), and the file must never claim a stale running/completed
        state: success cannot be observed before the required write lands.
        """
        harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
        with harness.client:
            registration = harness.register_model()
            original = ArtifactStore.write_metadata

            def fail_completed(self, created_at, run_id, payload):
                if payload["status"] == "completed":
                    raise OSError("injected terminal metadata write failure")
                return original(self, created_at, run_id, payload)

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(ArtifactStore, "write_metadata", fail_completed)
            try:
                detail = harness.submit(registration["id"], count=2)
                finished = harness.wait_terminal(detail["run_id"], timeout=20)
            finally:
                monkeypatch.undo()

            assert finished["status"] == "partial"
            assert finished["error"]["code"] == "storage_error"
            assert "terminal metadata write failed" in finished["error"]["message"]
            statuses = {image["artifact_id"]: image["status"] for image in finished["images"]}
            assert statuses == {"image-001": "completed", "image-002": "completed"}
            payload = self._metadata(harness, detail["run_id"])
            assert payload["status"] == "partial"
            assert payload["error"]["code"] == "storage_error"

    def test_persistent_metadata_failure_is_bounded_and_recovers(
        self, xdg_env, tmp_path: Path
    ) -> None:
        """Every metadata write fails: the run fails visibly, the runtime and
        queue stay serviceable, and the next run recovers full parity."""
        harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
        with harness.client:
            registration = harness.register_model()

            def always_fail(self, created_at, run_id, payload):
                raise OSError("disk full")

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(ArtifactStore, "write_metadata", always_fail)
            try:
                first = harness.submit(registration["id"], count=2)
                failed = harness.wait_terminal(first["run_id"], timeout=20)
            finally:
                monkeypatch.undo()

            # the first image was durably written before its metadata
            # write failed, so the run ends partial with that image retained
            assert failed["status"] == "partial"
            assert failed["error"]["code"] == "storage_error"
            completed = [i for i in failed["images"] if i["status"] == "completed"]
            assert len(completed) >= 1
            # bounded handling left the runtime serviceable
            runtime = harness.client.get("/api/runtime").json()
            assert runtime["state"] in ("idle", "generating")

            second = harness.submit(registration["id"], count=1)
            recovered = harness.wait_terminal(second["run_id"])
            assert recovered["status"] == "completed"
            assert recovered["error"] is None
            payload = self._metadata(harness, second["run_id"])
            assert payload["status"] == "completed"


class TestLateTerminalAfterStorageFailure:
    def test_last_image_failure_with_late_completed_keeps_durable_terminal(
        self, xdg_env, tmp_path: Path
    ) -> None:
        """Only the last image write fails; the worker still reports
        RunCompleted. The durable partial/storage_error terminal and its
        metadata must survive the late success event, retained images stay,
        and the next queued run still progresses."""
        harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
        with harness.client:
            registration = harness.register_model()
            original = ArtifactStore.write_image

            def last_image_fails(self, created_at, run_id, artifact, png):
                if artifact == "image-002":
                    raise OSError("injected last-image failure")
                return original(self, created_at, run_id, artifact, png)

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(ArtifactStore, "write_image", last_image_fails)
            try:
                first = harness.submit(registration["id"], count=2, steps=1)
                finished = harness.wait_terminal(first["run_id"], timeout=20)
                # give the late terminal event time to (wrongly) arrive
                import time as _time

                _time.sleep(0.2)
            finally:
                monkeypatch.undo()

            assert finished["status"] == "partial"
            assert finished["error"]["code"] == "storage_error"
            statuses = {i["artifact_id"]: i["status"] for i in finished["images"]}
            assert statuses["image-001"] == "completed"  # retained

            payload = self._metadata(harness, first["run_id"])
            assert payload["status"] == "partial"  # NOT the late completed
            assert payload["error"]["code"] == "storage_error"
            assert payload["images"][0]["status"] == "completed"

            # the queue keeps serving after the durable failure
            second = harness.submit(registration["id"], count=1, steps=1)
            recovered = harness.wait_terminal(second["run_id"])
            assert recovered["status"] == "completed"
            assert self._metadata(harness, second["run_id"])["status"] == "completed"

    def _metadata(self, harness, run_id: str) -> dict:
        response = harness.client.get(f"/api/generations/{run_id}/metadata")
        assert response.status_code == 200
        return json.loads(response.content)
