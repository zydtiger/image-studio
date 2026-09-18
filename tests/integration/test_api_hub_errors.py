"""HTTP integration tests for Hub error translation with realistic SDK errors."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from huggingface_hub.errors import GatedRepoError, RevisionNotFoundError

from image_studio.testing import FakeHub, FakeRuntime
from tests.conftest import build_harness


def _gated_error() -> GatedRepoError:
    return GatedRepoError(
        "gated",
        response=httpx.Response(
            403, request=httpx.Request("GET", "https://huggingface.co/api/models/test/gated")
        ),
    )


def _wait_terminal_job(harness, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    job = {}
    while time.monotonic() < deadline:
        job = next(
            item
            for item in harness.client.get("/api/downloads").json()["jobs"]
            if item["id"] == job_id
        )
        if job["status"] in ("failed", "completed", "cancelled"):
            return job
        time.sleep(0.01)
    return job


def test_unknown_repo_maps_to_404_not_500(harness) -> None:
    response = harness.client.get("/api/hub/models/nobody/nothing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_gated_repo_maps_to_403(harness) -> None:
    original = harness.hub.api.model_info

    def gated(*args, **kwargs):
        raise _gated_error()

    harness.hub.api.model_info = gated
    try:
        response = harness.client.get("/api/hub/models/test/gated")
    finally:
        harness.hub.api.model_info = original
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "gated_model"


def test_network_failure_maps_to_502(harness) -> None:
    original = harness.hub.api.list_models

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    harness.hub.api.list_models = unreachable
    try:
        response = harness.client.get("/api/hub/models", params={"q": "z"})
    finally:
        harness.hub.api.list_models = original
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "hub_unreachable"


def test_missing_revision_maps_to_404(harness) -> None:
    original = harness.hub.api.model_info

    def missing_revision(repo_id, revision=None, files_metadata=False):
        raise RevisionNotFoundError(
            "no such revision",
            response=httpx.Response(
                404,
                request=httpx.Request(
                    "GET", f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
                ),
            ),
        )

    harness.hub.api.model_info = missing_revision
    try:
        response = harness.client.get(
            "/api/hub/models/Tongyi-MAI/Z-Image/compatibility", params={"revision": "gone"}
        )
    finally:
        harness.hub.api.model_info = original
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "revision_not_found"


def test_download_failure_is_typed_and_queue_not_starved(xdg_env, tmp_path: Path) -> None:
    """A gated download fails with its real code; the next job still runs."""
    harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
    with harness.client:
        # drive the app's own hub stack directly
        app_state = harness.client.app.state.app_state
        original_lister = app_state.hub.file_lister

        def gated_lister(repo_id, revision):
            raise _gated_error()

        app_state.hub.file_lister = gated_lister
        first = harness.client.post(
            "/api/downloads",
            json={"repo_id": "Tongyi-MAI/Z-Image", "revision": "main", "profile": "z-image"},
        ).json()
        job = _wait_terminal_job(harness, first["id"])
        assert job["status"] == "failed"
        assert job["error"]["code"] == "gated_model"

        app_state.hub.file_lister = original_lister
        second = harness.client.post(
            "/api/downloads",
            json={"repo_id": "Tongyi-MAI/Z-Image", "revision": "main", "profile": "z-image"},
        ).json()
        job2 = _wait_terminal_job(harness, second["id"])
        assert job2["status"] == "completed"


def test_lazy_search_failure_during_iteration_maps_to_502(harness) -> None:
    """SDK list_models is a generator: errors surface while iterating."""
    original = harness.hub.api.list_models

    def lazy_boom(*args, **kwargs):
        yield from ()
        raise httpx.ConnectError("iteration-time offline")

    harness.hub.api.list_models = lazy_boom
    try:
        response = harness.client.get("/api/hub/models", params={"q": "z"})
    finally:
        harness.hub.api.list_models = original
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "hub_unreachable"
