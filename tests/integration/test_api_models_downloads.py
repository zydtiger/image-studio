"""Integration tests for model registration and download endpoints."""

from __future__ import annotations

import time


def _wait_terminal_status(harness, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    job = {}
    while time.monotonic() < deadline:
        job = next(
            item
            for item in harness.client.get("/api/downloads").json()["jobs"]
            if item["id"] == job_id
        )
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    return job


def test_register_ready_and_duplicate_conflict(harness) -> None:
    registration = harness.register_model()
    assert registration["status"] == "ready"
    assert registration["commit_sha"] == "a" * 40
    assert registration["snapshot_path"]

    duplicate = harness.client.post(
        "/api/models", json={"repo_id": "Tongyi-MAI/Z-Image", "profile": "z-image"}
    )
    assert duplicate.status_code == 409


def test_register_missing_snapshot_or_files(harness) -> None:
    missing_repo = harness.client.post(
        "/api/models", json={"repo_id": "someone/other-model", "profile": "z-image"}
    )
    assert missing_repo.status_code == 422
    assert missing_repo.json()["error"]["code"] == "cache_incomplete"

    harness.hub.seed_snapshot("Tongyi-MAI/Z-Image")
    for path in harness.hub.cache_dir.rglob("model_index.json"):
        path.unlink()
    incomplete = harness.client.post(
        "/api/models", json={"repo_id": "Tongyi-MAI/Z-Image", "profile": "z-image"}
    )
    assert incomplete.status_code == 422
    details = incomplete.json()["error"]["details"]
    assert any("model_index.json" in problem for problem in details["problems"])


def test_patch_display_name_allowed_while_resident(harness) -> None:
    registration = harness.register_model()
    detail = harness.submit(registration["id"], count=1)
    harness.wait_terminal(detail["run_id"])
    patched = harness.client.patch(
        f"/api/models/{registration['id']}", json={"display_name": "My Z"}
    )
    assert patched.status_code == 200
    assert patched.json()["display_name"] == "My Z"


def test_delete_guarded_while_referenced(harness) -> None:
    registration = harness.register_model()
    harness.runtime.image_delay = 0.05
    detail = harness.submit(registration["id"], count=4)
    # active run references the registration
    assert harness.client.delete(f"/api/models/{registration['id']}").status_code == 409
    # resident worker references the registration
    harness.client.post(f"/api/generations/{detail['run_id']}/cancel")
    harness.wait_terminal(detail["run_id"])
    assert (
        harness.client.get("/api/runtime").json()["resident"]["registration_id"]
        == registration["id"]
    )
    assert harness.client.delete(f"/api/models/{registration['id']}").status_code == 409
    # eject then delete succeeds and keeps history
    assert harness.client.post("/api/runtime/eject").status_code == 204
    assert harness.client.delete(f"/api/models/{registration['id']}").status_code == 204
    history = harness.client.get("/api/generations").json()
    assert history["total"] == 1
    assert history["runs"][0]["repo_id"] == "Tongyi-MAI/Z-Image"


def test_download_lifecycle_and_retry(harness) -> None:
    response = harness.client.post(
        "/api/downloads",
        json={"repo_id": "Tongyi-MAI/Z-Image", "revision": "main", "profile": "z-image"},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    job = _wait_terminal_status(harness, job_id)
    assert job["status"] == "completed"
    assert job["resolved_commit"] == "a" * 40
    assert job["progress"]["files_done"] == job["progress"]["files_total"]

    # completed jobs cannot be retried or cancelled
    assert harness.client.post(f"/api/downloads/{job_id}/retry").status_code == 409
    assert harness.client.post(f"/api/downloads/{job_id}/cancel").status_code == 409

    registrations = harness.client.get("/api/models").json()["registrations"]
    assert len(registrations) == 1
    assert registrations[0]["repo_id"] == "Tongyi-MAI/Z-Image"
    assert registrations[0]["commit_sha"] == job["resolved_commit"]
    assert registrations[0]["profile"] == job["profile"]
    assert registrations[0]["status"] == "ready"


def test_hub_search_detail_compatibility_cache(harness) -> None:
    search = harness.client.get("/api/hub/models", params={"q": "z-image", "limit": 5})
    assert search.status_code == 200
    results = search.json()["results"]
    assert any(item["repo_id"] == "Tongyi-MAI/Z-Image" for item in results)
    assert harness.hub.api.calls["list_models"][-1] == {"search": "z-image", "limit": 5}

    detail = harness.client.get("/api/hub/models/Tongyi-MAI/Z-Image")
    assert detail.status_code == 200
    assert detail.json()["license"] == "apache-2.0"

    compatibility = harness.client.get("/api/hub/models/Tongyi-MAI/Z-Image/compatibility")
    assert compatibility.status_code == 200
    report = compatibility.json()
    assert report["structurally_compatible"] is True
    assert set(report["selectable_profiles"]) == {"z-image", "z-image-turbo"}

    unknown = harness.client.get("/api/hub/models/nobody/nothing")
    assert unknown.status_code == 404

    cache_listing = harness.client.get("/api/cache/models")
    assert cache_listing.status_code == 200
    # the compatibility check fetched model_index.json into the cache, so the
    # repo is discoverable with a partial, detached snapshot
    listed = cache_listing.json()["repos"]
    assert [repo["repo_id"] for repo in listed] == ["Tongyi-MAI/Z-Image"]
    assert listed[0]["refs"] == []
    harness.hub.seed_snapshot("Tongyi-MAI/Z-Image-Turbo")
    repos = harness.client.get("/api/cache/models").json()["repos"]
    assert [repo["repo_id"] for repo in repos] == [
        "Tongyi-MAI/Z-Image",
        "Tongyi-MAI/Z-Image-Turbo",
    ]
