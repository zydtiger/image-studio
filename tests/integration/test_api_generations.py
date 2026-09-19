"""Integration tests for the public generation API over the fake runtime."""

from __future__ import annotations

import io

from PIL import Image


def test_submit_freezes_settings_and_completes(harness) -> None:
    registration = harness.register_model()
    detail = harness.submit(registration["id"], count=2, seed=100)
    assert detail["status"] in ("queued", "running")
    assert detail["initial_seed"] == 100
    assert detail["steps"] == 50  # z-image default frozen
    assert detail["guidance"] == 4.0
    assert detail["repo_id"] == "Tongyi-MAI/Z-Image"
    assert [image["artifact_id"] for image in detail["images"]] == ["image-001", "image-002"]

    finished = harness.wait_terminal(detail["run_id"])
    assert finished["status"] == "completed"
    assert [image["seed"] for image in finished["images"]] == [100, 101]
    assert all(image["status"] == "completed" for image in finished["images"])
    urls = finished["images"][0]
    assert urls["url"].startswith(f"/api/generations/{detail['run_id']}/artifacts/")
    assert urls["thumbnail_url"].endswith("/thumbnail")


def test_multiple_run_previews_match_completed_images(harness) -> None:
    registration = harness.register_model()
    expected = {}
    for count in (2, 1):
        submitted = harness.submit(registration["id"], count=count)
        detail = harness.wait_terminal(submitted["run_id"])
        assert detail["status"] == "completed"
        expected[detail["run_id"]] = detail

    response = harness.client.get(
        "/api/generations", params={"model": registration["repo_id"], "has_images": "true"}
    )
    assert response.status_code == 200
    listing = response.json()
    assert listing["total"] == 2
    assert {run["run_id"] for run in listing["runs"]} == set(expected)
    for run in listing["runs"]:
        completed = {
            image["artifact_id"]: image
            for image in expected[run["run_id"]]["images"]
            if image["status"] == "completed"
        }
        assert run["completed_count"] == len(completed)
        assert run["preview_artifact_id"] in completed
        thumbnail = harness.client.get(completed[run["preview_artifact_id"]]["thumbnail_url"])
        assert thumbnail.status_code == 200
        assert thumbnail.headers["content-type"] == "image/webp"
        with Image.open(io.BytesIO(thumbnail.content)) as image:
            assert image.format == "WEBP"


def test_profile_defaults_and_rejections(harness) -> None:
    registration = harness.register_model("Tongyi-MAI/Z-Image-Turbo", profile="z-image-turbo")
    detail = harness.submit(registration["id"], count=1)
    assert detail["steps"] == 9 and detail["guidance"] == 0.0

    rejected = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "x",
            "width": 256,
            "height": 256,
            "negative_prompt": "should fail",
        },
    )
    assert rejected.status_code == 422
    body = rejected.json()
    assert body["error"]["code"] == "validation"
    assert body["error"]["details"]["field"] == "negative_prompt"

    overflow = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "x",
            "width": 256,
            "height": 256,
            "seed": 4294967295,
            "count": 2,
        },
    )
    assert overflow.status_code == 422
    assert overflow.json()["error"]["details"]["reason"] == "seed_overflow"


def test_unknown_gpu_rejected_with_available_list(harness) -> None:
    registration = harness.register_model()
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-does-not-exist",
            "prompt": "x",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert details["field"] == "gpu_uuid"
    assert "GPU-fake-0001" in details["available"]


def test_missing_registration_and_incomplete_cache(harness) -> None:
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": "nope",
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "x",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 404

    harness.hub.seed_snapshot("Tongyi-MAI/Z-Image")
    registration = harness.client.post(
        "/api/models", json={"repo_id": "Tongyi-MAI/Z-Image", "profile": "z-image"}
    ).json()
    # simulate external cache deletion
    for path in harness.hub.cache_dir.rglob("model_index.json"):
        path.unlink()
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "x",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "cache_incomplete"


def test_partial_run_retains_completed_images(harness) -> None:
    harness.runtime.fail_image_at = 2
    registration = harness.register_model()
    detail = harness.submit(registration["id"], count=3)
    finished = harness.wait_terminal(detail["run_id"])
    assert finished["status"] == "partial"
    statuses = {image["artifact_id"]: image["status"] for image in finished["images"]}
    assert statuses == {
        "image-001": "completed",
        "image-002": "cancelled",
        "image-003": "cancelled",
    }
    assert finished["error"]["code"] == "worker_error"


def test_cancel_queued_and_running(harness) -> None:
    registration = harness.register_model()
    harness.runtime.image_delay = 0.05
    first = harness.submit(registration["id"], count=4, prompt="long")
    second = harness.submit(registration["id"], count=1, prompt="short")
    cancelled = harness.client.post(f"/api/generations/{second['run_id']}/cancel")
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"

    detail = harness.client.get(f"/api/generations/{first['run_id']}").json()
    assert detail["status"] in ("running", "queued")
    mid_cancel = harness.client.post(f"/api/generations/{first['run_id']}/cancel").json()
    assert mid_cancel["status"] in ("running", "partial", "cancelled")
    finished = harness.wait_terminal(first["run_id"])
    assert finished["status"] in ("cancelled", "partial")
    completed = [image for image in finished["images"] if image["status"] == "completed"]
    assert len(completed) < 4
    # healthy worker stays resident after cancellation
    runtime = harness.client.get("/api/runtime").json()
    assert runtime["state"] in ("idle", "generating")
    assert runtime["resident"] is not None


def test_favorite_trash_restore_and_listing(harness) -> None:
    registration = harness.register_model()
    detail = harness.submit(registration["id"], count=1)
    run_id = detail["run_id"]
    harness.wait_terminal(run_id)

    toggled = harness.client.patch(f"/api/generations/{run_id}", json={"favorite": True})
    assert toggled.json()["favorite"] is True
    listing = harness.client.get("/api/generations", params={"favorite": "true"}).json()
    assert [run["run_id"] for run in listing["runs"]] == [run_id]

    trashed = harness.client.post(f"/api/generations/{run_id}/trash").json()
    assert trashed["trashed"] is True
    assert harness.client.get("/api/generations").json()["total"] == 0
    trash_only = harness.client.get("/api/generations", params={"trashed": "only"}).json()
    assert [run["run_id"] for run in trash_only["runs"]] == [run_id]
    # double trash -> conflict
    assert harness.client.post(f"/api/generations/{run_id}/trash").status_code == 409

    restored = harness.client.post(f"/api/generations/{run_id}/restore").json()
    assert restored["trashed"] is False
    assert harness.client.get("/api/generations").json()["total"] == 1


def test_listing_image_filters_hide_only_empty_cancelled_runs(harness) -> None:
    registration = harness.register_model()
    # failed run without images
    harness.runtime.fail_image_at = 1
    failed = harness.submit(registration["id"], count=1, prompt="failed empty")
    assert harness.wait_terminal(failed["run_id"])["status"] == "failed"
    harness.runtime.fail_image_at = None
    # completed run with images
    with_images = harness.submit(registration["id"], count=1, prompt="with images")
    assert harness.wait_terminal(with_images["run_id"])["status"] == "completed"
    # cancelled while queued: no images; long run keeps the worker busy
    harness.runtime.image_delay = 0.05
    long_run = harness.submit(registration["id"], count=4, prompt="long run")
    queued = harness.submit(registration["id"], count=1, prompt="cancelled empty")
    cancelled = harness.client.post(f"/api/generations/{queued['run_id']}/cancel")
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert harness.wait_terminal(long_run["run_id"])["status"] == "completed"

    def prompts(**params: str) -> tuple[set[str], int]:
        listing = harness.client.get("/api/generations", params=params).json()
        return {run["prompt"] for run in listing["runs"]}, listing["total"]

    # Default listing behavior is unchanged: every run is returned.
    seen, total = prompts()
    assert total == 4
    assert seen == {"failed empty", "with images", "long run", "cancelled empty"}

    # The library default hides only the empty cancellation.
    seen, total = prompts(exclude_empty_cancelled="true")
    assert total == 3
    assert seen == {"failed empty", "with images", "long run"}

    # An explicit cancelled status filter exposes it again.
    seen, total = prompts(status="cancelled", exclude_empty_cancelled="true")
    assert total == 1 and seen == {"cancelled empty"}

    # has_images is tri-state: true requires, false forbids a completed
    # image, and the filtered total drives LIMIT/OFFSET paging.
    seen, total = prompts(has_images="true")
    assert total == 2 and seen == {"with images", "long run"}
    listing = harness.client.get(
        "/api/generations", params={"has_images": "false", "limit": 1, "offset": 1}
    ).json()
    assert listing["total"] == 2
    assert [run["prompt"] for run in listing["runs"]] == ["failed empty"]


def test_trash_active_run_conflict(harness) -> None:
    registration = harness.register_model()
    harness.runtime.image_delay = 0.05
    detail = harness.submit(registration["id"], count=4)
    assert harness.client.post(f"/api/generations/{detail['run_id']}/trash").status_code == 409
    harness.client.post(f"/api/generations/{detail['run_id']}/cancel")
    harness.wait_terminal(detail["run_id"])


def test_artifact_serving_thumbnails_and_metadata(harness) -> None:
    registration = harness.register_model()
    detail = harness.submit(registration["id"], count=2, seed=5)
    harness.wait_terminal(detail["run_id"])
    run_id = detail["run_id"]

    artifact = harness.client.get(f"/api/generations/{run_id}/artifacts/image-001")
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(artifact.content))
    assert image.size == (256, 256)
    assert "content-disposition" not in artifact.headers

    downloaded = harness.client.get(
        f"/api/generations/{run_id}/artifacts/image-001", params={"download": "1"}
    )
    assert downloaded.headers["content-disposition"].startswith("attachment")

    thumbnail = harness.client.get(f"/api/generations/{run_id}/artifacts/image-001/thumbnail")
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/webp"
    assert thumbnail.content[:4] == b"RIFF"

    import json

    metadata = harness.client.get(f"/api/generations/{run_id}/metadata")
    assert metadata.status_code == 200
    payload = json.loads(metadata.content)
    assert payload["run_id"] == run_id
    assert payload["parameters"]["initial_seed"] == 5
    assert payload["images"][0]["seed"] == 5
    assert payload["images"][1]["seed"] == 6


def test_artifact_path_confinement(harness) -> None:
    registration = harness.register_model()
    detail = harness.submit(registration["id"], count=1)
    harness.wait_terminal(detail["run_id"])
    for traversal in ("..%2F..%2Fetc%2Fpasswd", "image-999", "metadata"):
        response = harness.client.get(f"/api/generations/{detail['run_id']}/artifacts/{traversal}")
        assert response.status_code == 404, traversal
    assert harness.client.get(f"/api/generations/{detail['run_id']}0000").status_code == 404


def test_error_envelope_shape(harness) -> None:
    response = harness.client.get("/api/generations/unknown-run")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) >= {"code", "message"}
    unknown_api = harness.client.get("/api/definitely/not/a/route")
    assert unknown_api.status_code == 404
    assert unknown_api.json()["error"]["code"] == "not_found"


def test_gpu_resolved_before_profile_validation(harness) -> None:
    """Both errors present: the unknown GPU wins per contract §2 ordering."""
    harness.hub.seed_snapshot("Tongyi-MAI/Z-Image-Turbo")
    registration = harness.client.post(
        "/api/models", json={"repo_id": "Tongyi-MAI/Z-Image-Turbo", "profile": "z-image-turbo"}
    ).json()
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-does-not-exist",
            "prompt": "x",
            "width": 256,
            "height": 256,
            "negative_prompt": "invalid for turbo",
        },
    )
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert details.get("field") == "gpu_uuid"
