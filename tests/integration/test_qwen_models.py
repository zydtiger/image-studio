"""Qwen onboarding and generation contract through the real API with fake providers."""

from pathlib import Path

import pytest

from image_studio.testing import FakeHub, FakeRepoSpec, FakeRuntime, default_fake_repos
from tests.conftest import build_harness
from tests.integration.test_anima_models import wait_download
from tests.unit.test_qwen_image import qwen_files

REPO = "Qwen/Qwen-Image-2.1"
PROFILE = "qwen-image-2.1"


@pytest.fixture
def qwen_harness(tmp_path, xdg_env):
    hub = FakeHub(
        tmp_path / "hub",
        repos=[
            *default_fake_repos(),
            FakeRepoSpec(repo_id=REPO, files=qwen_files(), sha="e" * 40),
        ],
    )
    harness = build_harness(xdg_env, FakeRuntime(), hub)
    with harness.client:
        yield harness


def test_download_register_generate_and_missing_processor(qwen_harness):
    h = qwen_harness
    compat = h.client.get(f"/api/hub/models/{REPO}/compatibility").json()
    assert compat["selectable_profiles"] == [PROFILE]
    created = h.client.post("/api/downloads", json={"repo_id": REPO, "profile": PROFILE})
    assert created.status_code == 202, created.text
    job = wait_download(h, created.json()["id"])
    assert job["status"] == "completed", job
    calls = h.hub.api.calls["hf_hub_download"]
    assert {c["filename"] for c in calls} == set(qwen_files())
    assert all(c["revision"] == "e" * 40 for c in calls)
    download_count = len(calls)
    (registration,) = h.client.get("/api/models").json()["registrations"]
    assert registration["profile"] == PROFILE
    assert registration["commit_sha"] == "e" * 40
    run = h.submit(registration["id"], prompt="a mountain 山", seed=42, count=2)
    result = h.wait_terminal(run["run_id"])
    assert result["status"] == "completed"
    assert result["steps"] == 40 and result["guidance"] == 1
    assert [image["seed"] for image in result["images"]] == [42, 43]
    assert len(h.hub.api.calls["hf_hub_download"]) == download_count

    (Path(registration["snapshot_path"]) / "processor/chat_template.jinja").unlink()
    assert h.client.get("/api/models").json()["registrations"][0]["status"] == "missing_files"
    response = h.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "mountain",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 422


def test_profile_mismatch_rejected_on_registration_edit_and_download(qwen_harness):
    h = qwen_harness
    h.hub.seed_snapshot(REPO)
    assert (
        h.client.post("/api/models", json={"repo_id": REPO, "profile": "z-image"}).status_code
        == 422
    )
    registration = h.register_model(REPO, PROFILE)
    assert (
        h.client.patch(f"/api/models/{registration['id']}", json={"profile": "z-image"}).status_code
        == 422
    )
    h.hub.seed_snapshot("Tongyi-MAI/Z-Image")
    assert (
        h.client.post(
            "/api/models", json={"repo_id": "Tongyi-MAI/Z-Image", "profile": PROFILE}
        ).status_code
        == 422
    )
    created = h.client.post("/api/downloads", json={"repo_id": REPO, "profile": "z-image"})
    job = wait_download(h, created.json()["id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "unsupported_model"


@pytest.mark.parametrize(
    "extra", [{"width": 272}, {"height": 272}, {"guidance": 4}, {"negative_prompt": "blurry"}]
)
def test_profile_constraints_rejected_before_queueing(qwen_harness, extra):
    h = qwen_harness
    registration = h.register_model(REPO, PROFILE)
    response = h.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "mountain",
            "width": 256,
            "height": 256,
            **extra,
        },
    )
    assert response.status_code == 422
