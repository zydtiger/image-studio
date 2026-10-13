"""Unsloth onboarding downloads exactly the frozen FP8 recipe through the public API."""

from pathlib import Path

import pytest

from image_studio.hub import qwen_fp8 as recipe
from image_studio.hub.paths import snapshot_path
from image_studio.schemas import ProfileId
from image_studio.testing import FakeHub, FakeRepoSpec, FakeRuntime
from tests.conftest import build_harness
from tests.integration.test_anima_models import wait_download
from tests.unit.test_qwen_image import qwen_files


def repos():
    base = qwen_files()
    base.update({name: "{}" for name in recipe.BASE_FILES if name not in base})
    base["vae/diffusion_pytorch_model.safetensors"] = b"fake-vae"
    return [
        FakeRepoSpec(
            repo_id=recipe.REPO,
            sha="f" * 40,
            files={
                **{name: b"fake-fp8" for name in recipe.WEIGHT_FILES},
                "Qwen-Image-2.1-INT8.safetensors": b"do-not-download",
                "Qwen-Image-2.1-INT8-ConvRot.safetensors": b"do-not-download",
                "samples/example.png": b"do-not-download",
            },
        ),
        FakeRepoSpec(repo_id=recipe.BASE_REPO, sha=recipe.BASE_COMMIT, files=base),
    ]


@pytest.fixture
def fp8_harness(tmp_path, xdg_env):
    actual = tmp_path / "hub"
    actual.mkdir()
    link = tmp_path / "cache-link"
    link.symlink_to(actual, target_is_directory=True)
    hub = FakeHub(link, repos=repos())
    h = build_harness(xdg_env, FakeRuntime(), hub)
    with h.client:
        yield h


def download(h):
    response = h.client.post(
        "/api/downloads", json={"repo_id": recipe.REPO, "profile": "qwen-image-2.1"}
    )
    assert response.status_code == 202, response.text
    job = wait_download(h, response.json()["id"])
    assert job["status"] == "completed", job
    return job


def test_discover_download_register_and_generate_only_fp8(fp8_harness):
    h = fp8_harness
    found = h.client.get("/api/hub/models", params={"q": recipe.REPO}).json()["results"]
    assert any(item["repo_id"] == recipe.REPO for item in found)
    report = h.client.get(f"/api/hub/models/{recipe.REPO}/compatibility").json()
    assert report["structurally_compatible"]
    assert report["selectable_profiles"] == ["qwen-image-2.1"]
    job = download(h)
    calls = h.hub.api.calls["hf_hub_download"]
    assert {(c["repo_id"], c["filename"], c["revision"]) for c in calls} == {
        *((recipe.REPO, name, "f" * 40) for name in recipe.WEIGHT_FILES),
        *((recipe.BASE_REPO, name, recipe.BASE_COMMIT) for name in recipe.BASE_FILES),
    }
    assert job["progress"]["files_total"] == len(recipe.WEIGHT_FILES) + len(recipe.BASE_FILES)
    (registration,) = h.client.get("/api/models").json()["registrations"]
    assert registration["status"] == "ready"
    assert registration["sources"] == job["sources"]
    before = len(calls)
    run = h.submit(registration["id"], prompt="a mountain 山", seed=42)
    result = h.wait_terminal(run["run_id"])
    assert result["status"] == "completed"
    assert result["steps"] == 40 and result["guidance"] == 1
    assert result["sources"] == job["sources"]
    assert len(calls) == before
    cache = h.client.get("/api/cache/models").json()["repos"]
    primary = next(r for r in cache if r["repo_id"] == recipe.REPO)
    assert not primary["snapshots"][0]["incomplete"]

    # Registration also works via a symlinked cache root, after library removal.
    h.runtime.eject()
    h.client.delete(f"/api/models/{registration['id']}")
    response = h.client.post(
        "/api/models", json={"repo_id": recipe.REPO, "profile": "qwen-image-2.1"}
    )
    assert response.status_code == 201, response.text


def test_missing_component_blocks_registration_and_generation(fp8_harness):
    h = fp8_harness
    h.hub.seed_snapshot(recipe.REPO)
    assert (
        h.client.post(
            "/api/models", json={"repo_id": recipe.REPO, "profile": "qwen-image-2.1"}
        ).status_code
        == 422
    )
    download(h)
    (registration,) = h.client.get("/api/models").json()["registrations"]
    base = Path(registration["sources"][1]["snapshot_path"])
    (base / "processor/chat_template.jinja").unlink()
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
    assert "processor/chat_template.jinja" in response.text
    cache = h.client.get("/api/cache/models").json()["repos"]
    assert next(r for r in cache if r["repo_id"] == recipe.REPO)["snapshots"][0]["incomplete"]


def test_profile_mismatch_refused_before_transfer(fp8_harness):
    h = fp8_harness
    response = h.client.post("/api/downloads", json={"repo_id": recipe.REPO, "profile": "z-image"})
    assert response.status_code == 422
    assert not h.hub.api.calls["hf_hub_download"]


def test_recipe_rejects_drift_and_accepts_resolved_cache_paths(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    link = tmp_path / "link"
    link.symlink_to(cache, target_is_directory=True)
    sources = recipe.make_sources(link, recipe.REPO, "f" * 40, ProfileId.QWEN_IMAGE_21)
    for source in sources:
        for name in source.files:
            p = Path(source.snapshot_path) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
    root = snapshot_path(cache, recipe.REPO, "f" * 40)
    assert recipe.model_problems(root, recipe.REPO, ProfileId.QWEN_IMAGE_21, sources) == []
    changed = (sources[0], sources[1].model_copy(update={"commit_sha": "a" * 40}))
    assert recipe.model_problems(root, recipe.REPO, ProfileId.QWEN_IMAGE_21, changed)
