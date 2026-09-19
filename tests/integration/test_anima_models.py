"""Original-checkpoint onboarding and frozen component provenance, CPU only."""

import json
import time
from pathlib import Path

import pytest

from image_studio.hub.anima import RECIPES, SHARED_COMMIT, SHARED_FILES, SHARED_REPO
from image_studio.schemas import ProfileId


def wait_download(harness, job_id):
    for _ in range(300):
        jobs = harness.client.get("/api/downloads").json()["jobs"]
        job = next(job for job in jobs if job["id"] == job_id)
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.01)
    pytest.fail("download never reached a terminal state")


def download(harness, profile):
    recipe = RECIPES[profile]
    response = harness.client.post(
        "/api/downloads",
        json={
            "repo_id": recipe.repo_id,
            "profile": profile.value,
        },
    )
    assert response.status_code == 202, response.text
    return wait_download(harness, response.json()["id"])


def register(harness, profile, commit=None):
    response = harness.client.post(
        "/api/models",
        json={
            "repo_id": RECIPES[profile].repo_id,
            "profile": profile.value,
            "revision": commit,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("profile", list(RECIPES))
def test_original_checkpoint_download_register_generate(harness, profile):
    recipe = RECIPES[profile]
    compat = harness.client.get(f"/api/hub/models/{recipe.repo_id}/compatibility").json()
    assert compat["structurally_compatible"]
    assert compat["selectable_profiles"] == [profile.value]
    assert harness.hub.api.calls["hf_hub_download"] == []

    job = download(harness, profile)
    assert job["status"] == "completed", job
    assert job["sources"][1]["commit_sha"] == SHARED_COMMIT
    assert job["progress"]["files_done"] == 1 + len(SHARED_FILES)
    assert job["progress"]["bytes_done"] == job["progress"]["bytes_total"]
    calls = harness.hub.api.calls["hf_hub_download"]
    assert {(c["repo_id"], c["filename"]) for c in calls} == {
        (recipe.repo_id, recipe.checkpoint),
        *((SHARED_REPO, name) for name in SHARED_FILES),
    }
    assert all(c["revision"] in (job["resolved_commit"], SHARED_COMMIT) for c in calls)
    before = len(calls)
    registrations = harness.client.get("/api/models").json()["registrations"]
    assert len(registrations) == 1
    registration = registrations[0]
    assert registration["profile"] == profile.value
    assert registration["commit_sha"] == job["resolved_commit"]
    assert registration["sources"] == job["sources"]
    assert len(calls) == before  # registration is offline
    overrides = {"seed": 11, "count": 2}
    if profile == ProfileId.ANIMA_29B:
        overrides["negative_prompt"] = "blurry"
    run = harness.submit(registration["id"], prompt="a watercolor landscape 山", **overrides)
    result = harness.wait_terminal(run["run_id"])
    assert result["status"] == "completed"
    assert result["profile"] == profile.value
    assert result["steps"] == (10 if profile == ProfileId.ANIMA_TURBO else 40)
    assert result["guidance"] == (1 if profile == ProfileId.ANIMA_TURBO else 4)
    assert result["sources"] == job["sources"]
    assert [image["seed"] for image in result["images"]] == [11, 12]
    data_dir = harness.client.app.state.app_state.settings.outputs_dir
    metadata = next(data_dir.rglob("metadata.json"))
    model_metadata = json.loads(metadata.read_text())["model"]
    assert "snapshot_path" not in model_metadata
    assert model_metadata["sources"] == [
        {key: value for key, value in source.items() if key != "snapshot_path"}
        for source in result["sources"]
    ]


def test_missing_shared_component_disables_registration_and_submission(harness):
    profile = ProfileId.ANIMA_TURBO
    primary = harness.hub.seed_snapshot(RECIPES[profile].repo_id)
    response = harness.client.post(
        "/api/models",
        json={
            "repo_id": RECIPES[profile].repo_id,
            "profile": profile.value,
        },
    )
    assert response.status_code == 422
    shared = harness.hub.seed_snapshot(SHARED_REPO)
    registration = register(harness, profile)
    missing = shared / "vae/config.json"
    saved = missing.read_bytes()
    missing.unlink()
    library = harness.client.get("/api/models").json()["registrations"]
    assert library[0]["status"] == "missing_files"
    renamed = harness.client.patch(
        f"/api/models/{registration['id']}",
        json={"profile": profile.value, "display_name": "My Anima"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["display_name"] == "My Anima"
    cache = harness.client.get("/api/cache/models").json()["repos"]
    cached = next(repo for repo in cache if repo["repo_id"] == RECIPES[profile].repo_id)
    assert cached["snapshots"][0]["incomplete"]
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "a landscape",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 422
    assert "vae/config.json" in response.text
    assert SHARED_REPO in response.text
    missing.write_bytes(saved)
    assert harness.client.get("/api/models").json()["registrations"][0]["status"] == "ready"
    assert primary.exists()


@pytest.mark.parametrize("wrong_profile", ["z-image", "anima-2.9b"])
def test_anima_profile_mismatch_rejected_on_download_registration_and_edit(harness, wrong_profile):
    profile = ProfileId.ANIMA_TURBO
    recipe = RECIPES[profile]
    harness.hub.seed_snapshot(recipe.repo_id)
    harness.hub.seed_snapshot(SHARED_REPO)
    for endpoint in ("models", "downloads"):
        response = harness.client.post(
            f"/api/{endpoint}",
            json={
                "repo_id": recipe.repo_id,
                "profile": wrong_profile,
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unsupported_model"
    registration = register(harness, profile)
    response = harness.client.patch(
        f"/api/models/{registration['id']}",
        json={
            "profile": wrong_profile,
            "display_name": "must not change",
        },
    )
    assert response.status_code == 422
    assert harness.client.get("/api/models").json()["registrations"][0]["display_name"] is None


def test_retry_keeps_all_revisions_and_completed_files(harness, monkeypatch):
    profile = ProfileId.ANIMA_29B
    api = harness.hub.api
    original = api.hf_hub_download
    fail = True

    def unreliable(repo_id, filename, *, revision=None):
        if repo_id == SHARED_REPO and fail:
            raise OSError("connection interrupted")
        return original(repo_id, filename, revision=revision)

    monkeypatch.setattr(api, "hf_hub_download", unreliable)
    job = download(harness, profile)
    assert job["status"] == "failed"
    assert harness.client.get("/api/models").json()["registrations"] == []
    checkpoint = Path(job["sources"][0]["snapshot_path"]) / RECIPES[profile].checkpoint
    assert checkpoint.exists()
    frozen = job["sources"]
    api.calls["model_info"].clear()
    fail = False
    response = harness.client.post(f"/api/downloads/{job['id']}/retry")
    assert response.status_code == 202
    finished = wait_download(harness, job["id"])
    assert finished["status"] == "completed", finished
    assert finished["sources"] == frozen
    registrations = harness.client.get("/api/models").json()["registrations"]
    assert len(registrations) == 1
    assert registrations[0]["sources"] == frozen
    assert all(
        c["revision"] in (job["resolved_commit"], SHARED_COMMIT) for c in api.calls["model_info"]
    )


def test_turbo_rejects_cfg_override_and_negative_prompt(harness):
    harness.hub.seed_snapshot(RECIPES[ProfileId.ANIMA_TURBO].repo_id)
    harness.hub.seed_snapshot(SHARED_REPO)
    registration = register(harness, ProfileId.ANIMA_TURBO)
    for extra in ({"guidance": 0}, {"guidance": 4}, {"negative_prompt": "blurry"}):
        response = harness.client.post(
            "/api/generations",
            json={
                "registration_id": registration["id"],
                "gpu_uuid": "GPU-fake-0001",
                "prompt": "landscape",
                "width": 256,
                "height": 256,
                **extra,
            },
        )
        assert response.status_code == 422


@pytest.mark.parametrize("profile", list(RECIPES))
def test_registration_and_generation_accept_symlinked_hf_cache(tmp_path, xdg_env, profile):
    from image_studio.hub.anima import make_sources, model_problems
    from image_studio.hub.cache import find_snapshot
    from image_studio.testing import FakeHub, FakeRuntime
    from tests.conftest import build_harness

    actual_cache = tmp_path / "storage" / "hub"
    actual_cache.mkdir(parents=True)
    cache_link = tmp_path / "hf-cache"
    cache_link.symlink_to(actual_cache, target_is_directory=True)
    hub = FakeHub(cache_link)
    primary = hub.seed_snapshot(RECIPES[profile].repo_id)
    hub.seed_snapshot(SHARED_REPO)
    hit = find_snapshot(RECIPES[profile].repo_id, primary.name, hub.cache_dir)
    assert hit.path == primary.resolve()
    assert hit.path != primary
    sources = make_sources(hub.cache_dir, RECIPES[profile].repo_id, hit.commit_sha, profile)
    assert model_problems(hit.path, RECIPES[profile].repo_id, profile, sources) == []
    # Different directories must still be rejected, even with the same basename.
    other = tmp_path / "unrelated" / hit.commit_sha
    other.mkdir(parents=True)
    assert (
        "recipe mismatch"
        in model_problems(other, RECIPES[profile].repo_id, profile, sources)[0].detail
    )

    harness = build_harness(xdg_env, FakeRuntime(), hub)
    with harness.client:
        registration = register(harness, profile, hit.commit_sha)
        assert registration["sources"][0]["snapshot_path"] == str(primary)
        assert registration["snapshot_path"] == str(primary)
        run = harness.submit(registration["id"], prompt="a mountain landscape")
        assert harness.wait_terminal(run["run_id"])["status"] == "completed"
