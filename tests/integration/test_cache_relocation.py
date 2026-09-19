"""Model identities survive cache relocation without network or real inference."""

import json
from pathlib import Path

import pytest

from image_studio.hub.anima import SHARED_REPO, model_problems
from image_studio.schemas import DownloadStatus, FrozenRunSpec, ProfileId
from image_studio.testing import FakeHub, FakeRuntime
from tests.conftest import build_harness
from tests.integration.test_anima_models import wait_download

MODELS = (
    ("Tongyi-MAI/Z-Image", "z-image"),
    ("Tongyi-MAI/Z-Image-Turbo", "z-image-turbo"),
    ("circlestone-labs/Anima", "anima-turbo"),
    ("Gazingstars123/Anima-2.9B", "anima-2.9b"),
)


class RecordingRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.hold = False
        self.specs = []

    def submit(self, spec: FrozenRunSpec) -> None:
        assert (
            model_problems(
                Path(spec.model.snapshot_path),
                spec.model.repo_id,
                spec.model.profile,
                spec.model.sources,
            )
            == []
        )
        self.specs.append(spec)
        if not self.hold:
            super().submit(spec)


@pytest.mark.parametrize(("repo_id", "profile"), MODELS)
def test_relocated_cache_preserves_library_history_queue_and_download_retry(
    tmp_path, xdg_env, repo_id, profile
):
    old_hub = FakeHub(tmp_path / "host")
    old_hub.seed_snapshot(repo_id)
    old_hub.seed_snapshot(SHARED_REPO)
    runtime = RecordingRuntime()
    original = build_harness(xdg_env, runtime, old_hub)
    with original.client:
        response = original.client.post(
            "/api/models", json={"repo_id": repo_id, "profile": profile, "display_name": "Keep me"}
        )
        assert response.status_code == 201, response.text
        registration = response.json()
        completed = original.submit(registration["id"], seed=123, count=2)
        assert original.wait_terminal(completed["run_id"])["status"] == "completed"
        original.client.patch(f"/api/generations/{completed['run_id']}", json={"favorite": True})
        runtime.hold = True
        queued = original.submit(registration["id"], seed=456)
        app_state = original.client.app.state.app_state
        repository = app_state.repository
        job = repository.create_download(
            repo_id=repo_id,
            revision="main",
            profile=ProfileId(profile),
            sources=runtime.specs[-1].model.sources,
        )
        repository.update_download_progress(job.id, resolved_commit=registration["commit_sha"])
        repository.finish_download(job.id, DownloadStatus.FAILED, error_message="interrupted")
        for table in ("registrations", "downloads", "runs"):
            rows = repository._connection.execute(f"SELECT * FROM {table}").fetchall()
            assert all("snapshot_path" not in row.keys() for row in rows)
            assert all("snapshot_path" not in row["sources"] for row in rows)
        metadata_file = next(app_state.settings.outputs_dir.rglob("metadata.json"))
        old_metadata = metadata_file.read_bytes()
        assert "snapshot_path" not in json.loads(old_metadata)["model"]

    relocated_hub = FakeHub(tmp_path / "container")
    # Replace the empty fake cache with the very same files at a new location.
    relocated_hub.cache_dir.rmdir()
    old_hub.cache_dir.rename(relocated_hub.cache_dir)
    env = {**xdg_env, "HF_HUB_CACHE": str(relocated_hub.cache_dir)}
    resumed_runtime = RecordingRuntime()
    restarted = build_harness(env, resumed_runtime, relocated_hub)
    with restarted.client:
        library = restarted.client.get("/api/models").json()["registrations"]
        assert len(library) == 1
        current = library[0]
        assert current["id"] == registration["id"]
        assert current["commit_sha"] == registration["commit_sha"]
        assert current["display_name"] == "Keep me"
        assert current["status"] == "ready"
        assert Path(current["snapshot_path"]).is_relative_to(relocated_hub.cache_dir)
        history = restarted.client.get(f"/api/generations/{completed['run_id']}").json()
        assert history["status"] == "completed" and history["favorite"]
        assert [image["seed"] for image in history["images"]] == [123, 124]
        assert metadata_file.read_bytes() == old_metadata
        assert (
            restarted.client.get(f"/api/generations/{queued['run_id']}").json()["status"]
            == "paused"
        )
        assert restarted.client.post("/api/generations/queue/resume").status_code == 200
        assert restarted.wait_terminal(queued["run_id"])["status"] == "completed"
        assert resumed_runtime.specs[0].seeds == (456,)
        new_run = restarted.submit(registration["id"])
        assert restarted.wait_terminal(new_run["run_id"])["status"] == "completed"
        for spec in resumed_runtime.specs:
            assert spec.model.snapshot_path == current["snapshot_path"]
            assert spec.model.commit_sha == registration["commit_sha"]
            assert all(
                Path(s.snapshot_path).is_relative_to(relocated_hub.cache_dir)
                for s in spec.model.sources
            )
        assert relocated_hub.api.calls["model_info"] == []
        assert relocated_hub.api.calls["hf_hub_download"] == []

        assert restarted.client.post(f"/api/downloads/{job.id}/retry").status_code == 202
        finished = wait_download(restarted, job.id)
        assert finished["status"] == "completed", finished
        assert finished["resolved_commit"] == registration["commit_sha"]
        pinned = {registration["commit_sha"], *(s["commit_sha"] for s in registration["sources"])}
        assert all(
            call["revision"] in pinned for call in relocated_hub.api.calls["hf_hub_download"]
        )
        assert (
            restarted.client.get("/api/models").json()["registrations"][0]["id"]
            == registration["id"]
        )


@pytest.mark.parametrize(("repo_id", "profile"), MODELS)
def test_missing_pinned_revision_is_not_replaced_by_another_cached_revision(
    tmp_path, xdg_env, repo_id, profile
):
    hub = FakeHub(tmp_path / "hub")
    hub.seed_snapshot(repo_id)
    hub.seed_snapshot(SHARED_REPO)
    runtime = RecordingRuntime()
    runtime.hold = True
    original = build_harness(xdg_env, runtime, hub)
    with original.client:
        registration = original.register_model(repo_id, profile)
        queued = original.submit(registration["id"])
    snapshot = Path(registration["snapshot_path"])
    snapshot.rename(snapshot.with_name("e" * 40))
    (snapshot.parent.parent / "refs" / "main").write_text("e" * 40)
    after = RecordingRuntime()
    restarted = build_harness(xdg_env, after, hub)
    with restarted.client:
        current = restarted.client.get("/api/models").json()["registrations"][0]
        assert current["status"] == "missing_files"
        assert current["commit_sha"] == registration["commit_sha"]
        assert restarted.client.post("/api/generations/queue/resume").status_code == 200
        failed = restarted.wait_terminal(queued["run_id"])
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "cache_incomplete"
        response = restarted.client.post(
            "/api/generations",
            json={
                "registration_id": registration["id"],
                "gpu_uuid": "GPU-fake-0001",
                "prompt": "missing",
            },
        )
        assert response.status_code == 422
        assert after.specs == []
        assert hub.api.calls["model_info"] == []
        assert hub.api.calls["hf_hub_download"] == []
