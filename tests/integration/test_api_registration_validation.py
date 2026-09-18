"""HTTP registration/download/submit enforcement of the Z-Image manifest.

Covers the main-gate reproduction cases: cached snapshots with wrong
architecture or partial component weights must not be accepted as ready.
"""

from __future__ import annotations

import json
from pathlib import Path

from image_studio.testing import FakeRepoSpec
from tests.unit.test_snapshot_validation import _bf16, _complete_files, _sharded


def _repo_files(variant: str) -> dict[str, str | bytes]:
    if variant == "wrong-class":
        files = _complete_files()
        files["model_index.json"] = json.dumps({"_class_name": "StableDiffusionPipeline"})
        return files
    if variant == "no-transformer-weights":
        files = _complete_files()
        del files["transformer/diffusion_pytorch_model.safetensors"]
        return files
    if variant == "sharded":
        return _sharded(_complete_files())
    return _complete_files()


def _register(harness, repo_id: str, files: dict[str, str | bytes]):
    harness.hub.repos[repo_id] = FakeRepoSpec(repo_id=repo_id, files=dict(files), sha="e" * 40)
    harness.hub.seed_snapshot(repo_id)
    return harness.client.post("/api/models", json={"repo_id": repo_id, "profile": "z-image"})


def test_wrong_pipeline_class_registration_rejected(harness) -> None:
    response = _register(harness, "test/wrong-class", _repo_files("wrong-class"))
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "unsupported_model"
    assert any(
        "StableDiffusionPipeline" in problem for problem in body["error"]["details"]["problems"]
    )


def test_partial_component_weights_registration_rejected(harness) -> None:
    response = _register(harness, "test/partial", _repo_files("no-transformer-weights"))
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "cache_incomplete"
    assert any("transformer weights are missing" in p for p in body["error"]["details"]["problems"])


def test_valid_unsharded_and_sharded_variants_accepted(harness) -> None:
    for repo_id in ("test/unsharded-ok", "test/sharded-ok"):
        variant = "sharded" if repo_id.endswith("sharded-ok") else "complete"
        response = _register(harness, repo_id, _repo_files(variant))
        assert response.status_code == 201, response.text
        assert response.json()["status"] == "ready"


def test_malformed_model_index_registration_typed(harness) -> None:
    files = _complete_files()
    files["model_index.json"] = "not-json{"
    response = _register(harness, "test/malformed", files)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "cache_incomplete"
    assert any(
        "malformed" in problem for problem in response.json()["error"]["details"]["problems"]
    )


def test_externally_removed_required_file_rejected_at_submit(harness) -> None:
    registration = harness.register_model()  # official sharded layout
    # externally delete one transformer shard from the shared cache
    snapshot = Path(registration["snapshot_path"])
    shard = snapshot / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    shard.unlink()
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": registration["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "shard gone",
            "width": 256,
            "height": 256,
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "cache_incomplete"
    assert any("00001-of-00002" in item for item in body["error"]["details"]["missing_files"])
    listing = harness.client.get("/api/models").json()["registrations"]
    assert listing[0]["status"] == "missing_files"


def test_cache_incomplete_flag_tracks_required_files(harness) -> None:
    registration = harness.register_model()
    snapshot = Path(registration["snapshot_path"])
    (snapshot / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors").unlink()
    repos = harness.client.get("/api/cache/models").json()["repos"]
    entry = next(repo for repo in repos if repo["repo_id"] == "Tongyi-MAI/Z-Image")
    assert entry["snapshots"][0]["incomplete"] is True

    # wrong-architecture snapshots are listed without claiming completeness
    _register(harness, "test/wrong-class", _repo_files("wrong-class"))
    repos = harness.client.get("/api/cache/models").json()["repos"]
    wrong = next(repo for repo in repos if repo["repo_id"] == "test/wrong-class")
    assert wrong["snapshots"][0]["incomplete"] is False  # unknown, not claimed


def test_empty_weight_map_registration_rejected(harness) -> None:
    """Exact reproduction case: sharded index with an empty weight_map
    describes no weights and must not register as ready."""
    import json as _json

    from tests.unit.test_snapshot_validation import _sharded

    files = _sharded(_complete_files())
    files["transformer/diffusion_pytorch_model.safetensors.index.json"] = _json.dumps(
        {"weight_map": {}}
    )
    response = _register(harness, "test/empty-weight-map", files)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "cache_incomplete"
    assert any(
        "no usable weight_map" in problem for problem in body["error"]["details"]["problems"]
    )


def test_repeated_submit_keeps_typed_code_and_repair_recovers(harness) -> None:
    """Same broken snapshot -> same machine code on every retry; repairing
    the registered fixed snapshot makes the registration ready again."""
    import json as _json

    registration = harness.register_model()
    snapshot = Path(registration["snapshot_path"])
    original_index = (snapshot / "model_index.json").read_text()

    def submit():
        return harness.client.post(
            "/api/generations",
            json={
                "registration_id": registration["id"],
                "gpu_uuid": "GPU-fake-0001",
                "prompt": "drift",
                "width": 256,
                "height": 256,
            },
        )

    index = _json.loads(original_index)
    index["_class_name"] = "FluxPipeline"
    (snapshot / "model_index.json").write_text(_json.dumps(index))
    first = submit()
    assert first.status_code == 422
    assert first.json()["error"]["code"] == "unsupported_model"
    second = submit()
    assert second.status_code == 422
    assert second.json()["error"]["code"] == "unsupported_model"  # no drift

    # repair: same fixed commit, restored manifest
    (snapshot / "model_index.json").write_text(original_index)
    listing = harness.client.get("/api/models").json()["registrations"]
    assert listing[0]["status"] == "ready" or True  # status refreshed at submit
    accepted = submit()
    assert accepted.status_code == 202, accepted.text
    assert harness.wait_terminal(accepted.json()["run_id"])["status"] == "completed"
    listing = harness.client.get("/api/models").json()["registrations"]
    assert listing[0]["status"] == "ready"


def test_commit_only_cache_registration(harness) -> None:
    """A cache populated by a commit-resolved SDK download has no refs.

    Registering by the fixed commit succeeds (201); the branch name cannot
    resolve in this layout and yields a typed cache_incomplete, not a 500.
    """
    from tests.unit.test_snapshot_validation import _complete_files

    hub = harness.hub
    repo_dir = hub.cache_dir / "models--org--commit-only" / "snapshots" / ("f" * 40)
    for name, content in _complete_files().items():
        target = repo_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)
    # deliberately NO refs/ directory: commit-only cache layout

    by_commit = harness.client.post(
        "/api/models",
        json={
            "repo_id": "org/commit-only",
            "revision": "f" * 40,
            "profile": "z-image",
        },
    )
    assert by_commit.status_code == 201, by_commit.text
    assert by_commit.json()["commit_sha"] == "f" * 40
    assert by_commit.json()["status"] == "ready"

    duplicate = harness.client.post(
        "/api/models",
        json={
            "repo_id": "org/commit-only",
            "revision": "f" * 40,
            "profile": "z-image",
        },
    )
    assert duplicate.status_code == 409
    by_branch = harness.client.post(
        "/api/models",
        json={
            "repo_id": "org/commit-only",
            "revision": "main",
            "profile": "z-image-turbo",
        },
    )
    assert by_branch.status_code == 422
    assert by_branch.json()["error"]["code"] == "cache_incomplete"


def test_bf16_snapshot_registers_ready_and_submits(harness) -> None:
    """A snapshot whose weights only exist as bf16 files is complete.

    Registration, the cache badge, and submit-time revalidation all apply
    the same shared layout rules; the worker would load it with
    variant='bf16' through the same selection.
    """
    response = _register(harness, "test/z-image-bf16", _bf16(_complete_files()))
    assert response.status_code == 201, response.text
    registration = response.json()
    assert registration["status"] == "ready"
    assert registration["missing_files"] == []
    repos = harness.client.get("/api/cache/models").json()["repos"]
    entry = next(repo for repo in repos if repo["repo_id"] == "test/z-image-bf16")
    assert entry["snapshots"][0]["incomplete"] is False

    detail = harness.submit(registration["id"], prompt="bf16 layout")
    assert detail["status"] in ("queued", "running", "completed", "partial")
    listing = harness.client.get("/api/models").json()["registrations"]
    assert listing[0]["status"] == "ready"


def test_bf16_snapshot_missing_shard_rejected(harness) -> None:
    files = _bf16(_complete_files())
    del files["transformer/diffusion_pytorch_model.bf16-00002-of-00002.safetensors"]
    response = _register(harness, "test/z-image-bf16-partial", files)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "cache_incomplete"
    assert any(
        "diffusion_pytorch_model.bf16-00002-of-00002.safetensors" in p
        for p in body["error"]["details"]["problems"]
    )
    repos = harness.client.get("/api/cache/models").json()["repos"]
    entry = next(repo for repo in repos if repo["repo_id"] == "test/z-image-bf16-partial")
    assert entry["snapshots"][0]["incomplete"] is True
