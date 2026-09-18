"""Unit tests for cache discovery and snapshot resolution over a fake HF cache."""

from __future__ import annotations

from pathlib import Path

from image_studio.hub import cache as hub_cache
from image_studio.testing import FakeHub


def test_scan_lists_seeded_repo_with_refs_and_snapshots(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    hub.seed_snapshot("Tongyi-MAI/Z-Image")
    repos = hub_cache.scan(hub.cache_dir)
    assert [repo.repo_id for repo in repos] == ["Tongyi-MAI/Z-Image"]
    repo = repos[0]
    assert repo.refs == ["main"]
    assert len(repo.snapshots) == 1
    assert repo.snapshots[0].commit_sha == "a" * 40


def test_find_snapshot_by_branch_tag_or_commit(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    hub.seed_snapshot("Tongyi-MAI/Z-Image")
    for revision in (None, "main", "a" * 40):
        hit = hub_cache.find_snapshot("Tongyi-MAI/Z-Image", revision, hub.cache_dir)
        assert hit is not None
        assert hit.commit_sha == "a" * 40
        assert (hit.path / "model_index.json").is_file()
    assert hub_cache.find_snapshot("Tongyi-MAI/Z-Image", "unknown-ref", hub.cache_dir) is None


def test_snapshot_validation_accepts_complete_official_layout(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    snapshot = hub.seed_snapshot("Tongyi-MAI/Z-Image")
    assert hub_cache.snapshot_problems(snapshot) == []


def test_snapshot_validation_reports_missing_manifest_files(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    snapshot = hub.seed_snapshot("Tongyi-MAI/Z-Image")
    (snapshot / "model_index.json").unlink()
    problems = hub_cache.snapshot_problems(snapshot)
    assert len(problems) == 1
    assert problems[0].code.value == "cache_incomplete"
    assert "model_index.json" in problems[0].detail

    # a Z-Image snapshot missing a component weight is incomplete, too
    hub = FakeHub(tmp_path / "hub2")
    snapshot = hub.seed_snapshot("Tongyi-MAI/Z-Image")
    (snapshot / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").unlink()
    problems = hub_cache.snapshot_problems(snapshot)
    assert any(
        "diffusion_pytorch_model-00001-of-00002.safetensors" in problem.detail
        for problem in problems
    )


def test_scan_includes_incomplete_snapshots(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    snapshot = hub.seed_snapshot("Tongyi-MAI/Z-Image")
    (snapshot / "model_index.json").unlink()  # incomplete but still discovered
    repos = hub_cache.scan(hub.cache_dir)
    assert len(repos) == 1
    assert repos[0].snapshots[0].commit_sha == "a" * 40
    assert repos[0].snapshots[0].incomplete is True


def test_absent_cache_dir_is_empty_listing_without_creation(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    assert hub_cache.scan(absent) == []
    assert hub_cache.find_snapshot("Tongyi-MAI/Z-Image", None, absent) is None
    assert hub_cache.snapshot_for_commit("Tongyi-MAI/Z-Image", "a" * 40, absent) is None
    assert not absent.exists(), "inspection must never create the shared cache"


def test_scan_does_not_conflate_dataset_repos(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    hub.seed_snapshot("Tongyi-MAI/Z-Image")
    dataset = hub.cache_dir / "datasets--org--dataset" / "snapshots" / ("d" * 40)
    dataset.mkdir(parents=True)
    (dataset / "data.txt").write_text("data")
    repos = hub_cache.scan(hub.cache_dir)
    assert [repo.repo_id for repo in repos] == ["Tongyi-MAI/Z-Image"]
