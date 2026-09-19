"""Resolve fixed Hub identities within the current process's cache.

Paths are runtime values, never persistent model identities. Resolving a
snapshot does not query the Hub, follow a branch, or download any files.
"""

from pathlib import Path

from image_studio.schemas import ModelSource


def snapshot_path(cache_dir: Path, repo_id: str, commit: str) -> Path:
    return cache_dir / ("models--" + repo_id.replace("/", "--")) / "snapshots" / commit


def resolve_sources(cache_dir: Path, sources: list[dict]) -> tuple[ModelSource, ...]:
    """Attach local paths without changing recorded revisions or file selections."""
    return tuple(
        ModelSource(
            **source,
            snapshot_path=str(snapshot_path(cache_dir, source["repo_id"], source["commit_sha"])),
        )
        for source in sources
    )
