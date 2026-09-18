"""Local Hugging Face cache inspection and snapshot validation.

Uses ``scan_cache_dir()`` for discovery. The application never modifies the
shared cache through this module and never treats cache presence as proof a
model is runnable: ``snapshot_problems`` validates a cached Z-Image snapshot
against the official component manifest — pipeline class, component
declarations, per-component config/tokenizer files, and weights as either a
single safetensors file or an index referencing all its shards. No heavy
imports, no remote code, no network: validation reads local files only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from image_studio.schemas import CachedRepo, CachedSnapshot, ErrorCode

#: Pipeline class of the only supported architecture (Base and Turbo share it).
ZIMAGE_PIPELINE_CLASS = "ZImagePipeline"

#: Components every official Z-Image / Z-Image-Turbo model_index.json declares.
REQUIRED_COMPONENTS = ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")

#: Non-weight files each component must provide locally.
COMPONENT_CONFIG_FILES = {
    "scheduler": ("scheduler/scheduler_config.json",),
    "text_encoder": ("text_encoder/config.json",),
    "tokenizer": ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"),
    "transformer": ("transformer/config.json",),
    "vae": ("vae/config.json",),
}

#: Weight file layouts per weighted component: single file or sharded index.
COMPONENT_WEIGHT_FILES = {
    "text_encoder": (
        "text_encoder/model.safetensors",
        "text_encoder/model.safetensors.index.json",
    ),
    "transformer": (
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ),
    "vae": (
        "vae/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors.index.json",
    ),
}


@dataclass(frozen=True)
class SnapshotProblem:
    code: ErrorCode
    detail: str

    def __str__(self) -> str:  # pragma: no cover - convenience for callers
        return self.detail


@dataclass(frozen=True)
class SnapshotHit:
    commit_sha: str
    path: Path


def _is_zimage_snapshot(root: Path) -> bool:
    try:
        index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(index, dict) and index.get("_class_name") == ZIMAGE_PIPELINE_CLASS


def scan(cache_dir: Path) -> list[CachedRepo]:
    """Map every cached model repository, including unsupported and partial ones.

    An absent cache directory is an empty listing, never an error, and the
    shared cache is never created or modified merely to inspect it. Dataset
    and space caches are not conflated with model repositories.
    """
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        result = scan_cache_dir(cache_dir=cache_dir)
    except CacheNotFound:
        return []
    repos: list[CachedRepo] = []
    for repo in result.repos:
        if getattr(repo, "repo_type", "model") != "model":
            continue
        revisions = sorted(repo.revisions, key=lambda rev: rev.last_modified, reverse=True)
        snapshots = [
            CachedSnapshot(
                commit_sha=revision.commit_hash,
                path=str(revision.snapshot_path),
                size_bytes=revision.size_on_disk,
                incomplete=_snapshot_incomplete(Path(revision.snapshot_path)),
            )
            for revision in revisions
        ]
        repos.append(
            CachedRepo(
                repo_id=repo.repo_id,
                size_on_disk_bytes=repo.size_on_disk,
                refs=sorted(repo.refs),
                snapshots=snapshots,
            )
        )
    return sorted(repos, key=lambda repo: repo.repo_id)


def _snapshot_incomplete(root: Path) -> bool:
    """Truthful incompleteness for known required manifests only.

    A readable Z-Image ``model_index.json`` enables exact manifest checks.
    A snapshot that carries Diffusers component directories but no readable
    manifest is an interrupted pipeline download and counts as incomplete;
    snapshots of other architectures are listed without claiming validated
    completeness in either direction.
    """
    if (root / "model_index.json").is_file():
        if not _is_zimage_snapshot(root):
            return False
        return bool(snapshot_problems(root))
    return any((root / component).is_dir() for component in REQUIRED_COMPONENTS)


def find_snapshot(repo_id: str, revision: str | None, cache_dir: Path) -> SnapshotHit | None:
    """Resolve a revision (branch/tag/commit) to a cached model snapshot directory.

    Ref names resolve through the cache's ref mapping; a full commit hash may
    be requested directly. Without a revision, the most recently modified
    cached snapshot is used. An absent cache directory resolves to None.
    """
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        result = scan_cache_dir(cache_dir=cache_dir)
    except CacheNotFound:
        return None
    for repo in result.repos:
        if repo.repo_id != repo_id or getattr(repo, "repo_type", "model") != "model":
            continue
        revision_info = repo.refs.get(revision) if revision is not None else None
        if revision_info is None and revision is not None:
            revision_info = next(
                (item for item in repo.revisions if item.commit_hash == revision), None
            )
        elif revision_info is None:
            revision_info = (
                max(repo.revisions, key=lambda item: item.last_modified) if repo.revisions else None
            )
        if revision_info is not None:
            return SnapshotHit(
                commit_sha=revision_info.commit_hash,
                path=Path(revision_info.snapshot_path),
            )
    return None


def snapshot_for_commit(repo_id: str, commit_sha: str, cache_dir: Path) -> SnapshotHit | None:
    for repo in scan(cache_dir):
        if repo.repo_id != repo_id:
            continue
        for cached in repo.snapshots:
            if cached.commit_sha == commit_sha:
                return SnapshotHit(commit_sha=cached.commit_sha, path=Path(cached.path))
    return None


# ---------------------------------------------------------------------------
# Shared Z-Image snapshot validation (registration, downloads, submit, display)
# ---------------------------------------------------------------------------


#: Official component declarations inspected from the official Z-Image and
#: Z-Image-Turbo ``model_index.json`` files (Base and Turbo are identical).
#: Narrow by design: no generic plugin framework, no substitution guesses.
KNOWN_COMPONENT_DECLARATIONS: dict[str, tuple[str, str]] = {
    "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
    "text_encoder": ("transformers", "Qwen3Model"),
    "tokenizer": ("transformers", "Qwen2Tokenizer"),
    "transformer": ("diffusers", "ZImageTransformer2DModel"),
    "vae": ("diffusers", "AutoencoderKL"),
}


def manifest_problems(index: object) -> list[SnapshotProblem]:
    """Shared declaration rules for local snapshots and remote listings.

    Applies the same checks everywhere so local validation and remote
    compatibility can never drift: the manifest must be an object, the
    pipeline class must be ``ZImagePipeline``, and every required component
    must carry its official ``[library, class]`` declaration. Malformed
    declarations are ``cache_incomplete``; a supported-looking but foreign
    library/class is ``unsupported_model``.
    """
    if not isinstance(index, dict):
        return [
            SnapshotProblem(
                ErrorCode.CACHE_INCOMPLETE, "model_index.json is malformed or unreadable"
            )
        ]
    class_name = index.get("_class_name")
    if class_name != ZIMAGE_PIPELINE_CLASS:
        return [
            SnapshotProblem(
                ErrorCode.UNSUPPORTED_MODEL,
                f"unsupported pipeline class {class_name!r}; only {ZIMAGE_PIPELINE_CLASS} "
                "is supported",
            )
        ]
    problems: list[SnapshotProblem] = []
    for component in REQUIRED_COMPONENTS:
        declaration = index.get(component)
        if declaration is None:
            problems.append(
                SnapshotProblem(
                    ErrorCode.CACHE_INCOMPLETE,
                    f"model_index.json does not declare component {component!r}",
                )
            )
            continue
        if not (
            isinstance(declaration, (list, tuple))
            and len(declaration) == 2
            and all(isinstance(entry, str) and entry for entry in declaration)
        ):
            problems.append(
                SnapshotProblem(
                    ErrorCode.CACHE_INCOMPLETE,
                    f"model_index.json declares component {component!r} malformed; "
                    "expected [library, class]",
                )
            )
            continue
        expected = KNOWN_COMPONENT_DECLARATIONS[component]
        if (declaration[0], declaration[1]) != expected:
            problems.append(
                SnapshotProblem(
                    ErrorCode.UNSUPPORTED_MODEL,
                    f"unsupported {component} declaration "
                    f"{declaration[0]}.{declaration[1]}; expected "
                    f"{expected[0]}.{expected[1]}",
                )
            )
    return problems


def snapshot_problems(root: Path) -> list[SnapshotProblem]:
    """Validate a local snapshot against the official Z-Image manifest.

    Offline and dependency-free: reads local files only. Problems carry typed
    codes — ``unsupported_model`` for a foreign architecture,
    ``cache_incomplete`` for anything missing or malformed.
    """
    index_path = root / "model_index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, "model_index.json is missing")]
    except (OSError, ValueError) as exc:
        return [
            SnapshotProblem(
                ErrorCode.CACHE_INCOMPLETE,
                f"model_index.json is malformed or unreadable ({exc.__class__.__name__})",
            )
        ]

    problems = manifest_problems(index)
    if any(problem.code is ErrorCode.UNSUPPORTED_MODEL for problem in problems):
        return problems
    for component in REQUIRED_COMPONENTS:
        for relative in COMPONENT_CONFIG_FILES.get(component, ()):
            if not (root / relative).is_file():
                problems.append(
                    SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{relative} is missing")
                )
        if component in COMPONENT_WEIGHT_FILES:
            problems.extend(_weights_problems(root, component))
    return problems


def _safe_shard_name(name: str) -> bool:
    """A shard name must be a plain relative filename inside its component."""
    from pathlib import PurePosixPath

    candidate = PurePosixPath(name)
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in name
        and not name.startswith("/")
    )


def _weights_problems(root: Path, component: str) -> list[SnapshotProblem]:
    single, sharded_index = COMPONENT_WEIGHT_FILES[component]
    if (root / single).is_file():
        return []
    index_path = root / sharded_index
    if not index_path.is_file():
        return [
            SnapshotProblem(
                ErrorCode.CACHE_INCOMPLETE,
                f"{component} weights are missing (expected {single} or {sharded_index})",
            )
        ]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [
            SnapshotProblem(
                ErrorCode.CACHE_INCOMPLETE, f"{sharded_index} is malformed or unreadable"
            )
        ]
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        # an empty or non-object weight_map describes no weights at all
        return [
            SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{sharded_index} has no usable weight_map")
        ]
    shards: set[str] = set()
    for shard in weight_map.values():
        if not isinstance(shard, str) or not shard:
            return [
                SnapshotProblem(
                    ErrorCode.CACHE_INCOMPLETE,
                    f"{sharded_index} contains an invalid shard name",
                )
            ]
        if not _safe_shard_name(shard):
            return [
                SnapshotProblem(
                    ErrorCode.CACHE_INCOMPLETE,
                    f"{sharded_index} contains an unsafe shard name {shard!r}",
                )
            ]
        shards.add(shard)
    problems = []
    for shard in shards:
        if not (root / component / shard).is_file():
            problems.append(
                SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{component}/{shard} is missing")
            )
    return problems


def problem_codes(problems: list[SnapshotProblem]) -> ErrorCode:
    """Pick the dominant error code: unsupported architecture beats missing files."""
    for problem in problems:
        if problem.code is ErrorCode.UNSUPPORTED_MODEL:
            return ErrorCode.UNSUPPORTED_MODEL
    return ErrorCode.CACHE_INCOMPLETE if problems else ErrorCode.INTERNAL


def problem_details(problems: list[SnapshotProblem]) -> dict[str, list[str]]:
    return {"problems": [problem.detail for problem in problems]}
