"""Local Hugging Face cache inspection and snapshot validation.

Uses ``scan_cache_dir()`` for discovery. The application never modifies the
shared cache through this module and never treats cache presence as proof a
model is runnable: ``snapshot_problems`` validates a cached Z-Image snapshot
against the official component manifest — pipeline class, component
declarations, per-component config/tokenizer files, and weights as either a
single safetensors file or an index referencing all its shards, in the
default layout or consistently in the ``bf16`` variant layout. No heavy
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
#: These are the default (unvaried) filenames; ``variant_weight_files``
#: derives variant counterparts from them.
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

#: Weight-file variants a snapshot may consistently use, in preference
#: order: a complete default layout always wins, ``bf16`` applies only when
#: the whole snapshot carries bf16 weights. Other variants are not
#: supported; runtime dtype is a separate concept chosen per run.
SUPPORTED_WEIGHT_VARIANTS: tuple[str | None, ...] = (None, "bf16")


def variant_weight_files(component: str, variant: str | None) -> tuple[str, str]:
    """Filenames of one component's single weight file and sharding index.

    Variant naming follows the installed loaders (diffusers 0.40 /
    transformers 5.17 ``_add_variant``): the variant is inserted before the
    final extension — ``model.bf16.safetensors`` and the transformers-style
    sharded index ``model.safetensors.index.bf16.json``. Shard names come
    from each index's ``weight_map`` and are not assumed here.
    """
    single, sharded_index = COMPONENT_WEIGHT_FILES[component]
    if variant is None:
        return single, sharded_index
    stem, extension = sharded_index.rsplit(".", 1)
    return f"{single.rsplit('.', 1)[0]}.{variant}.safetensors", f"{stem}.{variant}.{extension}"


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
                incomplete=_snapshot_incomplete(Path(revision.snapshot_path), repo.repo_id),
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


def _snapshot_incomplete(root: Path, repo_id: str | None = None) -> bool:
    """Truthful incompleteness for known required manifests only.

    A readable Z-Image ``model_index.json`` enables exact manifest checks.
    A snapshot that carries Diffusers component directories but no readable
    manifest is an interrupted pipeline download and counts as incomplete;
    snapshots of other architectures are listed without claiming validated
    completeness in either direction.
    """
    from image_studio.hub.anima import RECIPES, make_sources, model_problems

    for profile, recipe in RECIPES.items():
        if repo_id == recipe.repo_id:
            sources = make_sources(root.parent.parent.parent, repo_id, root.name, profile)
            return bool(model_problems(root, repo_id, profile, sources))
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
    problems.extend(_select_weight_layout(root)[1])
    return problems


def snapshot_weight_variant(root: Path) -> str | None:
    """The weight variant a validated snapshot loads under, or ``None``.

    Shared with ``snapshot_problems`` so registration, submit validation,
    and the worker's pipeline load can never disagree: a complete default
    layout loads through the default filenames (``None``), a snapshot whose
    weights only exist as ``bf16`` files loads through ``variant='bf16'``.
    A snapshot with no complete layout returns ``None``; loading it then
    fails on its missing files like before, and validation reports why.
    """
    return _select_weight_layout(root)[0]


def _select_weight_layout(root: Path) -> tuple[str | None, list[SnapshotProblem]]:
    """Deterministically pick the single complete weight layout of a snapshot.

    Default first: ordinary snapshots keep their exact previous behavior
    even when extra bf16 files ride along. ``bf16`` applies only when every
    weighted component is complete in bf16, so partial layouts are never
    mixed. When neither layout is complete, the problems of both are
    reported — they show why each candidate layout fails.
    """
    default_problems = _layout_problems(root, None)
    if not default_problems:
        return None, []
    bf16_problems = _layout_problems(root, "bf16")
    if not bf16_problems:
        return "bf16", []
    return None, [*default_problems, *bf16_problems]


def _layout_problems(root: Path, variant: str | None) -> list[SnapshotProblem]:
    problems: list[SnapshotProblem] = []
    for component in COMPONENT_WEIGHT_FILES:
        problems.extend(_weights_problems(root, component, variant))
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


def _weights_problems(root: Path, component: str, variant: str | None) -> list[SnapshotProblem]:
    single, sharded_index = variant_weight_files(component, variant)
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
