"""Explicit recipes for the two supported original Anima checkpoints.

Only metadata is read in the API process. All tensor conversion belongs to
the worker. Shared components are pinned to the author's official export;
neither registration nor retries follow a moving dependency branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from image_studio.hub.cache import SnapshotProblem
from image_studio.hub.paths import snapshot_path
from image_studio.schemas import ErrorCode, ImageStudioError, ModelSource, ProfileId

SHARED_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
SHARED_COMMIT = "073c3a9db359c31ad0e8aa268d15775473c2176c"
SHARED_FILES = (
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
    "t5_tokenizer/tokenizer.json",
    "t5_tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "text_conditioner/config.json",
)


@dataclass(frozen=True)
class AnimaRecipe:
    repo_id: str
    checkpoint: str
    layers: int


RECIPES = {
    ProfileId.ANIMA_TURBO: AnimaRecipe(
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
        28,
    ),
    ProfileId.ANIMA_29B: AnimaRecipe(
        "Gazingstars123/Anima-2.9B", "Anima-2.9B-preview-v1.safetensors", 40
    ),
}


def recipe_for(repo_id: str, profile: ProfileId) -> AnimaRecipe | None:
    recipe = RECIPES.get(profile)
    if recipe is not None and recipe.repo_id != repo_id:
        raise ImageStudioError(
            ErrorCode.UNSUPPORTED_MODEL,
            f"{profile.value} requires {recipe.repo_id}",
        )
    if recipe is None and any(r.repo_id == repo_id for r in RECIPES.values()):
        raise ImageStudioError(
            ErrorCode.UNSUPPORTED_MODEL, "profile does not match this Anima repo"
        )
    return recipe


def make_sources(
    cache_dir: Path, repo_id: str, commit: str, profile: ProfileId
) -> tuple[ModelSource, ...]:
    recipe = recipe_for(repo_id, profile)
    if recipe is None:
        return ()
    return (
        ModelSource(
            repo_id=repo_id,
            commit_sha=commit,
            files=(recipe.checkpoint,),
            snapshot_path=str(snapshot_path(cache_dir, repo_id, commit)),
        ),
        ModelSource(
            repo_id=SHARED_REPO,
            commit_sha=SHARED_COMMIT,
            files=SHARED_FILES,
            snapshot_path=str(snapshot_path(cache_dir, SHARED_REPO, SHARED_COMMIT)),
        ),
    )


def resolve_sources(hub, repo_id: str, revision: str | None, profile: ProfileId):
    """Freeze the complete download before enqueueing; never download weights."""
    from image_studio.hub.client import translate_hub_error

    if recipe_for(repo_id, profile) is None:
        return ()
    try:
        commit, primary_files = hub.file_lister(repo_id, revision)
        if not commit:
            raise ImageStudioError(ErrorCode.REVISION_NOT_FOUND, "could not resolve model revision")
        sources = make_sources(hub.cache_dir, repo_id, commit, profile)
        shared_commit, shared_files = hub.file_lister(SHARED_REPO, SHARED_COMMIT)
        if shared_commit != SHARED_COMMIT:
            raise ImageStudioError(
                ErrorCode.REVISION_NOT_FOUND, "shared component revision mismatch"
            )
        for source, listing in zip(sources, (primary_files, shared_files), strict=True):
            missing = set(source.files) - {name for name, _ in listing}
            if missing:
                raise ImageStudioError(
                    ErrorCode.CACHE_INCOMPLETE,
                    f"{source.repo_id}@{source.commit_sha}: missing {', '.join(sorted(missing))}",
                )
        return sources
    except ImageStudioError:
        raise
    except Exception as exc:
        raise translate_hub_error(exc) from exc


def model_problems(
    root: Path, repo_id: str, profile: ProfileId, sources: tuple[ModelSource, ...] = ()
) -> list[SnapshotProblem]:
    """Same local recipe checks at registration, display, download and submit."""
    from image_studio.hub.cache import snapshot_problems

    try:
        recipe = recipe_for(repo_id, profile)
    except ImageStudioError as exc:
        return [SnapshotProblem(exc.code, exc.message)]
    if recipe is None:
        return snapshot_problems(root)
    if len(sources) != 2:
        return [SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, "Anima component sources are missing")]
    primary, shared = sources
    if (
        primary.repo_id != repo_id
        or Path(primary.snapshot_path).resolve() != root.resolve()
        or primary.files != (recipe.checkpoint,)
        or shared.repo_id != SHARED_REPO
        or shared.files != SHARED_FILES
    ):
        return [SnapshotProblem(ErrorCode.UNSUPPORTED_MODEL, "Anima component recipe mismatch")]
    problems = []
    for source in sources:
        for name in source.files:
            path = Path(source.snapshot_path) / name
            label = f"{source.repo_id}@{source.commit_sha[:12]}/{name}"
            if not path.is_file():
                problems.append(SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{label} is missing"))
            elif name.endswith(".json"):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError("expected an object")
                except (OSError, ValueError):
                    problems.append(
                        SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{label} is malformed")
                    )
    return problems
