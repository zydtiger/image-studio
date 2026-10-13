"""Fixed component recipe for Unsloth's Qwen-Image-2.1 FP8 export."""

from __future__ import annotations

import json
from pathlib import Path

from image_studio.hub.cache import SnapshotProblem
from image_studio.hub.paths import snapshot_path
from image_studio.schemas import ErrorCode, ImageStudioError, ModelSource, ProfileId

REPO = "unsloth/Qwen-Image-2.1-FP8"
BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_COMMIT = "d26bb61231c349cf6b7896fa83353113880e1ba3"
TRANSFORMER_FILE = "Qwen-Image-2.1-FP8.safetensors"
TEXT_ENCODER_FILE = "Qwen-Image-2.1-text_encoder-FP8.safetensors"
WEIGHT_FILES = (TRANSFORMER_FILE, TEXT_ENCODER_FILE)
BASE_FILES = (
    "model_index.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
)


def make_sources(cache_dir: Path, repo_id: str, commit: str, profile: ProfileId):
    if profile != ProfileId.QWEN_IMAGE_21 or repo_id != REPO:
        raise ImageStudioError(ErrorCode.UNSUPPORTED_MODEL, "Unsloth FP8 requires qwen-image-2.1")
    return tuple(
        ModelSource(
            repo_id=repo,
            commit_sha=sha,
            files=files,
            snapshot_path=str(snapshot_path(cache_dir, repo, sha)),
        )
        for repo, sha, files in (
            (REPO, commit, WEIGHT_FILES),
            (BASE_REPO, BASE_COMMIT, BASE_FILES),
        )
    )


def resolve_sources(hub, repo_id: str, revision: str | None, profile: ProfileId):
    from image_studio.hub.client import translate_hub_error

    if profile != ProfileId.QWEN_IMAGE_21:
        raise ImageStudioError(ErrorCode.UNSUPPORTED_MODEL, "Unsloth FP8 requires qwen-image-2.1")
    try:
        commit, listing = hub.file_lister(repo_id, revision)
        if not commit:
            raise ImageStudioError(ErrorCode.REVISION_NOT_FOUND, "could not resolve FP8 revision")
        sources = make_sources(hub.cache_dir, repo_id, commit, profile)
        base_commit, base_listing = hub.file_lister(BASE_REPO, BASE_COMMIT)
        if base_commit != BASE_COMMIT:
            raise ImageStudioError(ErrorCode.REVISION_NOT_FOUND, "Qwen component revision mismatch")
        for source, files in zip(sources, (listing, base_listing), strict=True):
            missing = set(source.files) - {name for name, _ in files}
            if missing:
                raise ImageStudioError(
                    ErrorCode.CACHE_INCOMPLETE,
                    f"{source.repo_id}: missing {', '.join(sorted(missing))}",
                )
        return sources
    except Exception as exc:
        raise translate_hub_error(exc) from exc


def model_problems(root: Path, repo_id: str, profile: ProfileId, sources=()):
    if profile != ProfileId.QWEN_IMAGE_21 or repo_id != REPO:
        return [SnapshotProblem(ErrorCode.UNSUPPORTED_MODEL, "Unsloth FP8 profile mismatch")]
    if len(sources) != 2:
        return [SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, "Unsloth FP8 sources are missing")]
    expected = make_sources(root.parent.parent.parent, repo_id, root.name, profile)
    if any(
        actual.model_dump(exclude={"snapshot_path"}) != wanted.model_dump(exclude={"snapshot_path"})
        or Path(actual.snapshot_path).resolve() != Path(wanted.snapshot_path).resolve()
        for actual, wanted in zip(sources, expected, strict=True)
    ):
        return [
            SnapshotProblem(ErrorCode.UNSUPPORTED_MODEL, "Unsloth FP8 component recipe mismatch")
        ]
    problems = []
    for source in sources:
        for name in source.files:
            path = Path(source.snapshot_path) / name
            label = f"{source.repo_id}/{name}"
            if not path.is_file():
                problems.append(SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{label} is missing"))
            elif name.endswith(".json"):
                try:
                    if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                        raise ValueError("expected an object")
                except (OSError, ValueError):
                    problems.append(
                        SnapshotProblem(ErrorCode.CACHE_INCOMPLETE, f"{label} is malformed")
                    )
    return problems
