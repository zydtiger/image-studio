"""Structural compatibility checks for Z-Image pipeline repositories.

The check inspects file listings and declarative configuration only. It
never executes remote code, never enables ``trust_remote_code``, and never
infers the distillation profile: Base versus Turbo is an explicit user
choice, and structural compatibility never implies output quality. Remote
inspection shares the local snapshot manifest spec (``hub.cache``) so both
paths apply the same component rules; all configuration reads use the
resolved fixed commit so a moving branch cannot drift between fetches.
"""

from __future__ import annotations

import fnmatch
import json
from typing import Any

from image_studio import schemas
from image_studio.hub.cache import (
    COMPONENT_CONFIG_FILES,
    COMPONENT_WEIGHT_FILES,
    REQUIRED_COMPONENTS,
    manifest_problems,
)
from image_studio.schemas import ImageStudioError, ProfileId

ALLOW_PATTERNS = ("*.json", "*.txt", "tokenizer*", "*.safetensors")
DISALLOWED_PATTERNS = ("*.bin", "*.pth", "*.pt", "*.ckpt", "*.msgpack", "*.onnx", "*.h5")


def allow_patterns() -> list[str]:
    """File selection for downloads: required formats, no duplicates."""
    return list(ALLOW_PATTERNS)


def filter_files(files: list[str]) -> list[str]:
    """Select the files a download should fetch for the selected profile."""
    selected = []
    for name in files:
        if not any(fnmatch.fnmatch(name, pattern) for pattern in ALLOW_PATTERNS):
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in DISALLOWED_PATTERNS):
            continue
        selected.append(name)
    return selected


def check_compatibility(
    api: Any, repo_id: str, revision: str | None = None
) -> schemas.CompatibilityReport:
    from image_studio.hub.client import translate_hub_error

    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except ImageStudioError:
        raise
    except Exception as exc:
        raise translate_hub_error(exc) from exc
    files = [sibling.rfilename for sibling in getattr(info, "siblings", None) or []]
    commit_sha = getattr(info, "sha", None)
    findings: list[str] = []
    notes: list[str] = [
        "Structural compatibility does not guarantee output quality.",
        "Distillation type is not inferable from the pipeline class; choose "
        "Base or Turbo explicitly.",
    ]

    has_model_index = "model_index.json" in files
    has_safetensors = any(name.endswith(".safetensors") for name in files)
    remote_code_files = [name for name in files if name.endswith(".py")]

    if not has_model_index:
        findings.append("model_index.json is missing; not a Diffusers pipeline repository.")
    if not has_safetensors:
        findings.append("no safetensors weights found.")
    if remote_code_files:
        findings.append("repository ships Python files; remote code execution is not supported.")

    compatible = has_model_index and has_safetensors and not remote_code_files
    if compatible:
        index = _load_model_index(api, repo_id, commit_sha or revision)
        if index is None:
            findings.append("model_index.json could not be read; verification incomplete.")
            compatible = False
        else:
            for problem in _remote_manifest_problems(index, set(files)):
                findings.append(problem)
                compatible = False

    selectable = [ProfileId.Z_IMAGE, ProfileId.Z_IMAGE_TURBO] if compatible else []
    return schemas.CompatibilityReport(
        repo_id=repo_id,
        revision=revision or "main",
        commit_sha=commit_sha,
        structurally_compatible=compatible,
        selectable_profiles=selectable,
        findings=findings,
        notes=notes,
    )


def _remote_manifest_problems(index: object, file_set: set[str]) -> list[str]:
    """Apply the shared component manifest to a remote file listing.

    Declaration rules come from ``hub.cache.manifest_problems`` so local and
    remote validation can never drift; a non-object manifest (JSON list,
    null, ...) is a typed malformed-manifest finding, never an attribute
    error. Shard contents are validated locally after download; remotely it
    is enough that the single weight file or its sharding index is listed.
    """
    declarations = manifest_problems(index)
    details = [problem.detail for problem in declarations]
    if any(problem.code.value == "unsupported_model" for problem in declarations):
        return details
    if not isinstance(index, dict):
        return details
    for component in REQUIRED_COMPONENTS:
        for relative in COMPONENT_CONFIG_FILES.get(component, ()):
            if relative not in file_set:
                details.append(f"{relative} is missing")
        if component in COMPONENT_WEIGHT_FILES:
            single, sharded_index = COMPONENT_WEIGHT_FILES[component]
            if single not in file_set and sharded_index not in file_set:
                details.append(
                    f"{component} weights are missing (expected {single} or {sharded_index})"
                )
    return details


def _load_model_index(api: Any, repo_id: str, revision: str | None) -> dict[str, Any] | None:
    """Fetch and parse model_index.json at a fixed revision.

    ``revision`` is passed as a keyword: the installed SDK declares it
    keyword-only, and this function is pinned against that signature by
    tests using a spec of the real bound method.
    """
    downloader = getattr(api, "hf_hub_download", None)
    if downloader is None:
        return None
    try:
        path = downloader(repo_id, "model_index.json", revision=revision or "main")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None
