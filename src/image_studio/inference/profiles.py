"""Profile consumption for the inference stack.

``schemas.PROFILES`` is the single truth for defaults, bounds, and field
visibility; this module only adapts it for inference-internal use and adds
the worker identity rule. Product-rule validation happens in the backend
before ``Runtime.submit`` — the runtime never re-validates frozen specs.
"""

from __future__ import annotations

from image_studio.schemas import (
    DEFAULT_DTYPE,
    PROFILES,
    FrozenRunSpec,
    ModelSource,
    ProfileId,
    ProfileSpec,
)

#: Default Hugging Face repositories per profile (documentation and test
#: convenience; the runtime itself only ever uses the backend-verified
#: ``snapshot_path`` from a ``FrozenRunSpec``).
DEFAULT_REPOSITORIES: dict[ProfileId, str] = {
    ProfileId.QWEN_IMAGE_21: "Qwen/Qwen-Image-2.1",
    ProfileId.ANIMA_TURBO: "circlestone-labs/Anima",
    ProfileId.ANIMA_29B: "Gazingstars123/Anima-2.9B",
    ProfileId.Z_IMAGE: "Tongyi-MAI/Z-Image",
    ProfileId.Z_IMAGE_TURBO: "Tongyi-MAI/Z-Image-Turbo",
}

#: dtypes the worker can map onto ``torch`` tensor dtypes.
SUPPORTED_DTYPES: frozenset[str] = frozenset({"bfloat16", "float16"})

#: Worker reuse key: identical repo id, commit, profile, dtype, and GPU.
#: Any difference is a full worker replacement (contract section 3).
WorkerIdentity = tuple[str, str, ProfileId, str, str, tuple[ModelSource, ...]]


def get_profile(profile_id: ProfileId) -> ProfileSpec:
    """Return the profile spec from the shared contract."""
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise ValueError(f"unknown profile: {profile_id!r}") from None


def worker_identity(spec: FrozenRunSpec) -> WorkerIdentity:
    """Return the reuse key a worker must match to serve ``spec``."""
    return (
        spec.model.repo_id,
        spec.model.commit_sha,
        spec.model.profile,
        spec.model.dtype,
        spec.gpu.uuid,
        spec.model.sources,
    )


def default_dtype() -> str:
    return DEFAULT_DTYPE
