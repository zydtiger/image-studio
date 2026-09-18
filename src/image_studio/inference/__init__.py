"""Model profiles, the global generation queue, and the resident worker.

Public composition surface for the backend: :func:`create_runtime` builds
the real ``schemas.Runtime`` implementation. Importing this package (or
any module in it) never imports torch or diffusers; the heavy inference
stack loads lazily inside the spawned worker process only.
"""

from image_studio.inference.gpus import list_nvidia_gpus, normalize_gpu_uuid
from image_studio.inference.profiles import (
    DEFAULT_REPOSITORIES,
    SUPPORTED_DTYPES,
    get_profile,
    worker_identity,
)
from image_studio.inference.supervisor import InferenceSupervisor, create_runtime

__all__ = [
    "DEFAULT_REPOSITORIES",
    "InferenceSupervisor",
    "SUPPORTED_DTYPES",
    "create_runtime",
    "get_profile",
    "list_nvidia_gpus",
    "normalize_gpu_uuid",
    "worker_identity",
]
