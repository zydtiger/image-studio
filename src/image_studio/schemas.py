"""Shared contract types for the Image Studio implementation.

This module is the single source of truth for the boundary between the
backend owner (configuration, storage, hub, HTTP, CLI) and the inference
owner (profiles, supervisor, worker, adapter), and for the HTTP shapes the
frontend consumes. It is a contract artifact, not a feature implementation.

Import rules for this module:

- Core dependencies only: stdlib plus pydantic. No torch, no diffusers, no
  FastAPI, no huggingface_hub, no filesystem access, no network access.
- ``image_studio.inference`` may import this module; it must never import
  ``image_studio.storage``, ``image_studio.api``, or SQLite directly.
- The backend owns identifiers, timestamps, profile normalization, and seed
  resolution before persistence and before ``Runtime.submit``.

Precise semantics are normative in ``docs/implementation-contract.md``.
"""

from __future__ import annotations

import random
import types
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

MAX_SEED: Final[int] = 0xFFFF_FFFF
MIN_DIMENSION: Final[int] = 256
MAX_DIMENSION: Final[int] = 2048
DIMENSION_MULTIPLE: Final[int] = 16
MIN_IMAGE_COUNT: Final[int] = 1
MAX_IMAGE_COUNT: Final[int] = 4
PROMPT_MAX_LENGTH: Final[int] = 8000
DEFAULT_DTYPE: Final[str] = "bfloat16"


def artifact_id(index: int) -> str:
    """Return the public artifact id for a 1-based image index."""
    return f"image-{index:03d}"


def plan_artifact_ids(count: int) -> tuple[str, ...]:
    """Return the artifact ids for a run producing ``count`` images."""
    return tuple(artifact_id(index) for index in range(1, count + 1))


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class ProfileId(StrEnum):
    """Selectable model profiles. Built-in architecture and distillation profiles."""

    Z_IMAGE = "z-image"
    Z_IMAGE_TURBO = "z-image-turbo"
    ANIMA_TURBO = "anima-turbo"
    ANIMA_29B = "anima-2.9b"
    QWEN_IMAGE_21 = "qwen-image-2.1"


class WorkerState(StrEnum):
    """Lifecycle of the single global inference worker.

    ``unloaded`` is the explicit no-worker state. ``switching`` covers the
    serialized replacement of a worker for a different model or GPU. Every
    state transition is reported through ``WorkerStateChanged``.
    """

    UNLOADED = "unloaded"
    LOADING = "loading"
    IDLE = "idle"
    GENERATING = "generating"
    SWITCHING = "switching"
    EJECTING = "ejecting"


class RunStatus(StrEnum):
    """Task states persisted by the backend.

    ``interrupted`` marks work that was active when the server stopped.
    ``paused`` marks queued work awaiting explicit queue resume after a
    restart. ``partial`` retains completed images with failed/cancelled
    remainder.
    """

    QUEUED = "queued"
    PAUSED = "paused"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ImageStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RegistrationStatus(StrEnum):
    READY = "ready"
    MISSING_FILES = "missing_files"


class ErrorCode(StrEnum):
    """Machine-readable error codes shared by HTTP errors and run records."""

    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    GATED_MODEL = "gated_model"
    HUB_AUTH_REQUIRED = "hub_auth_required"
    HUB_UNREACHABLE = "hub_unreachable"
    REVISION_NOT_FOUND = "revision_not_found"
    CACHE_INCOMPLETE = "cache_incomplete"
    UNSUPPORTED_MODEL = "unsupported_model"
    WORKER_ERROR = "worker_error"
    STORAGE_ERROR = "storage_error"
    INTERNAL = "internal"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class ImageStudioError(Exception):
    """Base typed error. The API layer maps ``code`` to an HTTP status."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class RuntimeConflictError(ImageStudioError):
    """Raised by ``Runtime.eject`` (and busy-state actions) as HTTP 409."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CONFLICT, message, details)


class ContractViolationError(ImageStudioError):
    """Raised by contract helpers for invalid generation parameters."""

    def __init__(self, message: str, reason: str, details: dict[str, Any] | None = None) -> None:
        merged = {"reason": reason, **(details or {})}
        super().__init__(ErrorCode.VALIDATION, message, merged)


# --------------------------------------------------------------------------
# Profile metadata: the single truth for validation and UI rendering
# --------------------------------------------------------------------------


class ProfileSpec(BaseModel):
    """Declarative parameter rules for one profile.

    ``guidance_fixed`` is ``None`` when the user may adjust guidance; a value
    means guidance is locked and the field is hidden. These constants are the
    single truth: ``image_studio.inference.profiles`` consumes them and the
    HTTP layer serves them as capability metadata (``GET /api/profiles``).
    """

    model_config = ConfigDict(frozen=True)

    profile_id: ProfileId
    label: str
    default_steps: int
    min_steps: int
    max_steps: int
    guidance_default: float
    guidance_fixed: float | None
    negative_prompt_supported: bool
    default_width: int
    default_height: int
    dimension_multiple: int = DIMENSION_MULTIPLE
    dtype: str


PROFILES: Final[Mapping[ProfileId, ProfileSpec]] = types.MappingProxyType(
    {
        ProfileId.QWEN_IMAGE_21: ProfileSpec(
            profile_id=ProfileId.QWEN_IMAGE_21,
            label="Qwen-Image-2.1",
            default_steps=40,
            min_steps=1,
            max_steps=100,
            guidance_default=1.0,
            guidance_fixed=1.0,
            negative_prompt_supported=False,
            default_width=1024,
            default_height=1024,
            dimension_multiple=32,
            dtype=DEFAULT_DTYPE,
        ),
        ProfileId.ANIMA_TURBO: ProfileSpec(
            profile_id=ProfileId.ANIMA_TURBO,
            label="Anima-Turbo",
            default_steps=10,
            min_steps=1,
            max_steps=100,
            guidance_default=1.0,
            guidance_fixed=1.0,
            negative_prompt_supported=False,
            default_width=1024,
            default_height=1024,
            dtype=DEFAULT_DTYPE,
        ),
        ProfileId.ANIMA_29B: ProfileSpec(
            profile_id=ProfileId.ANIMA_29B,
            label="Anima 2.9B",
            default_steps=40,
            min_steps=1,
            max_steps=100,
            guidance_default=4.0,
            guidance_fixed=None,
            negative_prompt_supported=True,
            default_width=1024,
            default_height=1024,
            dtype=DEFAULT_DTYPE,
        ),
        ProfileId.Z_IMAGE: ProfileSpec(
            profile_id=ProfileId.Z_IMAGE,
            label="Z-Image",
            default_steps=50,
            min_steps=1,
            max_steps=100,
            guidance_default=4.0,
            guidance_fixed=None,
            negative_prompt_supported=True,
            default_width=1024,
            default_height=1024,
            dtype=DEFAULT_DTYPE,
        ),
        ProfileId.Z_IMAGE_TURBO: ProfileSpec(
            profile_id=ProfileId.Z_IMAGE_TURBO,
            label="Z-Image-Turbo",
            default_steps=9,
            min_steps=1,
            max_steps=100,
            guidance_default=0.0,
            guidance_fixed=0.0,
            negative_prompt_supported=False,
            default_width=1024,
            default_height=1024,
            dtype=DEFAULT_DTYPE,
        ),
    }
)


# --------------------------------------------------------------------------
# Generation requests and backend-owned normalization
# --------------------------------------------------------------------------


class GenerationRequest(BaseModel):
    """Raw submission body for ``POST /api/generations``.

    Profile-dependent rules (locked guidance, hidden negative prompt) are
    enforced by :func:`validate_generation` once the registration's profile
    is known; unsupported inputs are rejected, never silently ignored.
    Effective defaults are applied during freeze, not here.
    """

    registration_id: str = Field(min_length=1)
    gpu_uuid: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=PROMPT_MAX_LENGTH)
    negative_prompt: str | None = Field(default=None, max_length=PROMPT_MAX_LENGTH)
    width: int = Field(ge=MIN_DIMENSION, le=MAX_DIMENSION, multiple_of=DIMENSION_MULTIPLE)
    height: int = Field(ge=MIN_DIMENSION, le=MAX_DIMENSION, multiple_of=DIMENSION_MULTIPLE)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    count: int = Field(default=MIN_IMAGE_COUNT, ge=MIN_IMAGE_COUNT, le=MAX_IMAGE_COUNT)


def validate_generation(request: GenerationRequest, profile: ProfileSpec) -> None:
    """Enforce profile-dependent rules; raise ``ContractViolationError``.

    Seed overflow rule: seeds within a run increase by one per image and are
    never allowed to wrap past ``MAX_SEED``. An explicit seed whose run would
    exceed ``MAX_SEED`` (``seed + count - 1 > MAX_SEED``) is rejected with
    reason ``seed_overflow``; randomly resolved seeds are drawn from a range
    that makes wrap-around impossible.
    """

    for field in ("width", "height"):
        if getattr(request, field) % profile.dimension_multiple:
            raise ContractViolationError(
                f"{field} must be a multiple of {profile.dimension_multiple} "
                f"for {profile.profile_id.value}",
                reason="dimension_multiple",
                details={"field": field, "profile": profile.profile_id.value},
            )
    if request.steps is not None and not (profile.min_steps <= request.steps <= profile.max_steps):
        raise ContractViolationError(
            f"steps must be {profile.min_steps}-{profile.max_steps} for {profile.profile_id.value}",
            reason="steps_out_of_range",
            details={"profile": profile.profile_id.value},
        )
    if profile.guidance_fixed is not None:
        if request.guidance is not None and request.guidance != profile.guidance_fixed:
            raise ContractViolationError(
                f"guidance is fixed at {profile.guidance_fixed} for "
                f"{profile.profile_id.value}; submit without guidance",
                reason="unsupported_field",
                details={"field": "guidance", "profile": profile.profile_id.value},
            )
        if request.negative_prompt:
            raise ContractViolationError(
                f"negative_prompt is not supported by {profile.profile_id.value}",
                reason="unsupported_field",
                details={"field": "negative_prompt", "profile": profile.profile_id.value},
            )
    if request.seed is not None and request.seed + request.count - 1 > MAX_SEED:
        raise ContractViolationError(
            f"seed {request.seed} with count {request.count} would exceed "
            f"MAX_SEED {MAX_SEED}; seeds never wrap",
            reason="seed_overflow",
            details={"seed": request.seed, "count": request.count, "max_seed": MAX_SEED},
        )


_SYSTEM_RANDOM: Final[random.Random] = random.SystemRandom()


def resolve_seeds(
    seed: int | None, count: int, rng: random.Random | None = None
) -> tuple[int, ...]:
    """Resolve the per-image seed sequence before persistence and submit.

    An explicit ``seed`` starts the increasing sequence. ``None`` draws a
    random first seed from ``[0, MAX_SEED - count + 1]`` so the sequence can
    never wrap. The returned tuple is the frozen truth; the runtime performs
    no seed arithmetic.
    """

    if not MIN_IMAGE_COUNT <= count <= MAX_IMAGE_COUNT:
        raise ContractViolationError(
            f"count must be {MIN_IMAGE_COUNT}-{MAX_IMAGE_COUNT}", reason="count_out_of_range"
        )
    if seed is None:
        generator = rng if rng is not None else _SYSTEM_RANDOM
        first = generator.randrange(0, MAX_SEED - count + 2)
    else:
        if not 0 <= seed <= MAX_SEED:
            raise ContractViolationError(f"seed must be 0-{MAX_SEED}", reason="seed_out_of_range")
        if seed + count - 1 > MAX_SEED:
            raise ContractViolationError(
                f"seed {seed} with count {count} would exceed MAX_SEED "
                f"{MAX_SEED}; seeds never wrap",
                reason="seed_overflow",
                details={"seed": seed, "count": count, "max_seed": MAX_SEED},
            )
        first = seed
    return tuple(first + offset for offset in range(count))


# --------------------------------------------------------------------------
# Frozen run specification: the only object crossing into the runtime
# --------------------------------------------------------------------------


class ModelSource(BaseModel):
    """Fixed Hub file selection with a runtime-only, environment-resolved path.

    Persistence keeps repo_id, commit_sha and files; snapshot_path is attached
    when reading in the current cache environment and is never stored.
    """

    model_config = ConfigDict(frozen=True)

    repo_id: str
    commit_sha: str
    files: tuple[str, ...]
    snapshot_path: str


class FrozenModel(BaseModel):
    """Resolved model identity for one run.

    ``snapshot_path`` is the absolute, verified local snapshot directory in
    the Hugging Face cache. Generation loads from it with
    ``local_files_only=True``; the runtime never re-resolves repo/revision
    against the Hub.
    """

    model_config = ConfigDict(frozen=True)

    registration_id: str
    repo_id: str
    commit_sha: str
    profile: ProfileId
    sources: tuple[ModelSource, ...] = ()
    dtype: str
    snapshot_path: str


class FrozenGpu(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    name: str


class FrozenRunSpec(BaseModel):
    """Everything a run needs, frozen by the backend at submission.

    Created before persistence: the run and image rows and the dispatch call
    both derive from this object. Effective parameters are explicit; the
    runtime must not recompute defaults. The seed and artifact-id sequences
    are immutable tuples so a frozen handoff cannot be mutated in place.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: datetime
    model: FrozenModel
    gpu: FrozenGpu
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance: float
    image_count: int
    seeds: tuple[int, ...]
    artifact_ids: tuple[str, ...]


# --------------------------------------------------------------------------
# Events: runtime -> backend reporting (Python objects, not HTTP payloads)
# --------------------------------------------------------------------------


class ResidentModel(BaseModel):
    """Identity and placement of the model currently held by the worker.

    ``gpu`` is required: once a run finishes and ``current_run_id`` clears,
    clients still learn the resident GPU for display, request defaults, and
    detecting a same-model cross-GPU switch. ``pipeline_class`` and
    ``dependency_versions`` are the worker-reported runtime identity,
    optional so older clients and fakes stay compatible.
    """

    registration_id: str
    repo_id: str
    commit_sha: str
    profile: ProfileId
    dtype: str
    gpu: FrozenGpu
    pipeline_class: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)


class WorkerStateChanged(BaseModel):
    event: Literal["worker_state_changed"] = "worker_state_changed"
    state: WorkerState
    resident: ResidentModel | None = None
    reason: str | None = None
    error: ErrorInfo | None = None


class RunStarted(BaseModel):
    """Dispatch began for one run.

    ``pipeline_class`` and ``dependency_versions`` carry the serving
    worker's reported runtime identity so every run — including runs on a
    reused worker — records the stack that produced it. Optional with
    empty defaults for compatibility.
    """

    event: Literal["run_started"] = "run_started"
    run_id: str
    started_at: datetime
    pipeline_class: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)


class RunProgress(BaseModel):
    event: Literal["run_progress"] = "run_progress"
    run_id: str
    image_index: int = Field(ge=1)
    step: int = Field(ge=0)
    total_steps: int = Field(ge=1)


class ImageCompleted(BaseModel):
    """One finished image. ``png`` carries the encoded PNG bytes."""

    event: Literal["image_completed"] = "image_completed"
    run_id: str
    artifact_id: str
    index: int = Field(ge=1)
    seed: int = Field(ge=0, le=MAX_SEED)
    width: int
    height: int
    png: bytes


class RunCompleted(BaseModel):
    event: Literal["run_completed"] = "run_completed"
    run_id: str
    completed_count: int = Field(ge=0)
    finished_at: datetime


class RunFailed(BaseModel):
    event: Literal["run_failed"] = "run_failed"
    run_id: str
    error: ErrorInfo
    completed_count: int = Field(ge=0)
    finished_at: datetime


class RunCancelled(BaseModel):
    event: Literal["run_cancelled"] = "run_cancelled"
    run_id: str
    completed_count: int = Field(ge=0)
    finished_at: datetime


type RuntimeEvent = Annotated[
    WorkerStateChanged
    | RunStarted
    | RunProgress
    | ImageCompleted
    | RunCompleted
    | RunFailed
    | RunCancelled,
    Field(discriminator="event"),
]


# --------------------------------------------------------------------------
# Runtime protocol: implemented by the inference owner, consumed by the API
# --------------------------------------------------------------------------


class RuntimeStatus(BaseModel):
    """Snapshot of the single global worker.

    ``implementation`` visibly identifies a fake runtime so no stub can
    masquerade as a complete inference stack; production defaults to
    ``"real"``.
    """

    implementation: Literal["real", "fake"]
    state: WorkerState
    resident: ResidentModel | None = None
    current_run_id: str | None = None
    queue_depth: int = Field(ge=0)
    last_error: ErrorInfo | None = None


class CancelOutcome(StrEnum):
    REMOVED_FROM_QUEUE = "removed_from_queue"
    CANCELLING = "cancelling"
    ALREADY_FINISHED = "already_finished"


class CancelResult(BaseModel):
    outcome: CancelOutcome


class GpuInfo(BaseModel):
    """One enumerable GPU. UUIDs are stable hardware identities."""

    uuid: str = Field(min_length=1)
    name: str
    index: int = Field(ge=0)
    memory_total_bytes: int | None = None


class EventSink(Protocol):
    """Backend-implemented receiver for runtime events."""

    def on_event(self, event: RuntimeEvent) -> None:
        """Handle one event synchronously; may raise to fail the run."""
        ...


class Runtime(Protocol):
    """Public face of the supervisor. All methods are non-blocking except
    ``eject`` and ``shutdown``. See the contract for full semantics."""

    def attach(self, sink: EventSink) -> None:
        """Register the single event sink; called once before any submit."""
        ...

    def list_gpus(self) -> list[GpuInfo]:
        """Enumerate GPUs without initializing CUDA or claiming devices."""
        ...

    def status(self) -> RuntimeStatus:
        """Return a consistent snapshot for polling."""
        ...

    def submit(self, spec: FrozenRunSpec) -> None:
        """Append to the global FIFO queue; never loads a model inline."""
        ...

    def cancel(self, run_id: str) -> CancelResult:
        """Cancel queued work immediately or request cooperative cancel."""
        ...

    def eject(self) -> None:
        """Stop an idle worker and confirm release.

        Raises ``RuntimeConflictError`` unless the worker is idle; eject and
        task dispatch are serialized under the same supervisor lock.
        """
        ...

    def shutdown(self) -> None:
        """Stop any worker and confirm exit; used on server shutdown."""
        ...


# --------------------------------------------------------------------------
# Hub, cache, registration, and download shapes (HTTP)
# --------------------------------------------------------------------------


class HubModelSummary(BaseModel):
    repo_id: str
    author: str | None = None
    private: bool = False
    gated: bool = False
    downloads: int | None = None
    likes: int | None = None
    last_modified: datetime | None = None
    pipeline_tag: str | None = None
    license: str | None = None


class HubFileEntry(BaseModel):
    path: str
    size: int | None = None


class HubRevision(BaseModel):
    revision: str
    commit_sha: str


class HubModelDetail(BaseModel):
    repo_id: str
    author: str | None = None
    private: bool = False
    gated: bool = False
    downloads: int | None = None
    likes: int | None = None
    last_modified: datetime | None = None
    pipeline_tag: str | None = None
    license: str | None = None
    default_revision: str | None = None
    revisions: list[HubRevision] = Field(default_factory=list)
    files: list[HubFileEntry] = Field(default_factory=list)


class CompatibilityReport(BaseModel):
    """Structural compatibility verdict for one repo revision.

    Structural compatibility never implies output quality; the distillation
    profile is an explicit user choice, not inferred from the class.
    """

    repo_id: str
    revision: str
    commit_sha: str | None = None
    structurally_compatible: bool
    selectable_profiles: list[ProfileId] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CachedSnapshot(BaseModel):
    commit_sha: str | None = None
    path: str
    size_bytes: int | None = None
    incomplete: bool = False


class CachedRepo(BaseModel):
    """One cached repository from ``scan_cache_dir()``, including
    unsupported and incomplete snapshots."""

    repo_id: str
    size_on_disk_bytes: int | None = None
    refs: list[str] = Field(default_factory=list)
    snapshots: list[CachedSnapshot] = Field(default_factory=list)


class ModelRegistration(BaseModel):
    id: str
    repo_id: str
    commit_sha: str
    profile: ProfileId
    sources: tuple[ModelSource, ...] = ()
    display_name: str | None = None
    status: RegistrationStatus
    missing_files: list[str] = Field(default_factory=list)
    snapshot_path: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None


class RegistrationCreate(BaseModel):
    """Register an existing snapshot; never triggers a download."""

    repo_id: str = Field(min_length=1)
    revision: str | None = None
    profile: ProfileId
    display_name: str | None = None


class RegistrationUpdate(BaseModel):
    display_name: str | None = None
    profile: ProfileId | None = None


class DownloadCreate(BaseModel):
    repo_id: str = Field(min_length=1)
    revision: str | None = None
    profile: ProfileId


class DownloadProgress(BaseModel):
    bytes_done: int = Field(ge=0)
    bytes_total: int | None = None
    files_done: int = Field(ge=0)
    files_total: int | None = None


class DownloadJob(BaseModel):
    id: str
    repo_id: str
    requested_revision: str | None
    resolved_commit: str | None
    profile: ProfileId
    sources: tuple[ModelSource, ...] = ()
    status: DownloadStatus
    error: ErrorInfo | None = None
    progress: DownloadProgress
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


# --------------------------------------------------------------------------
# Run responses (HTTP)
# --------------------------------------------------------------------------


class ArtifactView(BaseModel):
    """Per-image record with explicit same-origin URLs.

    URL templates: ``/api/generations/{run_id}/artifacts/{artifact_id}``
    and the ``/thumbnail`` suffix. ``?download=1`` on the artifact URL sets
    a ``Content-Disposition`` attachment header.
    """

    artifact_id: str
    index: int = Field(ge=1)
    seed: int = Field(ge=0, le=MAX_SEED)
    status: ImageStatus
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    error: str | None = None
    url: str | None = None
    thumbnail_url: str | None = None


class RunSummary(BaseModel):
    run_id: str
    created_at: datetime
    status: RunStatus
    favorite: bool = False
    trashed: bool = False
    prompt: str
    negative_prompt: str | None = None
    repo_id: str
    profile: ProfileId
    image_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    preview_artifact_id: str | None = None


class RunProgressSnapshot(BaseModel):
    image_index: int = Field(ge=1)
    step: int = Field(ge=0)
    total_steps: int = Field(ge=1)


class RunDetail(BaseModel):
    run_id: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: RunStatus
    favorite: bool = False
    trashed: bool = False
    registration_id: str
    repo_id: str
    commit_sha: str
    profile: ProfileId
    sources: tuple[ModelSource, ...] = ()
    dtype: str
    gpu: FrozenGpu | None = None
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance: float
    initial_seed: int = Field(ge=0, le=MAX_SEED)
    image_count: int = Field(ge=1)
    pipeline_class: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    runtime_meta: dict[str, Any] = Field(default_factory=dict)
    queue_position: int | None = None
    progress: RunProgressSnapshot | None = None
    error: ErrorInfo | None = None
    images: list[ArtifactView] = Field(default_factory=list)


class FavoriteUpdate(BaseModel):
    favorite: bool


class QueueState(BaseModel):
    paused: bool
    pending: list[RunSummary] = Field(default_factory=list)


# --------------------------------------------------------------------------
# System shapes (HTTP)
# --------------------------------------------------------------------------


class SystemPaths(BaseModel):
    config_file: str
    data_dir: str
    database_file: str
    outputs_dir: str
    trash_dir: str
    thumbnails_dir: str
    log_file: str
    hub_cache_dir: str


class DevelopmentFlags(BaseModel):
    """Explicitly enabled development modes. Always false in production."""

    fake_runtime: bool = False
    fake_hub: bool = False


class SystemInfo(BaseModel):
    host: str
    port: int
    paths: SystemPaths
    hf_logged_in: bool = False
    hf_username: str | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    development: DevelopmentFlags = Field(default_factory=DevelopmentFlags)


class ProfilesResponse(BaseModel):
    """Capability metadata driving form rendering."""

    profiles: list[ProfileSpec]


class ErrorBody(BaseModel):
    """Envelope for every non-2xx API response."""

    error: ErrorInfo


__all__ = [
    "ArtifactView",
    "CancelOutcome",
    "CancelResult",
    "CachedRepo",
    "CachedSnapshot",
    "CompatibilityReport",
    "ContractViolationError",
    "DEFAULT_DTYPE",
    "DIMENSION_MULTIPLE",
    "DevelopmentFlags",
    "DownloadCreate",
    "DownloadJob",
    "DownloadProgress",
    "DownloadStatus",
    "ErrorBody",
    "ErrorCode",
    "ErrorInfo",
    "EventSink",
    "FrozenGpu",
    "FrozenModel",
    "FrozenRunSpec",
    "GenerationRequest",
    "GpuInfo",
    "HubFileEntry",
    "HubModelDetail",
    "HubModelSummary",
    "HubRevision",
    "ImageCompleted",
    "ImageStatus",
    "ImageStudioError",
    "MAX_DIMENSION",
    "MAX_IMAGE_COUNT",
    "MAX_SEED",
    "MIN_DIMENSION",
    "MIN_IMAGE_COUNT",
    "ModelRegistration",
    "PROMPT_MAX_LENGTH",
    "PROFILES",
    "ProfileId",
    "ProfileSpec",
    "ProfilesResponse",
    "QueueState",
    "RegistrationCreate",
    "RegistrationStatus",
    "RegistrationUpdate",
    "ResidentModel",
    "RunCancelled",
    "RunCompleted",
    "RunDetail",
    "RunFailed",
    "RunProgress",
    "RunProgressSnapshot",
    "RunStarted",
    "RunStatus",
    "RunSummary",
    "Runtime",
    "RuntimeConflictError",
    "RuntimeEvent",
    "RuntimeStatus",
    "SystemInfo",
    "SystemPaths",
    "WorkerStateChanged",
    "WorkerState",
    "artifact_id",
    "plan_artifact_ids",
    "resolve_seeds",
    "validate_generation",
]


def utc_now() -> datetime:
    """Timezone-aware UTC now; used for backend-owned timestamps."""
    return datetime.now(UTC)
