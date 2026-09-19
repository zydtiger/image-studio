"""Internal message protocol between the supervisor and the worker process.

Everything in this module is stdlib-only (plus ``schemas`` types that cross
the boundary as payloads) and picklable under the ``spawn`` start method.
The parent side never imports torch or diffusers; the worker imports its
heavy stack lazily inside the child process only.

Command flow (parent -> child, over a ``multiprocessing.Pipe``):

- ``CmdRun`` — execute one frozen run; the worker reports per-step progress,
  per-image results, and one ``MsgRunFinished`` terminal.
- ``CmdCancel`` — cooperatively cancel the named run at the next safe
  inference boundary (step callback or between images). The model stays
  resident in a healthy worker.
- ``CmdImageAck`` — acknowledgement for one delivered ``MsgImageCompleted``.
  The worker blocks after reporting an image until this arrives, which is
  the backpressure that guarantees durable persistence of one image
  completes before further inference proceeds. ``ok=False`` instructs the
  worker to abort the remaining images of the run.
- ``CmdShutdown`` — stop serving, release the model, and exit.

Event flow (child -> parent, over a ``multiprocessing.Queue``): the worker
is the only writer. After ``MsgReady`` it serves commands until shutdown,
a fatal fault, or parent-process death (orphan prevention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from image_studio.schemas import ErrorInfo, ModelSource, ProfileId

# ---------------------------------------------------------------------------
# Launch payload (supervisor -> child process, at spawn time)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeScript:
    """Test hooks consumed by :mod:`image_studio.inference.fake`.

    Production launches leave this ``None``. The fields script deterministic
    behavior for spawn-process tests: load latency, per-step latency, fault
    injection, and an optional JSON-lines trace file used by tests to assert
    cross-process ordering (for example that an old worker fully exited
    before its replacement started loading).
    """

    load_delay: float = 0.05
    step_delay: float = 0.01
    fail_load: bool = False
    #: 1-based image index at which the worker hard-exits (simulated crash).
    crash_during: int | None = None
    #: 1-based image index at which the worker reports a fatal OOM-style fault.
    fault_during: int | None = None
    #: 1-based run ordinal (within one worker process) that hard-exits.
    crash_on_run: int | None = None
    #: 1-based run ordinal (within one worker process) that reports a fault.
    fault_on_run: int | None = None
    trace_path: str | None = None


@dataclass(frozen=True)
class WorkerLaunch:
    """Immutable identity and placement of one worker process.

    The reuse key is ``(repo_id, commit_sha, profile, dtype, gpu_uuid)``:
    any change is a full worker replacement, never an in-place mutation.
    """

    registration_id: str
    repo_id: str
    commit_sha: str
    profile: ProfileId
    dtype: str
    snapshot_path: str
    gpu_uuid: str
    gpu_name: str
    fake_script: FakeScript | None = None
    sources: tuple[ModelSource, ...] = ()

    @property
    def identity(self) -> tuple[str, str, ProfileId, str, str, tuple[ModelSource, ...]]:
        return (
            self.repo_id,
            self.commit_sha,
            self.profile,
            self.dtype,
            self.gpu_uuid,
            self.sources,
        )


@dataclass(frozen=True)
class RunTask:
    """Per-run execution payload derived from a ``FrozenRunSpec``.

    Effective parameters are already frozen by the backend; the worker
    recomputes nothing. ``seeds`` and ``artifact_ids`` are parallel tuples,
    one entry per image, in generation order.
    """

    run_id: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance: float
    seeds: tuple[int, ...]
    artifact_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Commands (parent -> child)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmdRun:
    task: RunTask


@dataclass(frozen=True)
class CmdCancel:
    run_id: str


@dataclass(frozen=True)
class CmdImageAck:
    run_id: str
    ok: bool


@dataclass(frozen=True)
class CmdShutdown:
    reason: str


# ---------------------------------------------------------------------------
# Events (child -> parent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MsgReady:
    """The worker finished loading and is resident on its selected device."""

    pipeline_class: str
    device_name: str
    device_uuid: str | None
    uuid_verified: bool
    versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MsgProgress:
    run_id: str
    image_index: int
    step: int
    total_steps: int


@dataclass(frozen=True)
class MsgImageCompleted:
    run_id: str
    artifact_id: str
    index: int
    seed: int
    width: int
    height: int
    png: bytes


@dataclass(frozen=True)
class MsgRunFinished:
    """Internal terminal report for one run.

    ``status`` is the worker's view: ``completed``, ``cancelled`` (user
    cancel honored at a safe boundary), ``aborted`` (persistence rejection
    via ``CmdImageAck.ok=False``, or a supervisor-directed abort), or
    ``failed`` (worker-side error). The supervisor owns the public terminal
    event and never emits two terminals for one run.
    """

    run_id: str
    status: str
    completed_count: int
    error: ErrorInfo | None = None


@dataclass(frozen=True)
class MsgFault:
    """Fatal worker-side fault (load failure, OOM, CUDA error).

    The parent fails the active run (if any), cleans up this process only,
    and transitions to ``unloaded`` with ``last_error`` recorded.
    """

    error: ErrorInfo
    stage: str
    run_id: str | None = None
