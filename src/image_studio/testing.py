"""Backend-owned fakes for development and CPU-only tests.

``FakeRuntime`` implements ``schemas.Runtime`` with deterministic timing,
failure injection, and seed-derived placeholder PNGs (Pillow only, no
inference stack). ``FakeHub`` provides a recorded Hub API plus an on-disk
Hugging Face cache layout under a caller-owned root so scan, registration,
and download flows run without network access. These fakes are development
and testing affordances only; production defaults never use them.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from image_studio import schemas
from image_studio.schemas import (
    CancelOutcome,
    CancelResult,
    ErrorCode,
    ErrorInfo,
    FrozenRunSpec,
    GpuInfo,
    ResidentModel,
    RuntimeConflictError,
    RuntimeEvent,
    RuntimeStatus,
    WorkerState,
)

FAKE_GPUS = (
    GpuInfo(uuid="GPU-fake-0001", name="Fake GPU A", index=0, memory_total_bytes=8 * 1024**3),
    GpuInfo(uuid="GPU-fake-0002", name="Fake GPU B", index=1, memory_total_bytes=8 * 1024**3),
)


def seed_png(seed: int, width: int, height: int) -> bytes:
    """Deterministic placeholder PNG derived from the seed."""
    import io

    from PIL import Image

    color = ((seed >> 16) & 0xFF, (seed >> 8) & 0xFF, seed & 0xFF)
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeRuntime:
    """Deterministic ``schemas.Runtime`` double.

    Events are delivered synchronously from one dispatcher thread in per-run
    order. ``image_delay`` keeps a run in flight long enough to exercise
    cooperative cancellation; ``fail_image_at`` fails that 1-based image and
    ends the run; ``crash_worker`` additionally unloads the worker as if the
    process died. ``gate_after_start`` blocks a run right after its start
    event; on shutdown the run is abandoned without a terminal event, which
    leaves the persisted status ``running`` so restart reconciliation marks
    it ``interrupted``.
    """

    def __init__(
        self,
        *,
        image_delay: float = 0.0,
        fail_image_at: int | None = None,
        crash_worker: bool = False,
        gate_after_start: bool = False,
    ) -> None:
        self.image_delay = image_delay
        self.fail_image_at = fail_image_at
        self.crash_worker = crash_worker
        self._gate_after_start = gate_after_start
        self._gate = threading.Event()
        self._sink: Any = None
        self._lock = threading.Lock()
        self._queue: deque[FrozenRunSpec] = deque()
        self._current: FrozenRunSpec | None = None
        self._cancel_requested: set[str] = set()
        self._state = WorkerState.UNLOADED
        self._resident: ResidentModel | None = None
        self._last_error: ErrorInfo | None = None
        self._stop = threading.Event()
        self._submitted = threading.Event()
        self._thread: threading.Thread | None = None

    # ----- lifecycle ----------------------------------------------------------

    def attach(self, sink: Any) -> None:
        self._sink = sink

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._dispatch_loop, name="fake-runtime", daemon=True
            )
            self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._submitted.set()
        self._gate.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ----- schemas.Runtime ------------------------------------------------------

    def list_gpus(self) -> list[GpuInfo]:
        return list(FAKE_GPUS)

    def status(self) -> RuntimeStatus:
        with self._lock:
            return RuntimeStatus(
                implementation="fake",
                state=self._state,
                resident=self._resident,
                current_run_id=self._current.run_id if self._current else None,
                queue_depth=len(self._queue),
                last_error=self._last_error,
            )

    def submit(self, spec: FrozenRunSpec) -> None:
        with self._lock:
            self._queue.append(spec)
        self.start()
        self._submitted.set()

    def cancel(self, run_id: str) -> CancelResult:
        with self._lock:
            for spec in self._queue:
                if spec.run_id == run_id:
                    self._queue.remove(spec)
                    return CancelResult(outcome=CancelOutcome.REMOVED_FROM_QUEUE)
            if self._current is not None and self._current.run_id == run_id:
                self._cancel_requested.add(run_id)
                return CancelResult(outcome=CancelOutcome.CANCELLING)
        return CancelResult(outcome=CancelOutcome.ALREADY_FINISHED)

    def eject(self) -> None:
        # Mirrors the real supervisor: eject is honored from idle (and is an
        # idempotent no-op from unloaded); busy states conflict.
        with self._lock:
            if self._state not in (WorkerState.IDLE, WorkerState.UNLOADED):
                raise RuntimeConflictError(f"worker is {self._state.value}; eject requires idle")
            if self._resident is not None:
                self._emit(
                    schemas.WorkerStateChanged(state=WorkerState.EJECTING, resident=self._resident)
                )
                self._resident = None
            self._state = WorkerState.UNLOADED
            self._emit(
                schemas.WorkerStateChanged(
                    state=WorkerState.UNLOADED, resident=None, reason="ejected"
                )
            )

    # ----- dispatch ------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                spec = self._queue.popleft() if self._queue else None
                if spec is not None:
                    self._current = spec
            if spec is None:
                self._submitted.wait(timeout=0.05)
                self._submitted.clear()
                continue
            try:
                self._run_one(spec)
            finally:
                with self._lock:
                    self._current = None
                    self._cancel_requested.discard(spec.run_id)

    def _set_state(self, state: WorkerState, *, reason: str | None = None) -> None:
        self._state = state
        self._emit(schemas.WorkerStateChanged(state=state, resident=self._resident, reason=reason))

    def _emit(self, event: RuntimeEvent) -> None:
        if self._sink is not None:
            self._sink.on_event(event)

    def _ensure_resident(self, spec: FrozenRunSpec) -> None:
        target = _resident_from(spec)
        if self._resident == target and self._state in (WorkerState.IDLE, WorkerState.GENERATING):
            return
        if self._resident is not None:
            self._set_state(WorkerState.SWITCHING, reason="model or GPU change")
            self._resident = None
            self._set_state(WorkerState.UNLOADED, reason="previous worker stopped")
        self._set_state(WorkerState.LOADING)
        self._resident = target
        self._set_state(WorkerState.IDLE, reason="model loaded")

    def _run_one(self, spec: FrozenRunSpec) -> None:
        self._ensure_resident(spec)
        with self._lock:
            if spec.run_id in self._cancel_requested:
                self._emit(
                    schemas.RunCancelled(
                        run_id=spec.run_id,
                        completed_count=0,
                        finished_at=datetime.now(UTC),
                    )
                )
                return
        self._set_state(WorkerState.GENERATING)
        self._emit(
            schemas.RunStarted(
                run_id=spec.run_id,
                started_at=datetime.now(UTC),
                pipeline_class=FAKE_PIPELINE_CLASS,
                dependency_versions=dict(FAKE_DEPENDENCY_VERSIONS),
            )
        )
        if self._gate_after_start:
            self._gate.wait()
            if self._stop.is_set():
                return  # abandoned mid-run: no terminal event, status stays running
        completed = 0
        failed = False
        for index, artifact in enumerate(spec.artifact_ids, start=1):
            with self._lock:
                cancelled = spec.run_id in self._cancel_requested
            if cancelled or self._stop.is_set():
                break
            self._emit(
                schemas.RunProgress(
                    run_id=spec.run_id, image_index=index, step=1, total_steps=spec.steps
                )
            )
            if self.image_delay:
                time.sleep(self.image_delay)
            with self._lock:
                cancelled = spec.run_id in self._cancel_requested
            if cancelled or self._stop.is_set():
                break
            if self.fail_image_at == index:
                error = ErrorInfo(
                    code=ErrorCode.WORKER_ERROR,
                    message=f"fake failure injected at image {index}",
                )
                self._last_error = error
                if self.crash_worker:
                    self._resident = None
                    self._set_state(WorkerState.UNLOADED, reason="simulated crash")
                self._emit(
                    schemas.RunFailed(
                        run_id=spec.run_id,
                        error=error,
                        completed_count=completed,
                        finished_at=datetime.now(UTC),
                    )
                )
                failed = True
                break
            seed = spec.seeds[index - 1]
            png = seed_png(seed, spec.width, spec.height)
            self._emit(
                schemas.ImageCompleted(
                    run_id=spec.run_id,
                    artifact_id=artifact,
                    index=index,
                    seed=seed,
                    width=spec.width,
                    height=spec.height,
                    png=png,
                )
            )
            completed += 1
        if self._stop.is_set():
            return  # abandoned mid-run: no terminal event against a closing app
        if failed:
            self._set_state(WorkerState.IDLE, reason="run failed; model stays resident")
            return
        if completed < len(spec.artifact_ids):
            self._emit(
                schemas.RunCancelled(
                    run_id=spec.run_id,
                    completed_count=completed,
                    finished_at=datetime.now(UTC),
                )
            )
        else:
            self._emit(
                schemas.RunCompleted(
                    run_id=spec.run_id,
                    completed_count=completed,
                    finished_at=datetime.now(UTC),
                )
            )
        self._set_state(WorkerState.IDLE)


#: Fake worker identity, mirroring what the inference fake worker process
#: reports in its ready message so both fakes look identical to the sink.
FAKE_PIPELINE_CLASS = "FakeZImagePipeline"
FAKE_DEPENDENCY_VERSIONS = {"fake": "1"}


def _resident_from(spec: FrozenRunSpec) -> ResidentModel:
    return ResidentModel(
        registration_id=spec.model.registration_id,
        repo_id=spec.model.repo_id,
        commit_sha=spec.model.commit_sha,
        profile=spec.model.profile,
        dtype=spec.model.dtype,
        gpu=spec.gpu,
        pipeline_class=FAKE_PIPELINE_CLASS,
        dependency_versions=dict(FAKE_DEPENDENCY_VERSIONS),
    )


# ----- Fake Hub ------------------------------------------------------------------


@dataclass
class FakeRepoSpec:
    repo_id: str
    files: dict[str, str | bytes]
    sha: str
    default_branch: str = "main"
    gated: bool = False
    license: str | None = None
    downloads: int | None = 1234
    likes: int | None = 7
    author: str | None = None


def default_fake_repos() -> list[FakeRepoSpec]:
    return [
        FakeRepoSpec(
            repo_id="Tongyi-MAI/Z-Image",
            files=_z_image_files(),
            sha="a" * 40,
            license="apache-2.0",
            author="Tongyi-MAI",
        ),
        FakeRepoSpec(
            repo_id="Tongyi-MAI/Z-Image-Turbo",
            files=_z_image_files(),
            sha="b" * 40,
            license="apache-2.0",
            author="Tongyi-MAI",
        ),
    ] + anima_fake_repos()


def anima_fake_repos() -> list[FakeRepoSpec]:
    """Original-repo metadata with tiny placeholders, never usable weights."""
    from image_studio.hub.anima import RECIPES, SHARED_COMMIT, SHARED_FILES, SHARED_REPO

    repos = [
        FakeRepoSpec(
            repo_id=recipe.repo_id,
            files={recipe.checkpoint: b"fake-anima"},
            sha=str(i) * 40,
            license="other",
        )
        for i, recipe in enumerate(RECIPES.values(), 1)
    ]
    shared_files = {
        name: "{}" if name.endswith(".json") else b"fake-component" for name in SHARED_FILES
    }
    repos.append(
        FakeRepoSpec(repo_id=SHARED_REPO, files=shared_files, sha=SHARED_COMMIT, license="other")
    )
    return repos


def _z_image_files() -> dict[str, str | bytes]:
    """Default fake catalog mirroring the official Z-Image repo layout.

    Ground truth (metadata only) from Tongyi-MAI/Z-Image and
    Tongyi-MAI/Z-Image-Turbo: sharded text_encoder and transformer weights,
    an unsharded vae, and the Qwen tokenizer files.
    """
    import json

    model_index = {
        "_class_name": "ZImagePipeline",
        "_diffusers_version": "0.40.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    transformer_shards = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    text_encoder_shards = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    files: dict[str, str | bytes] = {
        "model_index.json": json.dumps(model_index),
        "scheduler/scheduler_config.json": "{}",
        "text_encoder/config.json": "{}",
        "text_encoder/generation_config.json": "{}",
        "text_encoder/model.safetensors.index.json": json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {f"t{i}": s for i, s in enumerate(text_encoder_shards)},
            }
        ),
        "tokenizer/tokenizer.json": "{}",
        "tokenizer/tokenizer_config.json": "{}",
        "tokenizer/merges.txt": "",
        "tokenizer/vocab.json": "{}",
        "transformer/config.json": "{}",
        "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {f"w{i}": s for i, s in enumerate(transformer_shards)},
            }
        ),
        "vae/config.json": "{}",
        "vae/diffusion_pytorch_model.safetensors": b"weights-bytes",
    }
    for shard in transformer_shards:
        files[f"transformer/{shard}"] = b"shard-bytes"
    for shard in text_encoder_shards:
        files[f"text_encoder/{shard}"] = b"shard-bytes"
    return files


def _hub_error(kind: str, detail: str) -> Exception:
    """Raise-shaped realistic huggingface_hub error carrying an httpx response.

    The real SDK raises ``HfHubHTTPError`` subclasses, not ``FileNotFoundError``;
    the fake must match so error-translation coverage runs against the same
    shapes production sees.
    """
    import httpx
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    classes = {
        "repository": RepositoryNotFoundError,
        "revision": RevisionNotFoundError,
        "gated": GatedRepoError,
        "auth": HfHubHTTPError,  # 401 surfaces as a plain HTTP error
    }
    status = 401 if kind == "auth" else 403 if kind == "gated" else 404
    request = httpx.Request("GET", f"https://huggingface.co/api/models/{detail}")
    return classes[kind](detail, response=httpx.Response(status, request=request))


class FakeHubApi:
    """Recorded ``HfApi``-shaped object over an in-memory catalog."""

    def __init__(
        self,
        repos: list[FakeRepoSpec],
        *,
        logged_in_as: str | None = "tester",
        downloader_root: Path | None = None,
    ) -> None:
        self._repos = {repo.repo_id: repo for repo in repos}
        self.logged_in_as = logged_in_as
        self._downloader_root = downloader_root
        self.calls: dict[str, list[dict[str, Any]]] = {
            "list_models": [],
            "model_info": [],
            "list_repo_refs": [],
            "whoami": [],
            "hf_hub_download": [],
        }

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls[name].append(kwargs)

    def list_models(self, search: str | None = None, limit: int | None = None) -> Any:
        """Lazy generator, matching the installed SDK's list_models contract:
        auth/network failures surface during iteration, not at call time."""
        self._record("list_models", search=search, limit=limit)
        hits = [
            repo
            for repo in self._repos.values()
            if search is None or search.lower() in repo.repo_id.lower()
        ]
        for repo in hits[: limit or len(hits)]:
            yield self._summary(repo)

    def model_info(
        self, repo_id: str, revision: str | None = None, files_metadata: bool = False
    ) -> Any:
        self._record(
            "model_info", repo_id=repo_id, revision=revision, files_metadata=files_metadata
        )
        repo = self._repos.get(repo_id)
        if repo is None:
            raise _hub_error("repository", repo_id)
        if revision is not None and revision not in (repo.default_branch, repo.sha):
            raise _hub_error("revision", revision)
        siblings = [
            SimpleNamespace(rfilename=name, size=len(_content(repo, name))) for name in repo.files
        ]
        return SimpleNamespace(
            id=repo.repo_id,
            sha=repo.sha,
            siblings=siblings,
            author=repo.author,
            private=False,
            gated=repo.gated or False,
            downloads=repo.downloads,
            likes=repo.likes,
            lastModified=datetime(2026, 1, 1, tzinfo=UTC),
            pipeline_tag="text-to-image",
            card_data=SimpleNamespace(license=repo.license) if repo.license else None,
        )

    def list_repo_refs(self, repo_id: str) -> Any:
        # Real SDK classes: GitRefs/GitRefInfo carry name/ref/target_commit
        # (no sha, no default) — the fake must mirror that exact shape.
        from huggingface_hub.hf_api import GitRefInfo, GitRefs

        self._record("list_repo_refs", repo_id=repo_id)
        repo = self._repos.get(repo_id)
        if repo is None:
            raise _hub_error("repository", repo_id)
        return GitRefs(
            branches=[
                GitRefInfo(
                    name=repo.default_branch,
                    ref=f"refs/heads/{repo.default_branch}",
                    target_commit=repo.sha,
                )
            ],
            tags=[],
            converts=[],
        )

    def whoami(self) -> dict[str, Any]:
        self._record("whoami")
        if self.logged_in_as is None:
            raise RuntimeError("Not logged in")
        return {"name": self.logged_in_as}

    def hf_hub_download(self, repo_id: str, filename: str, *, revision: str | None = None) -> str:
        """Keyword-only ``revision``, matching the installed SDK signature."""
        self._record("hf_hub_download", repo_id=repo_id, filename=filename, revision=revision)
        repo = self._repos.get(repo_id)
        if repo is None:
            raise _hub_error("repository", repo_id)
        if filename not in repo.files:
            raise _hub_error("repository", f"{repo_id}/{filename}")
        if revision is None or revision == repo.default_branch:
            revision = repo.sha
        if revision != repo.sha:
            raise _hub_error("revision", revision)
        assert self._downloader_root is not None
        target = (
            self._downloader_root / _repo_dirname(repo.repo_id) / "snapshots" / repo.sha / filename
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_content(repo, filename))
        return str(target)

    def _summary(self, repo: FakeRepoSpec) -> Any:
        return SimpleNamespace(
            id=repo.repo_id,
            author=repo.author,
            private=False,
            gated=repo.gated or False,
            downloads=repo.downloads,
            likes=repo.likes,
            lastModified=datetime(2026, 1, 1, tzinfo=UTC),
            pipeline_tag="text-to-image",
            card_data=SimpleNamespace(license=repo.license) if repo.license else None,
        )


def _content(repo: FakeRepoSpec, name: str) -> bytes:
    value = repo.files[name]
    return value.encode("utf-8") if isinstance(value, str) else value


def _repo_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


class FakeHub:
    """Bundle of fake Hub API, Hub stack hooks, and an on-disk cache root."""

    def __init__(
        self,
        root: Path,
        *,
        repos: list[FakeRepoSpec] | None = None,
        logged_in_as: str | None = "tester",
    ) -> None:
        self.root = root
        self.cache_dir = root / "hub-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.repos = {
            repo.repo_id: repo for repo in (repos if repos is not None else default_fake_repos())
        }
        self.api = FakeHubApi(
            list(self.repos.values()), logged_in_as=logged_in_as, downloader_root=self.cache_dir
        )

    def stack(self) -> Any:
        from image_studio.hub.client import HubStack

        def lister(repo_id: str, revision: str | None):
            info = self.api.model_info(repo_id, revision=revision, files_metadata=True)
            files = [(s.rfilename, s.size) for s in info.siblings]
            return info.sha, files

        def downloader(repo_id: str, filename: str, revision: str) -> str:
            return self.api.hf_hub_download(repo_id, filename, revision=revision)

        return HubStack(
            self.api, file_lister=lister, downloader=downloader, cache_dir=self.cache_dir
        )

    def seed_snapshot(self, repo_id: str) -> Path:
        """Materialize a complete cached snapshot for offline registration."""
        repo = self.repos[repo_id]
        snapshot_dir = self.cache_dir / _repo_dirname(repo_id) / "snapshots" / repo.sha
        refs_dir = self.cache_dir / _repo_dirname(repo_id) / "refs"
        (self.cache_dir / _repo_dirname(repo_id) / "blobs").mkdir(parents=True, exist_ok=True)
        for name, content in repo.files.items():
            target = snapshot_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / repo.default_branch).write_text(repo.sha)
        return snapshot_dir
