"""Opt-in real-GPU tests for the Z-Image runtime.

Every test is marked ``pytest.mark.gpu`` and deselected by default. They
must only run with explicit authorization, an NVIDIA GPU, and the model
weights already present in the local Hugging Face cache — collection
performs no CUDA work and downloads nothing. A missing cache or driver
yields a skip, never a failure, and ``HF_HUB_OFFLINE=1`` is enforced to
prove generation never needs the network.
"""

from __future__ import annotations

import io
import time

import pytest

from image_studio.inference import create_runtime, list_nvidia_gpus
from image_studio.schemas import (
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    ProfileId,
    RuntimeConflictError,
    WorkerState,
    plan_artifact_ids,
    utc_now,
)

TURBO_REPO = "Tongyi-MAI/Z-Image-Turbo"
BASE_REPO = "Tongyi-MAI/Z-Image"
TERMINAL_EVENTS = ("run_completed", "run_failed", "run_cancelled")

pytestmark = pytest.mark.gpu


def local_snapshot(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - core dependency
        pytest.skip("huggingface_hub unavailable")
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - any miss means "not cached"
        pytest.skip(
            f"weights for {repo_id} are not in the local Hugging Face cache; "
            "download them first (this suite never downloads)"
        )


def first_gpu():
    gpus = list_nvidia_gpus()
    if not gpus:
        pytest.skip("no NVIDIA GPU enumerable via nvidia-smi")
    return gpus[0]


class RecordingSink:
    def __init__(self):
        self.events = []
        self.terminals = {}

    def on_event(self, event) -> None:
        self.events.append(event)
        if event.event in TERMINAL_EVENTS:
            self.terminals[event.run_id] = event

    def wait_terminal(self, run_id, timeout=900.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if run_id in self.terminals:
                return self.terminals[run_id]
            time.sleep(0.05)
        raise AssertionError(f"no terminal for {run_id}; saw {[e.event for e in self.events]}")

    def of(self, *names):
        return [e for e in self.events if e.event in names]


def make_gpu_runtime(sink):
    runtime = create_runtime(
        poll_interval=0.2,
        stop_grace=120.0,
        join_grace=30.0,
        action_timeout=600.0,
    )
    runtime.attach(sink)
    return runtime


def make_spec(
    run_id,
    *,
    repo=TURBO_REPO,
    profile=ProfileId.Z_IMAGE_TURBO,
    gpu_uuid,
    gpu_name,
    count=1,
    steps=2,
    seed=11,
    negative_prompt=None,
    guidance=0.0,
):
    snapshot = local_snapshot(repo)
    return FrozenRunSpec(
        run_id=run_id,
        created_at=utc_now(),
        model=FrozenModel(
            registration_id=f"reg-{repo}",
            repo_id=repo,
            commit_sha="local-snapshot",
            profile=profile,
            dtype="bfloat16",
            snapshot_path=snapshot,
        ),
        gpu=FrozenGpu(uuid=gpu_uuid, name=gpu_name),
        prompt="a small red boat on a calm lake, photorealistic",
        negative_prompt=negative_prompt,
        width=256,
        height=256,
        steps=steps,
        guidance=guidance,
        image_count=count,
        seeds=tuple(seed + offset for offset in range(count)),
        artifact_ids=plan_artifact_ids(count),
    )


def wait_state(runtime, state, timeout=900.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.status().state == state:
            return
        time.sleep(0.05)
    raise AssertionError(f"never reached {state}; now {runtime.status().state}")


def assert_real_png(png: bytes, width: int, height: int) -> None:
    from PIL import Image

    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG"
    assert (image.width, image.height) == (width, height)
    assert len(png) > 5_000, "a real 256x256 diffusion output is never this small"


def test_turbo_generates_real_image(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpu = first_gpu()
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        runtime.submit(make_spec("gpu-run-1", gpu_uuid=gpu.uuid, gpu_name=gpu.name))
        terminal = sink.wait_terminal("gpu-run-1")
        assert terminal.event == "run_completed"
        assert terminal.completed_count == 1

        (image,) = sink.of("image_completed")
        assert image.seed == 11
        assert_real_png(image.png, 256, 256)

        status = runtime.status()
        assert status.state == WorkerState.IDLE
        assert status.resident is not None
        assert status.resident.gpu.uuid == gpu.uuid
        assert status.resident.profile == ProfileId.Z_IMAGE_TURBO
    finally:
        runtime.shutdown()


def test_base_profile_with_negative_prompt(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpu = first_gpu()
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        runtime.submit(
            make_spec(
                "gpu-run-base",
                repo=BASE_REPO,
                profile=ProfileId.Z_IMAGE,
                gpu_uuid=gpu.uuid,
                gpu_name=gpu.name,
                steps=2,
                guidance=4.0,
                negative_prompt="blurry, low quality",
            )
        )
        terminal = sink.wait_terminal("gpu-run-base")
        assert terminal.event == "run_completed"
        (image,) = sink.of("image_completed")
        assert_real_png(image.png, 256, 256)
        assert runtime.status().resident.profile == ProfileId.Z_IMAGE
    finally:
        runtime.shutdown()


def test_same_model_same_gpu_reuses_worker(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpu = first_gpu()
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        for run_id in ("gpu-reuse-1", "gpu-reuse-2"):
            runtime.submit(make_spec(run_id, gpu_uuid=gpu.uuid, gpu_name=gpu.name))
            sink.wait_terminal(run_id)
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 1, "identical runs must reuse"
        assert states.count(WorkerState.SWITCHING) == 0
    finally:
        runtime.shutdown()


def test_profile_switch_is_full_replacement(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpu = first_gpu()
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        runtime.submit(make_spec("gpu-turbo", gpu_uuid=gpu.uuid, gpu_name=gpu.name))
        sink.wait_terminal("gpu-turbo")
        runtime.submit(
            make_spec(
                "gpu-base",
                repo=BASE_REPO,
                profile=ProfileId.Z_IMAGE,
                gpu_uuid=gpu.uuid,
                gpu_name=gpu.name,
                steps=2,
                guidance=4.0,
            )
        )
        sink.wait_terminal("gpu-base")
        switching = [e for e in sink.of("worker_state_changed") if e.state == WorkerState.SWITCHING]
        assert len(switching) == 1
        assert runtime.status().resident.repo_id == BASE_REPO
    finally:
        runtime.shutdown()


def test_cancel_preserves_images_and_keeps_worker(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpu = first_gpu()
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        runtime.submit(
            make_spec(
                "gpu-cancel",
                gpu_uuid=gpu.uuid,
                gpu_name=gpu.name,
                count=3,
                steps=20,
            )
        )
        deadline = time.monotonic() + 900.0
        while not sink.of("image_completed"):
            assert time.monotonic() < deadline, "first image never completed"
            time.sleep(0.05)
        outcome = runtime.cancel("gpu-cancel")
        assert outcome.outcome.value == "cancelling"
        terminal = sink.wait_terminal("gpu-cancel")
        assert terminal.event == "run_cancelled"
        assert terminal.completed_count >= 1

        wait_state(runtime, WorkerState.IDLE)
        assert runtime.status().resident is not None

        runtime.submit(make_spec("gpu-after-cancel", gpu_uuid=gpu.uuid, gpu_name=gpu.name))
        sink.wait_terminal("gpu-after-cancel")
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 1, "worker stays resident"
    finally:
        runtime.shutdown()


def test_eject_idle_unloads_and_busy_conflicts(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpu = first_gpu()
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        runtime.submit(make_spec("gpu-eject-1", gpu_uuid=gpu.uuid, gpu_name=gpu.name))
        sink.wait_terminal("gpu-eject-1")
        wait_state(runtime, WorkerState.IDLE)

        runtime.submit(
            make_spec(
                "gpu-eject-2",
                gpu_uuid=gpu.uuid,
                gpu_name=gpu.name,
                count=2,
                steps=20,
            )
        )
        wait_state(runtime, WorkerState.GENERATING)
        with pytest.raises(RuntimeConflictError):
            runtime.eject()
        runtime.cancel("gpu-eject-2")
        sink.wait_terminal("gpu-eject-2")
        wait_state(runtime, WorkerState.IDLE)

        runtime.eject()
        assert runtime.status().state == WorkerState.UNLOADED
        assert runtime.status().resident is None

        runtime.submit(make_spec("gpu-eject-3", gpu_uuid=gpu.uuid, gpu_name=gpu.name))
        sink.wait_terminal("gpu-eject-3")
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 2
    finally:
        runtime.shutdown()


def test_multi_gpu_replacement_when_two_devices(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    gpus = list_nvidia_gpus()
    if len(gpus) < 2:
        pytest.skip("cross-GPU replacement requires two available devices")
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        first, second = gpus[0], gpus[1]
        runtime.submit(make_spec("gpu-x-1", gpu_uuid=first.uuid, gpu_name=first.name))
        sink.wait_terminal("gpu-x-1")
        runtime.submit(make_spec("gpu-x-2", gpu_uuid=second.uuid, gpu_name=second.name))
        sink.wait_terminal("gpu-x-2")
        switching = [e for e in sink.of("worker_state_changed") if e.state == WorkerState.SWITCHING]
        assert len(switching) == 1
        assert switching[0].reason == "gpu-changed"
        assert runtime.status().resident.gpu.uuid == second.uuid
    finally:
        runtime.shutdown()
