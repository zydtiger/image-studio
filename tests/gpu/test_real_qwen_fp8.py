"""Opt-in cached Unsloth Qwen FP8 generation, residency, reuse and Eject."""

import io
import os
from pathlib import Path

import pytest

from image_studio.hub import qwen_fp8 as recipe
from image_studio.inference import list_nvidia_gpus
from image_studio.schemas import (
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    ProfileId,
    WorkerState,
    utc_now,
)
from tests.gpu.test_real_z_image import RecordingSink, make_gpu_runtime, wait_state

pytestmark = pytest.mark.gpu


def test_qwen_fp8_offline_defaults_and_reuse(monkeypatch, tmp_path):
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE
    from PIL import Image, ImageStat

    gpu_uuid = os.environ.get("IMAGE_STUDIO_TEST_GPU")
    if not gpu_uuid:
        pytest.skip("set IMAGE_STUDIO_TEST_GPU to an explicitly selected idle GPU UUID")
    gpu = next((g for g in list_nvidia_gpus() if g.uuid == gpu_uuid), None)
    assert gpu is not None, f"selected GPU is unavailable: {gpu_uuid}"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    candidates = sorted(
        revision.snapshot_path
        for repo in scan_cache_dir(HF_HUB_CACHE).repos
        if repo.repo_id == recipe.REPO
        for revision in repo.revisions
        if (revision.snapshot_path / recipe.TRANSFORMER_FILE).is_file()
    )
    assert candidates, "download Unsloth Qwen-Image-2.1 FP8 through Image Studio first"
    root = candidates[-1]
    profile = ProfileId.QWEN_IMAGE_21
    sources = recipe.make_sources(Path(HF_HUB_CACHE), recipe.REPO, root.name, profile)
    assert not recipe.model_problems(root, recipe.REPO, profile, sources)
    model = FrozenModel(
        registration_id="qwen-fp8",
        repo_id=recipe.REPO,
        commit_sha=root.name,
        profile=profile,
        dtype="bfloat16",
        snapshot_path=str(root),
        sources=sources,
    )
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        for index in range(2):
            spec = FrozenRunSpec(
                run_id=f"qwen-fp8-{index}",
                created_at=utc_now(),
                model=model,
                gpu=FrozenGpu(uuid=gpu.uuid, name=gpu.name),
                prompt="a red sailboat on a blue lake, mountains, sunlight",
                negative_prompt=None,
                width=1024,
                height=1024,
                steps=40,
                guidance=1.0,
                image_count=1,
                seeds=(12345 + index,),
                artifact_ids=(f"image-{index}",),
            )
            runtime.submit(spec)
            terminal = sink.wait_terminal(spec.run_id)
            assert terminal.event == "run_completed", terminal
            wait_state(runtime, WorkerState.IDLE)
            assert runtime.status().resident.repo_id == recipe.REPO
            (event,) = [e for e in sink.of("image_completed") if e.run_id == spec.run_id]
            with Image.open(io.BytesIO(event.png)) as image:
                assert image.size == (1024, 1024)
                assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 5
            (tmp_path / f"{spec.run_id}.png").write_bytes(event.png)
            assert len([e for e in sink.of("run_progress") if e.run_id == spec.run_id]) == 40
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.LOADING) == 1
        runtime.eject()
        wait_state(runtime, WorkerState.UNLOADED)
    finally:
        runtime.shutdown()
