"""Opt-in offline Anima generation; IMAGE_STUDIO_TEST_GPU must name an idle GPU."""

import io
import os
from pathlib import Path

import pytest

from image_studio.hub.anima import RECIPES, make_sources, model_problems
from image_studio.inference import list_nvidia_gpus
from image_studio.schemas import (
    PROFILES,
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    WorkerState,
    utc_now,
)
from tests.gpu.test_real_z_image import RecordingSink, make_gpu_runtime, wait_state

pytestmark = pytest.mark.gpu


def test_anima_defaults_and_model_switch_offline(monkeypatch, tmp_path):
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE
    from PIL import Image, ImageStat

    gpu_uuid = os.environ.get("IMAGE_STUDIO_TEST_GPU")
    if not gpu_uuid:
        pytest.skip("set IMAGE_STUDIO_TEST_GPU to an explicitly selected idle GPU UUID")
    gpu = next((gpu for gpu in list_nvidia_gpus() if gpu.uuid == gpu_uuid), None)
    assert gpu is not None, f"selected GPU is unavailable: {gpu_uuid}"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    cache = scan_cache_dir(HF_HUB_CACHE)
    specs = []
    for profile, recipe in RECIPES.items():
        # Studio downloads fixed commits, which need not create a main ref.
        candidates = sorted(
            revision.snapshot_path
            for repo in cache.repos
            if repo.repo_id == recipe.repo_id
            for revision in repo.revisions
            if (revision.snapshot_path / recipe.checkpoint).is_file()
        )
        assert candidates, f"download {profile.value} through Image Studio first"
        root = candidates[-1]
        sources = make_sources(Path(HF_HUB_CACHE), recipe.repo_id, root.name, profile)
        assert not model_problems(root, recipe.repo_id, profile, sources)
        defaults = PROFILES[profile]
        specs.append(
            FrozenRunSpec(
                run_id=profile.value,
                created_at=utc_now(),
                model=FrozenModel(
                    registration_id=profile.value,
                    repo_id=recipe.repo_id,
                    commit_sha=root.name,
                    profile=profile,
                    dtype=defaults.dtype,
                    snapshot_path=str(root),
                    sources=sources,
                ),
                gpu=FrozenGpu(uuid=gpu.uuid, name=gpu.name),
                prompt="anime illustration, a red sailboat on a blue lake, mountains, sunlight",
                negative_prompt="blurry, text" if defaults.negative_prompt_supported else None,
                width=1024,
                height=1024,
                steps=defaults.default_steps,
                guidance=defaults.guidance_default,
                image_count=1,
                seeds=(12345,),
                artifact_ids=("image-001",),
            )
        )
    sink = RecordingSink()
    runtime = make_gpu_runtime(sink)
    try:
        for spec in specs:
            runtime.submit(spec)
            terminal = sink.wait_terminal(spec.run_id)
            assert terminal.event == "run_completed", terminal
            wait_state(runtime, WorkerState.IDLE)
            assert runtime.status().resident.profile == spec.model.profile
            (event,) = [e for e in sink.of("image_completed") if e.run_id == spec.run_id]
            assert event.seed == 12345
            image = Image.open(io.BytesIO(event.png))
            assert image.format == "PNG" and image.size == (1024, 1024)
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 5
            (tmp_path / f"{spec.run_id}.png").write_bytes(event.png)
            progress = [e for e in sink.of("run_progress") if e.run_id == spec.run_id]
            assert len(progress) == spec.steps
        states = [e.state for e in sink.of("worker_state_changed")]
        assert states.count(WorkerState.SWITCHING) == 1
        runtime.eject()
        wait_state(runtime, WorkerState.UNLOADED)
    finally:
        runtime.shutdown()
