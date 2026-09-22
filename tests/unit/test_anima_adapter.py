"""Adapter mapping and real-stack tiny CPU checks, with no model downloads."""

import importlib.util
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from image_studio.inference import adapters, anima, qwen_image, z_image
from image_studio.inference.z_image import GenerationCancelled
from image_studio.schemas import ProfileId
from tests.unit.test_inference_adapter import FakeGenerator, FakeImage


@pytest.mark.parametrize("layers", [28, 40])
def test_checkpoint_prefixes_and_depth(layers):
    state = {f"blocks.{i}.self_attn.q_proj.weight": object() for i in range(layers)}
    state["llm_adapter.embed.weight"] = object()
    denoiser, conditioner = anima.split_checkpoint(state, layers)
    assert conditioner["embed.weight"] is state["llm_adapter.embed.weight"]
    assert (
        denoiser["net.blocks.0.self_attn.q_proj.weight"]
        is state["blocks.0.self_attn.q_proj.weight"]
    )
    with pytest.raises(ValueError, match="transformer layers"):
        anima.split_checkpoint(state, 40 if layers == 28 else 28)
    del state["llm_adapter.embed.weight"]
    with pytest.raises(ValueError, match="text conditioner"):
        anima.split_checkpoint(state, layers)


@pytest.mark.parametrize("profile", list(ProfileId))
def test_explicit_adapter_selection(profile):
    if profile is ProfileId.QWEN_IMAGE_21:
        expected = qwen_image
    else:
        expected = anima if profile.value.startswith("anima-") else z_image
    assert adapters.for_profile(profile) is expected


@pytest.mark.parametrize("guidance,negative", [(1, None), (4, "blurry")])
def test_generation_mapping_and_callback_cleanup(monkeypatch, guidance, negative):
    torch = types.ModuleType("torch")
    torch.Generator = FakeGenerator
    diffusers = types.ModuleType("diffusers")
    diffusers.ClassifierFreeGuidance = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    progress = SimpleNamespace(on_step=None)
    calls = []

    class Pipeline:
        def update_components(self, **kwargs):
            self.guider = kwargs["guider"]

        def __call__(self, **kwargs):
            calls.append(kwargs)
            for i in range(kwargs["num_inference_steps"]):
                progress.on_step(i)
            return [FakeImage(kwargs["width"], kwargs["height"])]

    pipeline = Pipeline()
    loaded = anima.LoadedAnima(pipeline, progress)
    seen = []
    kwargs = dict(
        prompt="landscape 山",
        negative_prompt=negative,
        width=256,
        height=512,
        steps=3,
        guidance=guidance,
        seed=42,
    )
    assert anima.generate_image(loaded, **kwargs, on_step=seen.append).startswith(b"\x89PNG")
    assert pipeline.guider.guidance_scale == guidance
    assert seen == [0, 1, 2] and progress.on_step is None
    assert calls[0]["negative_prompt"] == negative
    assert calls[0]["generator"].seed == 42
    assert calls[0]["output"] == "images"
    assert "guidance_scale" not in calls[0]  # guider owns CFG in modular pipelines

    def cancel(_):
        raise GenerationCancelled

    with pytest.raises(GenerationCancelled):
        anima.generate_image(loaded, **kwargs, on_step=cancel)
    assert progress.on_step is None
    seen.clear()
    anima.generate_image(loaded, **kwargs, on_step=seen.append)
    assert seen == [0, 1, 2]  # healthy resident reusable after cancellation


def test_real_diffusers_conversion_on_tiny_cpu_tensors(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "accelerate")):
        pytest.skip("optional inference extra is not installed")
    result = subprocess.run(
        [sys.executable, "-m", "tests.unit.anima_cpu_check", str(tmp_path)],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loader_only_uses_verified_local_files(monkeypatch, tmp_path):
    from pathlib import Path

    from image_studio.hub.anima import RECIPES, SHARED_REPO, make_sources
    from image_studio.testing import FakeHub

    profile = ProfileId.ANIMA_29B
    hub = FakeHub(tmp_path)
    root = hub.seed_snapshot(RECIPES[profile].repo_id)
    hub.seed_snapshot(SHARED_REPO)
    sources = make_sources(hub.cache_dir, RECIPES[profile].repo_id, root.name, profile)
    calls = []

    class Component:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert Path(path).is_absolute() and Path(path).is_dir()
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
            calls.append((path, kwargs))
            return cls()

    diffusers = types.ModuleType("diffusers")
    diffusers.AutoencoderKLQwenImage = Component
    diffusers.FlowMatchEulerDiscreteScheduler = lambda **kw: kw
    transformers = types.ModuleType("transformers")
    for name in ("Qwen3Model", "Qwen2Tokenizer", "T5Tokenizer"):
        setattr(transformers, name, Component)
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16 = "bf16", "fp16"
    safe = types.ModuleType("safetensors.torch")
    checkpoint = {}

    def load_file(path, device):
        assert path == str(root / RECIPES[profile].checkpoint) and device == "cpu"
        return checkpoint

    safe.load_file = load_file
    for name, module in (
        ("torch", torch),
        ("diffusers", diffusers),
        ("transformers", transformers),
        ("safetensors.torch", safe),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    denoiser, conditioner = object(), object()

    def convert(state, shared, layers, dtype):
        assert state is checkpoint and layers == 40 and dtype == "bf16"
        assert str(shared) == sources[1].snapshot_path
        return denoiser, conditioner

    monkeypatch.setattr(anima, "load_checkpoint_models", convert)
    components = {}
    devices = []
    pipeline = SimpleNamespace(
        update_components=lambda **kw: components.update(kw), to=devices.append
    )
    monkeypatch.setattr(anima, "make_pipeline", lambda: anima.LoadedAnima(pipeline, None))
    loaded, name = anima.load_pipeline(str(root), "bfloat16", profile, sources)
    assert loaded.pipeline is pipeline and name == "AnimaModularPipeline"
    assert len(calls) == 4 and devices == ["cuda"]
    assert components["transformer"] is denoiser
    assert components["text_conditioner"] is conditioner
    assert components["scheduler"] == {"shift": 3.0}
    (Path(sources[1].snapshot_path) / "vae/config.json").unlink()
    with pytest.raises(FileNotFoundError, match="vae/config.json"):
        anima.load_pipeline(str(root), "bfloat16", profile, sources)


def test_worker_reports_cooperative_cancellation_without_fault():
    import queue

    from image_studio.inference.protocol import MsgRunFinished, RunTask
    from image_studio.inference.worker import _execute_run

    def cancelled(*args, **kwargs):
        raise GenerationCancelled

    events = queue.Queue()
    task = RunTask(
        run_id="cancel-me",
        prompt="landscape",
        negative_prompt=None,
        width=256,
        height=256,
        steps=10,
        guidance=1,
        seeds=(1,),
        artifact_ids=("image-001",),
    )
    shutdown = _execute_run(
        object(),
        task,
        SimpleNamespace(poll=lambda *_: False),
        events,
        SimpleNamespace(generate_image=cancelled),
    )
    assert shutdown is False
    message = events.get_nowait()
    assert isinstance(message, MsgRunFinished)
    assert message.status == "cancelled" and message.completed_count == 0
    assert events.empty()
