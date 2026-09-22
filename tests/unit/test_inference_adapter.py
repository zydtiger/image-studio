"""Adapter argument-mapping tests with fake torch/diffusers modules.

Generation mocks never validate the real pipeline; these tests exist to pin
the *exact* kwargs, seed, progress, and cancellation wiring the adapter
passes to a pipeline object, using injected fake modules (the real stack is
not installed in CPU environments). Real-pipeline verification lives in the
opt-in GPU tests.
"""

from __future__ import annotations

import io
import sys
import types
from types import SimpleNamespace

import pytest

from image_studio.inference import common
from image_studio.inference.adapters.z_image import ZImageAdapter
from image_studio.inference.common import GenerationCancelled
from image_studio.schemas import ProfileId


class FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeImage:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def save(self, buffer, format=None):  # noqa: A002 - PIL signature
        buffer.write(b"\x89PNG-fake")


class FakePipeline:
    instances = []

    def __init__(self):
        self.calls = []
        self.to_devices = []
        FakePipeline.instances.append(self)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.pretrained_kwargs = {"path": path, **kwargs}
        return cls()

    def to(self, device):
        self.to_devices.append(device)
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        for step_index in range(kwargs["num_inference_steps"]):
            if callback is not None:
                callback(self, step_index, 0, {"latents": "fake"})
        image = FakeImage(kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[image])


@pytest.fixture
def fake_torch(monkeypatch):
    module = types.ModuleType("torch")
    module.bfloat16 = "torch.bfloat16"
    module.float16 = "torch.float16"
    module.Generator = FakeGenerator
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


@pytest.fixture
def fake_diffusers(monkeypatch):
    FakePipeline.instances = []
    FakePipeline.pretrained_kwargs = None
    module = types.ModuleType("diffusers")
    module.ZImagePipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", module)
    return module


def test_load_pipeline_offline_kwargs(fake_torch, fake_diffusers):
    pipeline, class_name = ZImageAdapter().load_pipeline(
        "/snap/path", "bfloat16", ProfileId.Z_IMAGE, ()
    )
    assert class_name == "FakePipeline"
    assert FakePipeline.pretrained_kwargs["path"] == "/snap/path"
    assert FakePipeline.pretrained_kwargs["torch_dtype"] == "torch.bfloat16"
    assert FakePipeline.pretrained_kwargs["variant"] is None
    assert FakePipeline.pretrained_kwargs["local_files_only"] is True
    assert FakePipeline.pretrained_kwargs["use_safetensors"] is True
    assert FakePipeline.pretrained_kwargs["trust_remote_code"] is False
    assert pipeline.to_devices == ["cuda"]


def test_load_pipeline_selects_bf16_variant_for_bf16_snapshot(fake_torch, fake_diffusers, tmp_path):
    from tests.unit.test_snapshot_validation import _bf16, _complete_files, _write

    root = _write(tmp_path / "snap", _bf16(_complete_files()))
    ZImageAdapter().load_pipeline(str(root), "bfloat16", ProfileId.Z_IMAGE_TURBO, ())
    assert FakePipeline.pretrained_kwargs["variant"] == "bf16"
    assert FakePipeline.pretrained_kwargs["torch_dtype"] == "torch.bfloat16"


def test_load_pipeline_keeps_default_variant_for_ordinary_snapshot(
    fake_torch, fake_diffusers, tmp_path
):
    from tests.unit.test_snapshot_validation import _complete_files, _write

    root = _write(tmp_path / "snap", _complete_files())
    ZImageAdapter().load_pipeline(str(root), "bfloat16", ProfileId.Z_IMAGE, ())
    assert FakePipeline.pretrained_kwargs["variant"] is None


def test_load_pipeline_rejects_unsupported_dtype(fake_torch, fake_diffusers):
    with pytest.raises(ValueError, match="unsupported dtype"):
        ZImageAdapter().load_pipeline("/snap/path", "float32", ProfileId.Z_IMAGE, ())


def test_generate_image_exact_kwargs_and_seed(fake_torch, fake_diffusers):
    steps_seen = []
    pipeline = FakePipeline()
    png = ZImageAdapter().generate_image(
        pipeline,
        prompt="a cat",
        negative_prompt=None,
        width=256,
        height=512,
        steps=3,
        guidance=0.0,
        seed=1234,
        on_step=steps_seen.append,
    )
    (kwargs,) = pipeline.calls
    assert kwargs["prompt"] == "a cat"
    assert kwargs["negative_prompt"] is None
    assert kwargs["width"] == 256
    assert kwargs["height"] == 512
    assert kwargs["num_inference_steps"] == 3
    assert kwargs["guidance_scale"] == 0.0
    assert kwargs["output_type"] == "pil"
    generator = kwargs["generator"]
    assert isinstance(generator, FakeGenerator)
    assert generator.device == "cuda"
    assert generator.seed == 1234
    assert steps_seen == [0, 1, 2], "progress must fire once per step, 0-based"
    assert png.startswith(b"\x89PNG-fake")


def test_generate_image_passes_negative_prompt_when_present(fake_torch, fake_diffusers):
    pipeline = FakePipeline()
    ZImageAdapter().generate_image(
        pipeline,
        prompt="a cat",
        negative_prompt="blurry",
        width=256,
        height=256,
        steps=1,
        guidance=4.0,
        seed=7,
        on_step=lambda _step: None,
    )
    assert pipeline.calls[0]["negative_prompt"] == "blurry"


def test_step_callback_timestep_kwargs_passthrough(fake_torch, fake_diffusers):
    received = {}

    def on_step(step_index):
        received[step_index] = True

    pipeline = FakePipeline()
    ZImageAdapter().generate_image(
        pipeline,
        prompt="p",
        negative_prompt=None,
        width=256,
        height=256,
        steps=2,
        guidance=0.0,
        seed=1,
        on_step=on_step,
    )
    # The callback must return the tensor-inputs dict unchanged for
    # diffusers to keep intermediate state consistent.
    callback = pipeline.calls[0]["callback_on_step_end"]
    assert callback(pipeline, 0, 0, {"latents": "keep"}) == {"latents": "keep"}
    assert received == {0: True, 1: True}


def test_cancellation_raises_from_step_boundary(fake_torch, fake_diffusers):
    def on_step(step_index):
        if step_index == 1:
            raise GenerationCancelled

    pipeline = FakePipeline()
    with pytest.raises(GenerationCancelled):
        ZImageAdapter().generate_image(
            pipeline,
            prompt="p",
            negative_prompt=None,
            width=256,
            height=256,
            steps=3,
            guidance=0.0,
            seed=1,
            on_step=on_step,
        )
    assert len(pipeline.calls) == 1


def test_encode_png_validates_dimensions():
    with pytest.raises(ValueError, match="expected"):
        common.encode_png(FakeImage(64, 128), 256, 256)

    buffer = io.BytesIO()
    FakeImage(8, 8).save(buffer)
    assert common.encode_png(FakeImage(8, 8), 8, 8) == buffer.getvalue()


def test_fresh_generator_per_image(fake_torch, fake_diffusers):
    pipeline = FakePipeline()
    ZImageAdapter().generate_image(
        pipeline,
        prompt="p",
        negative_prompt=None,
        width=256,
        height=256,
        steps=1,
        guidance=0.0,
        seed=10,
        on_step=lambda _s: None,
    )
    ZImageAdapter().generate_image(
        pipeline,
        prompt="p",
        negative_prompt=None,
        width=256,
        height=256,
        steps=1,
        guidance=0.0,
        seed=11,
        on_step=lambda _s: None,
    )
    assert [call["generator"].seed for call in pipeline.calls] == [10, 11]
