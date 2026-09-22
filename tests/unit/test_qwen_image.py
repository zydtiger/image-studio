"""Qwen-Image-2.1 manifests and worker argument mapping without model weights."""

import io
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from image_studio.hub.cache import _snapshot_incomplete, snapshot_problems
from image_studio.hub.compatibility import check_compatibility, filter_files
from image_studio.inference import adapters, qwen_image
from image_studio.inference.z_image import GenerationCancelled
from image_studio.schemas import ErrorCode, ProfileId
from tests.unit.test_hub_compat_client import _api_with_files
from tests.unit.test_snapshot_validation import _complete_files, _sharded, _write


def qwen_files():
    files = _sharded(_complete_files())
    files = {name: value for name, value in files.items() if not name.startswith("tokenizer/")}
    files["model_index.json"] = json.dumps(
        {
            "_class_name": "QwenImage21Pipeline",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
            "processor": ["transformers", "Qwen3VLProcessor"],
            "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage21"],
        }
    )
    files.update(
        {
            "processor/tokenizer_config.json": "{}",
            "processor/tokenizer.json": "{}",
            "processor/preprocessor_config.json": "{}",
            "processor/video_preprocessor_config.json": "{}",
            "processor/chat_template.jinja": "{{ messages }}",
        }
    )
    return files


def test_manifest_remote_local_and_profile_agree(tmp_path):
    files = qwen_files()
    report = check_compatibility(_api_with_files(tmp_path / "hub", files).api, "test/repo")
    assert report.structurally_compatible
    assert report.selectable_profiles == [ProfileId.QWEN_IMAGE_21]
    assert not any("Turbo" in note for note in report.notes)
    assert "processor/chat_template.jinja" in filter_files(list(files))
    root = _write(tmp_path / "snapshot", files)
    assert snapshot_problems(root, ProfileId.QWEN_IMAGE_21) == []
    assert not _snapshot_incomplete(root)
    assert snapshot_problems(root, ProfileId.Z_IMAGE)[0].code is ErrorCode.UNSUPPORTED_MODEL
    z_root = _write(tmp_path / "z", _complete_files())
    assert snapshot_problems(z_root, ProfileId.QWEN_IMAGE_21)[0].code is ErrorCode.UNSUPPORTED_MODEL


@pytest.mark.parametrize(
    "missing",
    [
        "processor/chat_template.jinja",
        "processor/preprocessor_config.json",
        "processor/video_preprocessor_config.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
    ],
)
def test_missing_processor_files_rejected_locally_and_remotely(tmp_path, missing):
    files = qwen_files()
    del files[missing]
    report = check_compatibility(_api_with_files(tmp_path / "hub", files).api, "test/repo")
    assert not report.structurally_compatible
    assert f"{missing} is missing" in report.findings
    root = _write(tmp_path / "snapshot", files)
    assert _snapshot_incomplete(root)
    assert any(missing in problem.detail for problem in snapshot_problems(root))


def test_wrong_processor_class_and_missing_weight_shard_rejected(tmp_path):
    files = qwen_files()
    index = json.loads(files["model_index.json"])
    index["processor"] = ["transformers", "AutoProcessor"]
    files["model_index.json"] = json.dumps(index)
    assert not check_compatibility(
        _api_with_files(tmp_path / "hub", files).api, "test/repo"
    ).structurally_compatible
    files = qwen_files()
    del files["transformer/diffusion_pytorch_model-00002-of-00002.safetensors"]
    problems = snapshot_problems(_write(tmp_path / "snapshot", files))
    assert any("00002-of-00002" in problem.detail for problem in problems)


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


@pytest.fixture
def torch_stub(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(Generator=Generator, bfloat16="bf16", float16="fp16"),
    )


def test_load_offline_and_adapter_dispatch(torch_stub, monkeypatch):
    calls = {}

    class QwenImage21Pipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.update(path=path, **kwargs)
            return cls()

        def to(self, device):
            calls["device"] = device

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(QwenImage21Pipeline=QwenImage21Pipeline)
    )
    launch = SimpleNamespace(
        profile=ProfileId.QWEN_IMAGE_21, snapshot_path="/snapshot", dtype="bfloat16", sources=()
    )
    _, name = adapters.load(launch)
    assert name == "QwenImage21Pipeline"
    assert calls == {
        "path": "/snapshot",
        "torch_dtype": "bf16",
        "variant": None,
        "local_files_only": True,
        "use_safetensors": True,
        "trust_remote_code": False,
        "device": "cuda",
    }


def test_generation_exact_signature_progress_seeds_rgba_and_cancel(torch_stub):
    generators = []
    steps = []

    def pipeline(
        *,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
        true_cfg_scale,
        generator,
        output_type,
        use_kv_cache,
        callback_on_step_end,
    ):
        assert prompt == "山 watercolor"
        assert negative_prompt is None
        assert true_cfg_scale == 1.0
        assert output_type == "pil" and use_kv_cache is True
        assert generator.device == "cuda"
        generators.append(generator)
        for i in range(num_inference_steps):
            tensors = {"latents": object()}
            assert callback_on_step_end(None, i, 0, tensors) is tensors
        return SimpleNamespace(images=[Image.new("RGBA", (width, height), (12, 34, 56, 78))])

    kwargs = dict(
        prompt="山 watercolor",
        negative_prompt=None,
        width=256,
        height=288,
        steps=3,
        guidance=1.0,
    )
    for seed in (42, 43):
        png = qwen_image.QwenImageAdapter().generate_image(
            pipeline, **kwargs, seed=seed, on_step=steps.append
        )
        with Image.open(io.BytesIO(png)) as image:
            assert image.size == (256, 288)
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0)) == (12, 34, 56, 78)
    assert steps == [0, 1, 2, 0, 1, 2]
    assert [g.seed for g in generators] == [42, 43]
    assert generators[0] is not generators[1]

    def cancel(step):
        if step == 1:
            raise GenerationCancelled

    with pytest.raises(GenerationCancelled):
        qwen_image.QwenImageAdapter().generate_image(pipeline, **kwargs, seed=44, on_step=cancel)
