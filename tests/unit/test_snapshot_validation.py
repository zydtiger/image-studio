"""Offline matrix for the shared Z-Image snapshot/component validation."""

from __future__ import annotations

import json
from pathlib import Path

from image_studio.hub.cache import problem_codes, snapshot_problems, snapshot_weight_variant
from image_studio.schemas import ErrorCode


def _complete_files() -> dict[str, str | bytes]:
    """An unsharded-but-valid variant: single weight files everywhere."""
    return {
        "model_index.json": json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        "scheduler/scheduler_config.json": "{}",
        "text_encoder/config.json": "{}",
        "text_encoder/model.safetensors": b"weights",
        "tokenizer/tokenizer_config.json": "{}",
        "tokenizer/tokenizer.json": "{}",
        "transformer/config.json": "{}",
        "transformer/diffusion_pytorch_model.safetensors": b"weights",
        "vae/config.json": "{}",
        "vae/diffusion_pytorch_model.safetensors": b"weights",
    }


def _write(root: Path, files: dict[str, str | bytes]) -> Path:
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)
    return root


def _sharded(files: dict[str, str | bytes]) -> dict[str, str | bytes]:
    """Convert transformer weights to a valid two-shard layout."""
    files = dict(files)
    del files["transformer/diffusion_pytorch_model.safetensors"]
    shards = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    for shard in shards:
        files[f"transformer/{shard}"] = b"shard"
    files["transformer/diffusion_pytorch_model.safetensors.index.json"] = json.dumps(
        {"weight_map": {f"w{i}": shard for i, shard in enumerate(shards)}}
    )
    return files


def test_unsharded_complete_variant_accepted(tmp_path: Path) -> None:
    root = _write(tmp_path / "snap", _complete_files())
    assert snapshot_problems(root) == []


def test_sharded_complete_variant_accepted(tmp_path: Path) -> None:
    root = _write(tmp_path / "snap", _sharded(_complete_files()))
    assert snapshot_problems(root) == []


def _bf16(files: dict[str, str | bytes]) -> dict[str, str | bytes]:
    """Convert all weighted components to the bf16 variant layout.

    Mirrors the official Tongyi-MAI/Z-Image-Turbo commit that ships bf16
    weights only: sharded text_encoder and transformer with the
    transformers-style ``*.index.bf16.json`` index and bf16 shard names,
    plus a single bf16 vae file.
    """
    files = dict(files)
    del files["text_encoder/model.safetensors"]
    text_encoder_shards = [
        "model.bf16-00001-of-00002.safetensors",
        "model.bf16-00002-of-00002.safetensors",
    ]
    for shard in text_encoder_shards:
        files[f"text_encoder/{shard}"] = b"shard"
    files["text_encoder/model.safetensors.index.bf16.json"] = json.dumps(
        {"weight_map": {f"t{i}": shard for i, shard in enumerate(text_encoder_shards)}}
    )
    del files["transformer/diffusion_pytorch_model.safetensors"]
    transformer_shards = [
        "diffusion_pytorch_model.bf16-00001-of-00002.safetensors",
        "diffusion_pytorch_model.bf16-00002-of-00002.safetensors",
    ]
    for shard in transformer_shards:
        files[f"transformer/{shard}"] = b"shard"
    files["transformer/diffusion_pytorch_model.safetensors.index.bf16.json"] = json.dumps(
        {"weight_map": {f"w{i}": shard for i, shard in enumerate(transformer_shards)}}
    )
    del files["vae/diffusion_pytorch_model.safetensors"]
    files["vae/diffusion_pytorch_model.bf16.safetensors"] = b"weights"
    return files


def test_bf16_complete_layout_accepted_and_selected(tmp_path: Path) -> None:
    root = _write(tmp_path / "snap", _bf16(_complete_files()))
    assert snapshot_problems(root) == []
    assert snapshot_weight_variant(root) == "bf16"


def test_default_layouts_select_no_variant(tmp_path: Path) -> None:
    for files in (_complete_files(), _sharded(_complete_files())):
        root = _write(tmp_path / "snap", files)
        assert snapshot_problems(root) == []
        assert snapshot_weight_variant(root) is None


def test_default_layout_preferred_when_both_variants_complete(tmp_path: Path) -> None:
    files = _bf16(_complete_files())
    files.update(_complete_files())  # every component exists in both layouts
    root = _write(tmp_path / "snap", files)
    assert snapshot_problems(root) == []
    assert snapshot_weight_variant(root) is None


def test_bf16_missing_shard_rejected(tmp_path: Path) -> None:
    files = _bf16(_complete_files())
    del files["transformer/diffusion_pytorch_model.bf16-00002-of-00002.safetensors"]
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert any(
        "diffusion_pytorch_model.bf16-00002-of-00002.safetensors" in problem.detail
        for problem in problems
    )
    assert snapshot_weight_variant(root) is None


def test_bf16_malformed_index_rejected(tmp_path: Path) -> None:
    files = _bf16(_complete_files())
    files["text_encoder/model.safetensors.index.bf16.json"] = "{oops"
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert any("index.bf16.json" in problem.detail for problem in problems)


def test_bf16_unsafe_shard_name_rejected(tmp_path: Path) -> None:
    files = _bf16(_complete_files())
    files["transformer/diffusion_pytorch_model.safetensors.index.bf16.json"] = json.dumps(
        {"weight_map": {"w0": "../../escape.bf16.safetensors"}}
    )
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert any("unsafe shard name" in problem.detail for problem in problems)


def test_mixed_partial_layouts_rejected_without_mixing(tmp_path: Path) -> None:
    # text_encoder only exists as bf16 while transformer/vae only exist in
    # the default layout: neither variant is complete, and the snapshot must
    # not validate by mixing them.
    files = _bf16(_complete_files())
    del files["transformer/diffusion_pytorch_model.bf16-00001-of-00002.safetensors"]
    del files["transformer/diffusion_pytorch_model.bf16-00002-of-00002.safetensors"]
    del files["transformer/diffusion_pytorch_model.safetensors.index.bf16.json"]
    del files["vae/diffusion_pytorch_model.bf16.safetensors"]
    files["transformer/diffusion_pytorch_model.safetensors"] = b"weights"
    files["vae/diffusion_pytorch_model.safetensors"] = b"weights"
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    details = [problem.detail for problem in problems]
    assert any("text_encoder weights are missing" in detail for detail in details)
    assert any("transformer weights are missing" in detail for detail in details)
    assert any("vae weights are missing" in detail for detail in details)


def test_legacy_bf16_index_layout_not_supported(tmp_path: Path) -> None:
    # The deprecated legacy sharding format puts the variant before
    # "index" (*.bf16.index.json); only the transformers-style
    # *.index.bf16.json layout is supported.
    files = _complete_files()
    del files["text_encoder/model.safetensors"]
    files["text_encoder/model-00001-of-00002.safetensors"] = b"shard"
    files["text_encoder/model-00002-of-00002.safetensors"] = b"shard"
    files["text_encoder/model.safetensors.bf16.index.json"] = json.dumps(
        {
            "weight_map": {
                "t0": "model-00001-of-00002.safetensors",
                "t1": "model-00002-of-00002.safetensors",
            }
        }
    )
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert any("text_encoder weights are missing" in problem.detail for problem in problems)


def test_wrong_pipeline_class_rejected(tmp_path: Path) -> None:
    files = _complete_files()
    files["model_index.json"] = json.dumps({"_class_name": "StableDiffusionPipeline"})
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.UNSUPPORTED_MODEL
    assert "StableDiffusionPipeline" in problems[0].detail


def test_partial_component_weights_rejected(tmp_path: Path) -> None:
    files = _complete_files()
    del files["transformer/diffusion_pytorch_model.safetensors"]  # vae weights intact
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert any("transformer weights are missing" in problem.detail for problem in problems)


def test_missing_indexed_shard_rejected(tmp_path: Path) -> None:
    files = _sharded(_complete_files())
    del files["transformer/diffusion_pytorch_model-00002-of-00002.safetensors"]
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert any(
        "diffusion_pytorch_model-00002-of-00002.safetensors" in problem.detail
        for problem in problems
    )


def test_malformed_model_index_is_typed_cache_incomplete(tmp_path: Path) -> None:
    root = _write(tmp_path / "snap", {"model_index.json": "not-json{"})
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert "malformed" in problems[0].detail


def test_malformed_shard_index_rejected(tmp_path: Path) -> None:
    files = _sharded(_complete_files())
    files["transformer/diffusion_pytorch_model.safetensors.index.json"] = "{oops"
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert any("index.json" in problem.detail for problem in problems)


def test_missing_component_declaration_rejected(tmp_path: Path) -> None:
    files = _complete_files()
    index = json.loads(files["model_index.json"])
    del index["vae"]
    files["model_index.json"] = json.dumps(index)
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert any("does not declare component 'vae'" in problem.detail for problem in problems)


def test_missing_tokenizer_file_rejected(tmp_path: Path) -> None:
    files = _complete_files()
    del files["tokenizer/tokenizer.json"]
    root = _write(tmp_path / "snap", files)
    problems = snapshot_problems(root)
    assert any("tokenizer/tokenizer.json" in problem.detail for problem in problems)


def test_missing_model_index_rejected(tmp_path: Path) -> None:
    root = _write(tmp_path / "snap", {"README.md": ""})
    problems = snapshot_problems(root)
    assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
    assert "model_index.json is missing" == problems[0].detail


class TestManifestDeclarations:
    def _index(self, **overrides: object) -> str:
        base = json.loads(_complete_files()["model_index.json"])
        base.update(overrides)
        return json.dumps(base)

    def _problems(self, model_index: str):
        files = _complete_files()
        files["model_index.json"] = model_index
        return snapshot_problems(_write(_tmp() / "snap", files))

    def test_null_required_declaration_rejected(self) -> None:
        problems = self._problems(self._index(text_encoder=None))
        assert problems, "null declaration must not validate"
        assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
        assert any("does not declare component 'text_encoder'" in p.detail for p in problems)

    def test_wrong_component_class_rejected(self) -> None:
        problems = self._problems(self._index(text_encoder=["transformers", "LlamaModel"]))
        assert problem_codes(problems) is ErrorCode.UNSUPPORTED_MODEL
        assert any("text_encoder declaration transformers.LlamaModel" in p.detail for p in problems)

    def test_wrong_component_library_rejected(self) -> None:
        problems = self._problems(
            self._index(scheduler=["torch", "FlowMatchEulerDiscreteScheduler"])
        )
        assert problem_codes(problems) is ErrorCode.UNSUPPORTED_MODEL
        assert any("scheduler declaration torch." in p.detail for p in problems)

    def test_malformed_declaration_rejected(self) -> None:
        problems = self._problems(self._index(vae="not-a-declaration"))
        assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
        assert any("declares component 'vae' malformed" in p.detail for p in problems)


def _tmp() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp())


class TestWeightMapHardening:
    def _sharded_with_map(self, weight_map: object):
        files = _sharded(_complete_files())
        files["transformer/diffusion_pytorch_model.safetensors.index.json"] = json.dumps(
            {"weight_map": weight_map}
        )
        return snapshot_problems(_write(_tmp() / "snap", files))

    def test_empty_weight_map_rejected_not_accepted(self) -> None:
        problems = self._sharded_with_map({})
        assert problems, "an empty weight_map describes no weights and must not validate"
        assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
        assert "no usable weight_map" in problems[0].detail

    def test_non_object_weight_map_is_typed_not_attribute_error(self) -> None:
        for bad_map in ([], None, "shard.safetensors", 42):
            problems = self._sharded_with_map(bad_map)
            assert problems, f"weight_map={bad_map!r} must not validate"
            assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
            assert "no usable weight_map" in problems[0].detail

    def test_invalid_shard_name_value_is_typed(self) -> None:
        problems = self._sharded_with_map({"w0": 123})
        assert problems and problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
        assert "invalid shard name" in problems[0].detail

    def test_unsafe_shard_names_rejected_without_path_operations(self) -> None:
        for name in ("../../escape.safetensors", "/etc/passwd", "..\\..\\win.safetensors"):
            problems = self._sharded_with_map({"w0": name})
            assert problems, f"shard {name!r} must be rejected"
            assert problem_codes(problems) is ErrorCode.CACHE_INCOMPLETE
            assert "unsafe shard name" in problems[0].detail

    def test_complete_variants_still_pass(self) -> None:
        assert snapshot_problems(_write(_tmp() / "a", _complete_files())) == []
        assert snapshot_problems(_write(_tmp() / "b", _sharded(_complete_files()))) == []
