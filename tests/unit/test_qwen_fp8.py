"""Real CPU tensors verify FP8 restoration, layerwise computation and offline assembly."""

import json

import pytest

from image_studio.hub import qwen_fp8 as recipe
from image_studio.inference.adapters import qwen_fp8
from image_studio.inference.adapters.qwen_image import QwenImageAdapter
from image_studio.schemas import ProfileId


def save_checkpoint(path, state, component):
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict

    tensors, header = flatten_tensor_state_dict(state)
    header["unsloth_format"] = f"unsloth_prequant_{component}_state_dict_v1"
    header["unsloth_metadata"] = json.dumps(
        {
            "base_model_id": recipe.BASE_REPO,
            "scheme": "fp8",
            "torch_dtype": "bfloat16",
            "fp8_granularity": "per_row",
        }
    )
    save_file(tensors, path, metadata=header)


def quantized_weight(weight):
    import torch
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        Float8Tensor,
        QuantizeTensorToFloat8Kwargs,
    )

    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448
    return Float8Tensor(
        qdata=(weight.float() / scale).to(torch.float8_e4m3fn),
        scale=scale,
        block_size=[1, weight.shape[1]],
        mm_config=Float8MMConfig(emulate=False, use_fast_accum=True, pad_inner_dim=False),
        act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
            granularity=PerRow(),
            hp_value_lb=1e-12,
            kernel_preference=KernelPreference.TORCH,
        ),
        kernel_preference=KernelPreference.TORCH,
        dtype=torch.bfloat16,
    )


def test_scaled_transformer_roundtrip_preserves_quantization(tmp_path):
    import torch

    weight = quantized_weight(torch.arange(32, dtype=torch.bfloat16).reshape(4, 8))
    path = tmp_path / "transformer.safetensors"
    save_checkpoint(path, {"linear.weight": weight}, "transformer")
    restored = qwen_fp8.read_checkpoint(path, component="transformer")["linear.weight"]
    assert type(restored) is type(weight)
    assert restored.qdata.dtype == torch.float8_e4m3fn
    assert torch.equal(restored.qdata.float(), weight.qdata.float())
    assert torch.equal(restored.scale, weight.scale)
    assert restored.block_size == [1, 8]
    assert restored.act_quant_kwargs.hp_value_lb == 1e-12
    assert restored.mm_config.use_fast_accum


def test_text_encoder_leaf_computes_bf16_and_returns_to_fp8():
    import torch

    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4)).to(torch.float8_e4m3fn)
    weight = encoder[0].weight.to(torch.bfloat16).clone()
    bias = encoder[0].bias.to(torch.bfloat16).clone()
    qwen_fp8.prepare_text_encoder(encoder)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    with torch.inference_mode():
        result = encoder(x)
    torch.testing.assert_close(result, torch.nn.functional.linear(x, weight, bias))
    assert encoder[0].weight.dtype == torch.float8_e4m3fn
    assert not encoder[0].weight.requires_grad


def test_real_tiny_qwen_components_load_strictly_with_real_buffers(tmp_path):
    import torch
    from diffusers import QwenImage21Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    base = tmp_path / "base"
    primary = tmp_path / "primary"
    primary.mkdir()
    transformer = QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=1,
        context_in_dim=24,
        axes_dims_rope=(4, 6, 6),
    ).to(torch.bfloat16)
    transformer.save_config(base / "transformer")
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=12,
            rope_parameters={"rope_type": "default", "mrope_section": [2, 2, 2]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=24,
            deepstack_visual_indexes=[],
        ),
    )
    config.save_pretrained(base / "text_encoder")
    encoder = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
    state = transformer.state_dict()
    key = "transformer_blocks.0.attn.to_q.weight"
    state[key] = quantized_weight(state[key])
    save_checkpoint(primary / recipe.TRANSFORMER_FILE, state, "transformer")
    for module in encoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.float8_e4m3fn)
    save_checkpoint(primary / recipe.TEXT_ENCODER_FILE, encoder.state_dict(), "text_encoder")
    loaded_transformer, loaded_encoder = qwen_fp8.load_components(primary, base)
    assert type(loaded_transformer.state_dict()[key]).__name__ == "Float8Tensor"
    assert loaded_encoder.dtype == torch.bfloat16
    assert not any(t.is_meta for t in loaded_encoder.buffers())
    assert not any(t.is_meta for t in loaded_transformer.buffers())
    with torch.inference_mode():
        output = loaded_encoder(input_ids=torch.tensor([[1, 2, 3]]), output_hidden_states=True)
    assert torch.isfinite(output.logits).all()
    assert (
        loaded_encoder.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.float8_e4m3fn
    )


def test_offline_pipeline_receives_overrides_and_device_only_move(tmp_path, monkeypatch):
    import diffusers
    import torch

    sources = recipe.make_sources(tmp_path, recipe.REPO, "f" * 40, ProfileId.QWEN_IMAGE_21)
    calls = {}
    transformer, encoder = object(), object()
    monkeypatch.setattr(recipe, "model_problems", lambda *args: [])
    monkeypatch.setattr(qwen_fp8, "load_components", lambda *args: (transformer, encoder))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))

    class QwenImage21Pipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.update(path=path, **kwargs)
            return cls()

        def to(self, *args, **kwargs):
            calls["move"] = (args, kwargs)

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", QwenImage21Pipeline)
    _, name = QwenImageAdapter().load_pipeline(
        sources[0].snapshot_path, "bfloat16", ProfileId.QWEN_IMAGE_21, sources
    )
    assert name == "QwenImage21Pipeline"
    assert calls == dict(
        path=sources[1].snapshot_path,
        transformer=transformer,
        text_encoder=encoder,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        move=(("cuda",), {}),
    )
    with pytest.raises(ValueError, match="bfloat16 compute"):
        qwen_fp8.load_pipeline(sources[0].snapshot_path, "float16", sources)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    with pytest.raises(ValueError, match="compute capability"):
        qwen_fp8.load_pipeline(sources[0].snapshot_path, "bfloat16", sources)


@pytest.mark.parametrize("mutation", ["scheme", "missing", "unexpected", "format"])
def test_invalid_checkpoints_are_refused(tmp_path, mutation):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    path = tmp_path / "encoder.safetensors"
    save_checkpoint(
        path, {"linear.weight": torch.ones(4, 8).to(torch.float8_e4m3fn)}, "text_encoder"
    )
    with safe_open(path, framework="pt") as handle:
        header = handle.metadata()
    tensors = load_file(path)
    if mutation == "scheme":
        meta = json.loads(header["unsloth_metadata"])
        meta["scheme"] = "int8"
        header["unsloth_metadata"] = json.dumps(meta)
    elif mutation == "missing":
        tensors.clear()
    elif mutation == "unexpected":
        tensors["extra.weight"] = torch.zeros(1)
    else:
        header["unsloth_format"] = "unknown"
    # Avoid replacing a file while its tensors are still memory-mapped.
    bad = tmp_path / "invalid.safetensors"
    save_file(tensors, bad, metadata=header)
    with pytest.raises(ValueError):
        qwen_fp8.read_checkpoint(bad, component="text_encoder")
