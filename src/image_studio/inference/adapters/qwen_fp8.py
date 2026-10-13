"""Offline Unsloth FP8 component loading; imported only by the model worker."""

from __future__ import annotations

import json
from pathlib import Path

from image_studio.hub import qwen_fp8 as recipe
from image_studio.schemas import ProfileId


def read_checkpoint(path: Path, *, component: str):
    """Restore only the declared safetensors format, without pickle or remote code."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
        expected_format = f"unsloth_prequant_{component}_state_dict_v1"
        if header.get("unsloth_format") != expected_format:
            raise ValueError(f"unsupported Unsloth {component} checkpoint format")
        metadata = json.loads(header.get("unsloth_metadata", "{}"))
        if (
            metadata.get("base_model_id") != recipe.BASE_REPO
            or metadata.get("scheme") != "fp8"
            or metadata.get("torch_dtype") != "bfloat16"
        ):
            raise ValueError(f"Unsloth {component} must contain Qwen-Image-2.1 FP8/BF16 weights")
        names = json.loads(header.get("tensor_names", "[]"))
        if not names or len(set(names)) != len(names):
            raise ValueError("checkpoint tensor manifest is empty or duplicated")
        allowed = {"Tensor", "Float8Tensor"} if component == "transformer" else {"Tensor"}
        kinds = {json.loads(header[name]).get("_type") for name in names}
        if not kinds <= allowed:
            raise ValueError("unsupported tensor type in FP8 checkpoint")
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    if component == "transformer":
        from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict

        if metadata.get("fp8_granularity") != "per_row" or "Float8Tensor" not in kinds:
            raise ValueError("Qwen FP8 transformer requires per-row Float8Tensor weights")
        state, leftovers = unflatten_tensor_state_dict(state, header)
        if leftovers:
            raise ValueError("FP8 checkpoint contains undeclared tensors")
    if set(state) != set(names):
        raise ValueError("FP8 checkpoint tensor manifest is incomplete")
    return state


def prepare_text_encoder(encoder):
    """Keep pre-cast leaves in FP8 storage and use BF16 only during their forward."""
    import torch
    from diffusers.hooks.layerwise_casting import apply_layerwise_casting_hook

    count = 0
    for module in encoder.modules():
        parameters = list(module.parameters(recurse=False))
        if not any(p.dtype == torch.float8_e4m3fn for p in parameters):
            continue
        if list(module.children()) or any(p.dtype != torch.float8_e4m3fn for p in parameters):
            raise ValueError("unsupported mixed or non-leaf FP8 text encoder module")
        apply_layerwise_casting_hook(module, torch.float8_e4m3fn, torch.bfloat16, False)
        count += 1
    if not count:
        raise ValueError("text encoder checkpoint contains no FP8 layers")
    return encoder.eval().requires_grad_(False)


def load_components(primary: Path, base: Path):
    """Meta-initialize parameters, preserve real buffers, then assign checkpoint tensors."""
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    transformer_config = json.loads((base / "transformer/config.json").read_text())
    encoder_config = Qwen3VLConfig.from_pretrained(
        str(base / "text_encoder"), local_files_only=True, trust_remote_code=False
    )
    with init_empty_weights(include_buffers=False):
        transformer = QwenImage21Transformer2DModel.from_config(transformer_config)
        encoder = Qwen3VLForConditionalGeneration(encoder_config)
    transformer.load_state_dict(
        read_checkpoint(primary / recipe.TRANSFORMER_FILE, component="transformer"),
        strict=True,
        assign=True,
    )
    encoder.load_state_dict(
        read_checkpoint(primary / recipe.TEXT_ENCODER_FILE, component="text_encoder"),
        strict=True,
        assign=True,
    )
    encoder.tie_weights()
    transformer.eval().requires_grad_(False)
    return transformer, prepare_text_encoder(encoder)


def load_pipeline(snapshot_path: str, dtype: str, sources):
    import torch
    from diffusers import QwenImage21Pipeline

    if dtype != "bfloat16":
        raise ValueError("Unsloth Qwen-Image-2.1 FP8 requires bfloat16 compute")
    if torch.cuda.get_device_capability() < (8, 9):
        raise ValueError(
            "Unsloth Qwen-Image-2.1 FP8 requires NVIDIA compute capability 8.9 or later"
        )
    primary = Path(snapshot_path)
    problems = recipe.model_problems(primary, recipe.REPO, ProfileId.QWEN_IMAGE_21, sources)
    if problems:
        raise FileNotFoundError("; ".join(p.detail for p in problems))
    base = Path(sources[1].snapshot_path)
    transformer, encoder = load_components(primary, base)
    pipeline = QwenImage21Pipeline.from_pretrained(
        str(base),
        transformer=transformer,
        text_encoder=encoder,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
    )
    # Device-only movement preserves Float8Tensor scales and text encoder FP8 storage.
    pipeline.to("cuda")
    return pipeline, QwenImage21Pipeline.__name__
