"""Offline original-checkpoint Anima adapter (heavy imports are worker-only).

The denoiser conversion delegates to Diffusers' Cosmos key mapping. The
checkpoint's own LLM adapter supplies AnimaTextConditioner, rather than
silently substituting the base model's conditioner. Nothing is exported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_studio.hub.anima import RECIPES, model_problems
from image_studio.inference.z_image import StepCallback, encode_png, resolve_torch_dtype
from image_studio.schemas import ModelSource, ProfileId


def split_checkpoint(state: dict[str, Any], layers: int) -> tuple[dict, dict]:
    """Normalize the original checkpoint prefixes and check transformer depth."""
    transformer, conditioner = {}, {}
    found_layers = set()
    for key, tensor in state.items():
        key = key.removeprefix("model.diffusion_model.").removeprefix("net.")
        if key.startswith("llm_adapter."):
            target, name = conditioner, key.removeprefix("llm_adapter.")
        else:
            target, name = transformer, "net." + key
            if key.startswith("blocks."):
                found_layers.add(int(key.split(".")[1]))
        if name in target:
            raise ValueError(f"duplicate Anima weight: {name}")
        target[name] = tensor
    if found_layers != set(range(layers)):
        raise ValueError(f"Anima checkpoint must contain exactly {layers} transformer layers")
    if not conditioner:
        raise ValueError("Anima checkpoint is missing its text conditioner")
    return transformer, conditioner


def load_checkpoint_models(checkpoint: dict, shared: Path, layers: int, dtype: Any):
    """Instantiate empty models, then strictly assign converted CPU tensors."""
    from accelerate import init_empty_weights
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
    )

    transformer_state, conditioner_state = split_checkpoint(checkpoint, layers)
    config = json.loads((shared / "transformer/config.json").read_text())
    config["num_layers"] = layers
    conditioner_config = json.loads((shared / "text_conditioner/config.json").read_text())
    with init_empty_weights():
        transformer = CosmosTransformer3DModel.from_config(config)
        conditioner = AnimaTextConditioner.from_config(conditioner_config)
    converted = convert_cosmos_transformer_checkpoint_to_diffusers(transformer_state)
    transformer.load_state_dict(converted, strict=True, assign=True)
    conditioner.load_state_dict(conditioner_state, strict=True, assign=True)
    return transformer.to(dtype=dtype), conditioner.to(dtype=dtype)


@dataclass
class LoadedAnima:
    pipeline: Any
    progress: Any


def make_pipeline() -> LoadedAnima:
    """Use the native modular loop with a cooperative per-step hook."""
    from diffusers import AnimaAutoBlocks, AnimaModularPipeline
    from diffusers.modular_pipelines.anima.denoise import AnimaDenoiseStep

    class ProgressDenoiseStep(AnimaDenoiseStep):
        on_step: StepCallback | None = None

        def loop_step(self, components, state, **kwargs):
            result = super().loop_step(components, state, **kwargs)
            if self.on_step is not None:
                self.on_step(kwargs["i"])
            return result

    blocks = AnimaAutoBlocks()
    progress = ProgressDenoiseStep()
    blocks.sub_blocks["denoise"].sub_blocks["text2image"].sub_blocks["denoise"] = progress
    return LoadedAnima(AnimaModularPipeline(blocks=blocks), progress)


def load_pipeline(
    snapshot_path: str, dtype: str, profile: ProfileId, sources: tuple[ModelSource, ...]
) -> tuple[LoadedAnima, str]:
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer, Qwen3Model, T5Tokenizer

    recipe = RECIPES[profile]
    problems = model_problems(Path(snapshot_path), recipe.repo_id, profile, sources)
    if problems:
        raise FileNotFoundError("; ".join(problem.detail for problem in problems))
    torch_dtype = resolve_torch_dtype(dtype)
    shared = Path(sources[1].snapshot_path)
    checkpoint = load_file(str(Path(snapshot_path) / recipe.checkpoint), device="cpu")
    transformer, conditioner = load_checkpoint_models(
        checkpoint, shared, recipe.layers, torch_dtype
    )
    del checkpoint
    offline = {"local_files_only": True, "trust_remote_code": False}
    loaded = make_pipeline()
    loaded.pipeline.update_components(
        transformer=transformer,
        text_conditioner=conditioner,
        text_encoder=Qwen3Model.from_pretrained(
            str(shared / "text_encoder"), dtype=torch_dtype, use_safetensors=True, **offline
        ),
        vae=AutoencoderKLQwenImage.from_pretrained(
            str(shared / "vae"), torch_dtype=torch_dtype, use_safetensors=True, **offline
        ),
        tokenizer=Qwen2Tokenizer.from_pretrained(str(shared / "tokenizer"), **offline),
        t5_tokenizer=T5Tokenizer.from_pretrained(str(shared / "t5_tokenizer"), **offline),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    loaded.pipeline.to("cuda")
    return loaded, "AnimaModularPipeline"


def generate_image(
    loaded: LoadedAnima,
    *,
    prompt: str,
    negative_prompt: str | None,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    on_step: StepCallback,
) -> bytes:
    import torch
    from diffusers import ClassifierFreeGuidance

    loaded.pipeline.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance))
    loaded.progress.on_step = on_step
    try:
        images = loaded.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pil",
            output="images",
        )
        return encode_png(images[0], width, height)
    finally:
        # Do not retain a run's IPC callback in the resident pipeline.
        loaded.progress.on_step = None
