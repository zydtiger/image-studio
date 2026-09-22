"""Offline Qwen-Image-2.1 text-to-image adapter; heavy imports stay worker-only."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from image_studio.inference.adapter import PipelineAdapter
from image_studio.inference.z_image import StepCallback, encode_png, resolve_torch_dtype
from image_studio.schemas import ModelSource, ProfileId

if TYPE_CHECKING:
    from diffusers import QwenImage21Pipeline


class QwenImageAdapter(PipelineAdapter["QwenImage21Pipeline"]):
    def load_pipeline(
        self, snapshot_path: str, dtype: str, profile: ProfileId, sources: tuple[ModelSource, ...]
    ) -> tuple[QwenImage21Pipeline, str]:
        from diffusers import QwenImage21Pipeline

        from image_studio.hub.cache import snapshot_weight_variant

        pipeline = QwenImage21Pipeline.from_pretrained(
            snapshot_path,
            torch_dtype=resolve_torch_dtype(dtype),
            variant=snapshot_weight_variant(Path(snapshot_path)),
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        )
        pipeline.to("cuda")
        return pipeline, QwenImage21Pipeline.__name__

    def generate_image(
        self,
        loaded: QwenImage21Pipeline,
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

        def on_step_end(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
            on_step(step_index)
            return kwargs

        result = loaded(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=guidance,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pil",
            use_kv_cache=True,
            callback_on_step_end=on_step_end,
        )
        return encode_png(result.images[0], width, height)
