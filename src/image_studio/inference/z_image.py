"""Real Z-Image pipeline adapter; imported only inside the worker process.

This module never imports torch or diffusers at module scope, so importing
it stays cheap and CPU-safe. The heavy stack is imported inside functions,
after the worker child has pinned ``CUDA_VISIBLE_DEVICES`` to the selected
GPU UUID (UUID-based selection must happen before the first torch import).

Loading is strictly offline: it uses the backend-verified snapshot path with
``local_files_only=True`` and never re-resolves repo or revision against the
Hub. The exact ``pipeline(...)`` argument mapping is pinned by unit tests
with a fake pipeline module; real-pipeline verification lives in the opt-in
GPU tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from image_studio.inference.profiles import SUPPORTED_DTYPES


#: Raised from the diffusers step callback to abort generation at a safe
#: inference boundary (cooperative cancel). Never leaks out of the worker.
class GenerationCancelled(Exception):
    pass


#: Per-step hook: called with the 0-based step index after each denoising
#: step. May raise :class:`GenerationCancelled` to abort the image.
StepCallback = Callable[[int], None]


def resolve_torch_dtype(dtype: str) -> Any:
    """Map a frozen dtype string onto a ``torch`` dtype object."""

    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(SUPPORTED_DTYPES)}"
        ) from None


def load_pipeline(snapshot_path: str, dtype: str) -> tuple[Any, str]:
    """Load the resident pipeline from a verified local snapshot.

    Returns ``(pipeline, pipeline_class_name)``. Raises on missing files,
    unsupported dtype, or a pipeline-class mismatch — the worker converts
    any raise into a fatal fault report; no network access is attempted.

    The weight-file variant is independent of the runtime ``dtype``: it is
    re-derived from the snapshot through the same shared layout rules as
    registration and submit validation, so a snapshot whose weights only
    exist as bf16 files loads through ``variant='bf16'`` and an ordinary
    snapshot through the default filenames.
    """

    from pathlib import Path

    from diffusers import ZImagePipeline

    from image_studio.hub.cache import snapshot_weight_variant

    torch_dtype = resolve_torch_dtype(dtype)
    variant = snapshot_weight_variant(Path(snapshot_path))
    pipeline = ZImagePipeline.from_pretrained(
        snapshot_path,
        torch_dtype=torch_dtype,
        variant=variant,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
    )
    pipeline.to("cuda")
    return pipeline, ZImagePipeline.__name__


def dependency_versions() -> dict[str, str]:
    """Record the inference stack versions for run metadata."""

    from importlib.metadata import PackageNotFoundError, version

    recorded: dict[str, str] = {}
    for distribution in ("torch", "torchvision", "diffusers", "transformers", "accelerate"):
        try:
            recorded[distribution] = version(distribution)
        except PackageNotFoundError:
            recorded[distribution] = "not-installed"
    return recorded


def generate_image(
    pipeline: Any,
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
    """Generate one image and return its PNG bytes.

    Exact pipeline kwargs: ``prompt``, ``negative_prompt`` (``None`` when
    unsupported or absent), ``width``/``height``, ``num_inference_steps``,
    ``guidance_scale``, a fresh per-image ``generator`` seeded with the
    frozen per-image seed, ``output_type="pil"``, and
    ``callback_on_step_end`` wiring progress and cooperative cancellation.
    """

    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def _on_step_end(_pipe: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
        on_step(step_index)
        return kwargs

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pil",
        callback_on_step_end=_on_step_end,
    )
    image = result.images[0]
    return encode_png(image, width, height)


def encode_png(image: Any, width: int, height: int) -> bytes:
    """Encode one PIL image to PNG bytes and validate its dimensions."""

    import io

    if image.width != width or image.height != height:
        raise ValueError(
            f"pipeline returned {image.width}x{image.height}, expected {width}x{height}"
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
