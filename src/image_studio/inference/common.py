"""Shared worker and adapter helpers; heavy dependencies are imported only when used."""

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
