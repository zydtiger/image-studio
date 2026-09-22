"""Shared interface for model adapters; no inference dependencies are imported."""

from collections.abc import Callable
from typing import Protocol

from image_studio.schemas import ModelSource, ProfileId


class PipelineAdapter[Loaded](Protocol):
    """Load a resident model and generate images with that same model type.

    Complete Diffusers snapshots may ignore ``profile`` and ``sources``.
    The worker owns the loaded object and its lifetime; adapters are stateless.
    """

    def load_pipeline(
        self, snapshot_path: str, dtype: str, profile: ProfileId, sources: tuple[ModelSource, ...]
    ) -> tuple[Loaded, str]: ...

    def generate_image(
        self,
        loaded: Loaded,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        on_step: Callable[[int], None],
    ) -> bytes: ...
