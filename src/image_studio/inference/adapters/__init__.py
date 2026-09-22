"""Small explicit adapter dispatch; imports remain CPU-safe."""

from typing import Any

from image_studio.inference.adapters.anima import AnimaAdapter
from image_studio.inference.adapters.base import PipelineAdapter
from image_studio.inference.adapters.qwen_image import QwenImageAdapter
from image_studio.inference.adapters.z_image import ZImageAdapter
from image_studio.inference.protocol import WorkerLaunch
from image_studio.schemas import ProfileId


def for_profile(profile: ProfileId) -> PipelineAdapter[Any]:
    if profile is ProfileId.QWEN_IMAGE_21:
        return QwenImageAdapter()
    if profile in (ProfileId.ANIMA_TURBO, ProfileId.ANIMA_29B):
        return AnimaAdapter()
    if profile in (ProfileId.Z_IMAGE, ProfileId.Z_IMAGE_TURBO):
        return ZImageAdapter()
    raise ValueError(f"unsupported profile {profile!r}")


def load(launch: WorkerLaunch) -> tuple[Any, str]:
    adapter = for_profile(launch.profile)
    return adapter.load_pipeline(launch.snapshot_path, launch.dtype, launch.profile, launch.sources)
