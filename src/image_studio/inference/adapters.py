"""Small explicit adapter dispatch; imports remain CPU-safe."""

from image_studio.inference import anima, qwen_image, z_image
from image_studio.schemas import ProfileId


def for_profile(profile: ProfileId):
    if profile is ProfileId.QWEN_IMAGE_21:
        return qwen_image
    if profile in (ProfileId.ANIMA_TURBO, ProfileId.ANIMA_29B):
        return anima
    if profile in (ProfileId.Z_IMAGE, ProfileId.Z_IMAGE_TURBO):
        return z_image
    raise ValueError(f"unsupported profile {profile!r}")


def load(launch):
    adapter = for_profile(launch.profile)
    if adapter is anima:
        return adapter.load_pipeline(
            launch.snapshot_path, launch.dtype, launch.profile, launch.sources
        )
    return adapter.load_pipeline(launch.snapshot_path, launch.dtype)
