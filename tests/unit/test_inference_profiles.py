"""Profile consumption and import purity for the inference package."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import image_studio.inference as inference_pkg
from image_studio.inference.profiles import (
    DEFAULT_REPOSITORIES,
    SUPPORTED_DTYPES,
    get_profile,
    worker_identity,
)
from image_studio.schemas import (
    PROFILES,
    FrozenGpu,
    FrozenModel,
    FrozenRunSpec,
    ProfileId,
    utc_now,
)


def make_spec(
    *,
    repo="repo/a",
    commit="c1",
    profile=ProfileId.Z_IMAGE_TURBO,
    dtype="bfloat16",
    gpu_uuid="GPU-AAA",
) -> FrozenRunSpec:
    return FrozenRunSpec(
        run_id="run-1",
        created_at=utc_now(),
        model=FrozenModel(
            registration_id="reg-1",
            repo_id=repo,
            commit_sha=commit,
            profile=profile,
            dtype=dtype,
            snapshot_path="/nonexistent/snapshot",
        ),
        gpu=FrozenGpu(uuid=gpu_uuid, name="GPU A"),
        prompt="a cat",
        negative_prompt=None,
        width=256,
        height=256,
        steps=3,
        guidance=0.0,
        image_count=1,
        seeds=(10,),
        artifact_ids=("image-001",),
    )


def test_get_profile_returns_contract_constants():
    for profile_id in ProfileId:
        assert get_profile(profile_id) is PROFILES[profile_id]
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile("not-a-profile")  # type: ignore[arg-type]


def test_profile_metadata_is_consumed_not_duplicated():
    turbo = get_profile(ProfileId.Z_IMAGE_TURBO)
    assert turbo.default_steps == 9
    assert turbo.guidance_fixed == 0.0
    assert turbo.negative_prompt_supported is False
    base = get_profile(ProfileId.Z_IMAGE)
    assert base.default_steps == 50
    assert base.guidance_fixed is None
    assert base.negative_prompt_supported is True
    assert DEFAULT_REPOSITORIES == {
        ProfileId.ANIMA_TURBO: "circlestone-labs/Anima",
        ProfileId.ANIMA_29B: "Gazingstars123/Anima-2.9B",
        ProfileId.Z_IMAGE: "Tongyi-MAI/Z-Image",
        ProfileId.Z_IMAGE_TURBO: "Tongyi-MAI/Z-Image-Turbo",
    }
    assert SUPPORTED_DTYPES == frozenset({"bfloat16", "float16"})


def test_worker_identity_reuse_rule():
    assert worker_identity(make_spec()) == worker_identity(make_spec())

    variants = {
        "repo": make_spec(repo="repo/b"),
        "commit": make_spec(commit="c2"),
        "profile": make_spec(profile=ProfileId.Z_IMAGE),
        "dtype": make_spec(dtype="float16"),
        "gpu": make_spec(gpu_uuid="GPU-BBB"),
    }
    baseline = worker_identity(make_spec())
    for changed, identity in variants.items():
        assert identity != baseline, changed


def test_importing_package_never_pulls_torch_or_diffusers():
    for module_info in pkgutil.walk_packages(
        inference_pkg.__path__, prefix="image_studio.inference."
    ):
        importlib.import_module(module_info.name)
    import sys

    assert "torch" not in sys.modules, "torch leaked into the parent process"
    assert "diffusers" not in sys.modules, "diffusers leaked into the parent process"
