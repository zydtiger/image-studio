"""Unit tests for contract schemas: profiles, normalization rules, events."""

from __future__ import annotations

import random

import pytest
from pydantic import TypeAdapter, ValidationError

from image_studio import schemas as s


def _request(**overrides: object) -> s.GenerationRequest:
    body: dict[str, object] = {
        "registration_id": "reg-1",
        "gpu_uuid": "GPU-fake-0001",
        "prompt": "hello",
        "width": 1024,
        "height": 1024,
        "count": 2,
    }
    body.update(overrides)
    return s.GenerationRequest(**body)


class TestProfiles:
    def test_single_truth_constants(self) -> None:
        assert set(s.PROFILES) == {s.ProfileId.Z_IMAGE, s.ProfileId.Z_IMAGE_TURBO}
        base = s.PROFILES[s.ProfileId.Z_IMAGE]
        turbo = s.PROFILES[s.ProfileId.Z_IMAGE_TURBO]
        assert (base.default_steps, base.guidance_default) == (50, 4.0)
        assert base.guidance_fixed is None and base.negative_prompt_supported
        assert (turbo.default_steps, turbo.guidance_fixed) == (9, 0.0)
        assert not turbo.negative_prompt_supported

    def test_dimension_and_count_constraints(self) -> None:
        with pytest.raises(ValidationError):
            _request(width=255)
        with pytest.raises(ValidationError):
            _request(height=1025)
        with pytest.raises(ValidationError):
            _request(count=5)
        with pytest.raises(ValidationError):
            _request(seed=s.MAX_SEED + 1)


class TestTurboRejection:
    def test_negative_prompt_rejected_not_ignored(self) -> None:
        turbo = s.PROFILES[s.ProfileId.Z_IMAGE_TURBO]
        with pytest.raises(s.ContractViolationError) as excinfo:
            s.validate_generation(_request(negative_prompt="blur"), turbo)
        assert excinfo.value.details["reason"] == "unsupported_field"
        assert excinfo.value.details["field"] == "negative_prompt"

    def test_nonzero_guidance_rejected(self) -> None:
        turbo = s.PROFILES[s.ProfileId.Z_IMAGE_TURBO]
        with pytest.raises(s.ContractViolationError) as excinfo:
            s.validate_generation(_request(guidance=1.5), turbo)
        assert excinfo.value.details["field"] == "guidance"

    def test_zero_guidance_and_defaults_accepted(self) -> None:
        turbo = s.PROFILES[s.ProfileId.Z_IMAGE_TURBO]
        s.validate_generation(_request(guidance=0.0), turbo)

    def test_base_accepts_negative_and_guidance(self) -> None:
        base = s.PROFILES[s.ProfileId.Z_IMAGE]
        s.validate_generation(_request(negative_prompt="blur", guidance=5.0), base)

    def test_steps_out_of_profile_range(self) -> None:
        # field-level bounds are 1..100, so exercise the profile branch with a
        # tighter custom profile
        narrow = s.PROFILES[s.ProfileId.Z_IMAGE].model_copy(
            update={"min_steps": 1, "max_steps": 20}
        )
        with pytest.raises(s.ContractViolationError) as excinfo:
            s.validate_generation(_request(steps=50), narrow)
        assert excinfo.value.details["reason"] == "steps_out_of_range"


class TestSeeds:
    def test_explicit_seed_sequence(self) -> None:
        assert s.resolve_seeds(41, 3) == (41, 42, 43)

    def test_random_seeds_increasing_and_wrap_free(self) -> None:
        rng = random.Random(1234)
        for _ in range(50):
            seeds = s.resolve_seeds(None, 4, rng=rng)
            assert seeds == tuple(sorted(seeds))
            assert seeds[-1] <= s.MAX_SEED

    def test_explicit_overflow_rejected(self) -> None:
        turbo = s.PROFILES[s.ProfileId.Z_IMAGE_TURBO]
        with pytest.raises(s.ContractViolationError) as excinfo:
            s.validate_generation(_request(seed=s.MAX_SEED, count=2), turbo)
        assert excinfo.value.details["reason"] == "seed_overflow"
        with pytest.raises(s.ContractViolationError):
            s.resolve_seeds(s.MAX_SEED, 2)

    def test_boundary_seed_allowed(self) -> None:
        assert s.resolve_seeds(s.MAX_SEED, 1) == (s.MAX_SEED,)


class TestFrozenSpec:
    def _spec(self, **overrides: object) -> s.FrozenRunSpec:
        values: dict[str, object] = {
            "run_id": "ab" * 16,
            "created_at": "2026-09-16T00:00:00+00:00",
            "model": s.FrozenModel(
                registration_id="reg-1",
                repo_id="Tongyi-MAI/Z-Image",
                commit_sha="c" * 40,
                profile=s.ProfileId.Z_IMAGE,
                dtype="bfloat16",
                snapshot_path="/hf/cache/snapshots/" + "c" * 40,
            ),
            "gpu": s.FrozenGpu(uuid="GPU-fake-0001", name="Fake GPU A"),
            "prompt": "p",
            "negative_prompt": None,
            "width": 256,
            "height": 256,
            "steps": 50,
            "guidance": 4.0,
            "image_count": 2,
            "seeds": [10, 11],
            "artifact_ids": ["image-001", "image-002"],
        }
        values.update(overrides)
        return s.FrozenRunSpec(**values)

    def test_sequences_coerced_to_tuples(self) -> None:
        spec = self._spec()
        assert isinstance(spec.seeds, tuple)
        assert isinstance(spec.artifact_ids, tuple)

    def test_in_place_mutation_rejected(self) -> None:
        spec = self._spec()
        with pytest.raises(TypeError):
            spec.seeds[0] = 99  # type: ignore[index]
        with pytest.raises(ValidationError):
            spec.seeds = (99,)  # type: ignore[misc]

    def test_artifact_id_plan(self) -> None:
        assert s.plan_artifact_ids(4) == ("image-001", "image-002", "image-003", "image-004")


class TestResidentAndEvents:
    def _resident(self) -> s.ResidentModel:
        return s.ResidentModel(
            registration_id="reg-1",
            repo_id="Tongyi-MAI/Z-Image",
            commit_sha="c" * 40,
            profile=s.ProfileId.Z_IMAGE,
            dtype="bfloat16",
            gpu=s.FrozenGpu(uuid="GPU-fake-0001", name="Fake GPU A"),
        )

    def test_resident_gpu_required_and_visible(self) -> None:
        with pytest.raises(ValidationError):
            s.ResidentModel(
                registration_id="r",
                repo_id="a/b",
                commit_sha="c" * 40,
                profile=s.ProfileId.Z_IMAGE,
                dtype="bfloat16",
            )
        status = s.RuntimeStatus(
            implementation="real",
            state=s.WorkerState.IDLE,
            resident=self._resident(),
            current_run_id=None,
            queue_depth=0,
        )
        assert status.resident.gpu.uuid == "GPU-fake-0001"

    def test_event_union_discriminates_and_carries_bytes(self) -> None:
        adapter = TypeAdapter(s.RuntimeEvent)
        progress = adapter.validate_python(
            {"event": "run_progress", "run_id": "x", "image_index": 1, "step": 2, "total_steps": 9}
        )
        assert type(progress) is s.RunProgress
        image = adapter.validate_python(
            {
                "event": "image_completed",
                "run_id": "x",
                "artifact_id": "image-001",
                "index": 1,
                "seed": 7,
                "width": 8,
                "height": 8,
                "png": b"\x89PNG",
            }
        )
        assert image.png == b"\x89PNG"

    def test_explicit_worker_states_include_unloaded_and_ejecting(self) -> None:
        assert {state.value for state in s.WorkerState} >= {
            "unloaded",
            "loading",
            "idle",
            "generating",
            "switching",
            "ejecting",
        }
