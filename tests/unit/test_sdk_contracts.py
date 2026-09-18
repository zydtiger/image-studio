"""Contract tests against the installed HfApi (no network)."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

from image_studio.hub.client import HubClients
from image_studio.hub.compatibility import check_compatibility
from image_studio.schemas import ErrorCode, ImageStudioError


def test_compat_config_fetch_matches_real_hf_hub_download_signature(tmp_path):
    """The compatibility path calls the real SDK keyword-only contract.

    ``create_autospec`` of the real bound method enforces the installed
    signature: a positional ``revision`` (the P1 reproduction) raises
    TypeError inside the fetch, the check degrades to "incompatible", and
    this test fails. All configuration inspection must use the resolved
    fixed commit so a moving branch cannot drift between fetches.
    """
    from huggingface_hub import HfApi

    model_index = tmp_path / "model_index.json"
    model_index.write_text(
        json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    listing = _shard_names()
    real_api = HfApi()
    downloader = create_autospec(real_api.hf_hub_download, return_value=str(model_index))
    api = SimpleNamespace(
        model_info=Mock(
            return_value=SimpleNamespace(
                sha="f" * 40,
                siblings=[SimpleNamespace(rfilename=name, size=1) for name in listing],
            )
        ),
        hf_hub_download=downloader,
    )
    report = check_compatibility(api, "org/model", "main")
    assert report.structurally_compatible, report.findings
    downloader.assert_called_once_with("org/model", "model_index.json", revision="f" * 40)


def _shard_names() -> list[str]:
    from tests.unit.test_snapshot_validation import _complete_files, _sharded

    return sorted(_sharded(_complete_files()))


def test_fake_hub_api_download_signature_matches_real():
    """The fake must be keyword-only so it catches positional misuse."""
    from huggingface_hub import HfApi

    from image_studio.testing import FakeHubApi

    real = inspect.signature(HfApi.hf_hub_download)
    fake = inspect.signature(FakeHubApi.hf_hub_download)
    assert fake.parameters["revision"].kind is real.parameters["revision"].kind


def test_fake_hub_list_models_is_lazy_like_real():
    from huggingface_hub import HfApi

    from image_studio.testing import FakeHubApi

    assert inspect.isgeneratorfunction(inspect.unwrap(HfApi.list_models))
    assert inspect.isgeneratorfunction(FakeHubApi.list_models)


def test_search_consumes_lazy_iterable_inside_error_boundary():
    """A failure raised during iteration must still translate, not escape."""

    def lazy_fail_late(**kwargs: Any):
        yield from ()
        raise ConnectionError("iteration-time failure")

    api = SimpleNamespace(list_models=lazy_fail_late)
    clients = HubClients(api)
    try:
        clients.search_models("q", 5)
        raise AssertionError("expected a translated error")
    except ImageStudioError as exc:
        assert exc.code is ErrorCode.HUB_UNREACHABLE


def test_model_detail_reads_real_gitrefs_shape():
    """model_detail against real GitRefs/GitRefInfo (installed SDK classes).

    The installed SDK exposes name/ref/target_commit and no sha/default;
    reading the invented fields (the P1 reproduction) crashes this test.
    """
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    refs = GitRefs(
        branches=[
            GitRefInfo(name="main", ref="refs/heads/main", target_commit="a" * 40),
            GitRefInfo(name="dev", ref="refs/heads/dev", target_commit="b" * 40),
        ],
        tags=[GitRefInfo(name="v1", ref="refs/tags/v1", target_commit="c" * 40)],
        converts=[],
    )
    api = SimpleNamespace(
        model_info=Mock(
            return_value=SimpleNamespace(
                id="org/model",
                sha="a" * 40,
                siblings=[SimpleNamespace(rfilename="model_index.json", size=1)],
                author="org",
                private=False,
                gated=False,
                downloads=1,
                likes=2,
                lastModified=None,
                pipeline_tag="text-to-image",
                card_data=None,
            )
        ),
        list_repo_refs=Mock(return_value=refs),
    )
    detail = HubClients(api).model_detail("org/model")
    by_name = {revision.revision: revision.commit_sha for revision in detail.revisions}
    assert by_name == {"main": "a" * 40, "dev": "b" * 40, "v1": "c" * 40}
    # honest default derivation: the branch matching model_info.sha
    assert detail.default_revision == "main"


def test_model_detail_default_revision_requires_unique_branch_match():
    """Two branches sharing model_info.sha: none is justified as default."""
    from huggingface_hub.hf_api import GitRefInfo, GitRefs

    def detail_for(branches: list):
        info = SimpleNamespace(
            id="org/model",
            sha="a" * 40,
            siblings=[],
            author=None,
            private=False,
            gated=False,
            downloads=None,
            likes=None,
            lastModified=None,
            pipeline_tag=None,
            card_data=None,
        )
        api = SimpleNamespace(
            model_info=lambda repo_id, revision=None, files_metadata=False: info,
            list_repo_refs=lambda repo_id: GitRefs(branches=branches, tags=[], converts=[]),
        )
        return HubClients(api).model_detail("org/model")

    branch = lambda name, commit: GitRefInfo(  # noqa: E731 - local helper
        name=name, ref=f"refs/heads/{name}", target_commit=commit
    )

    unique = detail_for([branch("main", "a" * 40), branch("dev", "b" * 40)])
    assert unique.default_revision == "main"

    ambiguous = detail_for([branch("main", "a" * 40), branch("release", "a" * 40)])
    assert ambiguous.default_revision is None
