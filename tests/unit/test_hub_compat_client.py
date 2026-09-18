"""Unit tests for compatibility verdicts and Hub client argument mapping."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from image_studio.hub.client import HubClients
from image_studio.hub.compatibility import check_compatibility, filter_files
from image_studio.testing import FakeHub


def _api_with_files(tmp_path: Path, files: dict[str, bytes | str]) -> FakeHub:
    hub = FakeHub(
        tmp_path / "hub",
        repos=[
            type(
                "Repo",
                (),
                {
                    "repo_id": "test/repo",
                    "files": files,
                    "sha": "d" * 40,
                    "default_branch": "main",
                    "gated": False,
                    "license": "mit",
                    "downloads": 5,
                    "likes": 1,
                    "author": "test",
                },
            )()
        ],
    )
    return hub


def _z_files(**overrides) -> dict[str, bytes | str]:
    import copy

    from image_studio.testing import default_fake_repos

    files = copy.deepcopy(default_fake_repos()[0].files)
    files.update(overrides)
    return files


def test_compatible_repo_passes_with_explicit_profile_choice(tmp_path: Path) -> None:
    hub = _api_with_files(tmp_path, _z_files())
    report = check_compatibility(hub.api, "test/repo", None)
    assert report.structurally_compatible
    assert set(report.selectable_profiles) == {"z-image", "z-image-turbo"}
    assert report.commit_sha == "d" * 40
    assert any("does not guarantee output quality" in note for note in report.notes)


def test_missing_model_index_or_weights(tmp_path: Path) -> None:
    files = _z_files()
    del files["model_index.json"]
    report = check_compatibility(_api_with_files(tmp_path / "a", files).api, "test/repo", None)
    assert not report.structurally_compatible
    files = _z_files()
    for name in list(files):
        if name.endswith(".safetensors"):
            del files[name]
    report = check_compatibility(_api_with_files(tmp_path / "b", files).api, "test/repo", None)
    assert not report.structurally_compatible


def test_remote_code_rejected(tmp_path: Path) -> None:
    report = check_compatibility(
        _api_with_files(tmp_path, _z_files(**{"custom_pipeline.py": "import os"})).api,
        "test/repo",
        None,
    )
    assert not report.structurally_compatible
    assert any("remote code" in finding for finding in report.findings)


def test_wrong_pipeline_class_rejected(tmp_path: Path) -> None:
    files = _z_files(**{"model_index.json": json.dumps({"_class_name": "FluxPipeline"})})
    report = check_compatibility(_api_with_files(tmp_path, files).api, "test/repo", None)
    assert not report.structurally_compatible
    assert any("FluxPipeline" in finding for finding in report.findings)


def test_filter_files_selects_formats_without_duplicates() -> None:
    names = [
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.bin",  # duplicate weight format
        "training_args.bin",
        "README.md",
        "tokenizer/tokenizer.json",
        "vae/vae.safetensors",
    ]
    selected = filter_files(names)
    assert selected == [
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer.json",
        "vae/vae.safetensors",
    ]


class TestHubClientMapping:
    def test_search_and_detail_argument_mapping(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub")
        clients = HubClients(hub.api)
        clients.search_models("z-image", 15)
        call = hub.api.calls["list_models"][-1]
        assert call == {"search": "z-image", "limit": 15}
        detail = clients.model_detail("Tongyi-MAI/Z-Image")
        assert hub.api.calls["model_info"][-1] == {
            "repo_id": "Tongyi-MAI/Z-Image",
            "files_metadata": True,
            "revision": None,
        }
        assert detail.repo_id == "Tongyi-MAI/Z-Image"
        assert any(revision.revision == "main" for revision in detail.revisions)
        assert any(file.path == "model_index.json" for file in detail.files)

    def test_search_limit_clamped(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub")
        HubClients(hub.api).search_models("", 500)
        assert hub.api.calls["list_models"][-1]["limit"] == 100

    def test_whoami_logged_in_and_out(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub", logged_in_as="alice")
        assert HubClients(hub.api).whoami() == (True, "alice")
        hub.api.logged_in_as = None
        assert HubClients(hub.api).whoami() == (False, None)


def _complete_files() -> dict[str, str | bytes]:
    from tests.unit.test_snapshot_validation import _complete_files as _official

    return _official()


class TestRemoteManifestSharedRules:
    """Remote compatibility applies the same declaration rules as local."""

    def _api(self, model_index_text: str):
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        (tmp / "model_index.json").write_text(model_index_text)
        listing = sorted(_complete_files())
        info = SimpleNamespace(
            sha="c" * 40,
            siblings=[SimpleNamespace(rfilename=name, size=1) for name in listing],
        )
        return SimpleNamespace(
            model_info=lambda repo_id, revision=None, files_metadata=False: info,
            hf_hub_download=lambda repo_id, filename, *, revision=None: str(
                tmp / "model_index.json"
            ),
        )

    def test_non_object_model_index_is_typed_not_attribute_error(self) -> None:
        report = check_compatibility(self._api("[1, 2, 3]"), "org/bad", None)
        assert not report.structurally_compatible
        assert any("malformed" in finding for finding in report.findings)

    def test_null_declaration_reported_remotely(self) -> None:
        import json as _json

        index = _json.loads(_complete_files()["model_index.json"])
        index["text_encoder"] = None
        report = check_compatibility(self._api(_json.dumps(index)), "org/null", None)
        assert not report.structurally_compatible
        assert any("does not declare component 'text_encoder'" in f for f in report.findings)

    def test_wrong_component_class_reported_remotely(self) -> None:
        import json as _json

        index = _json.loads(_complete_files()["model_index.json"])
        index["vae"] = ["diffusers", "WrongVAE"]
        report = check_compatibility(self._api(_json.dumps(index)), "org/wrong", None)
        assert not report.structurally_compatible
        assert any("unsupported vae declaration" in f for f in report.findings)
