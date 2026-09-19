"""Integration tests for system info, profiles, SPA serving, single instance."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from image_studio.app import create_app
from image_studio.config import load_settings
from image_studio.testing import FakeHub, FakeRuntime
from tests.conftest import build_harness


def test_system_info_reports_paths_gpus_and_development_flags(harness) -> None:
    response = harness.client.get("/api/system")
    assert response.status_code == 200
    info = response.json()
    assert info["host"] == "127.0.0.1" and info["port"] == 7860
    paths = info["paths"]
    assert paths["database_file"].endswith("app.sqlite")
    assert paths["outputs_dir"].endswith("outputs")
    assert paths["trash_dir"].endswith("trash")
    assert paths["hub_cache_dir"]
    assert [gpu["uuid"] for gpu in info["gpus"]] == ["GPU-fake-0001", "GPU-fake-0002"]
    assert info["development"]["fake_runtime"] is True
    assert info["development"]["fake_hub"] is True
    assert info["hf_logged_in"] is True and info["hf_username"] == "tester"


def test_profiles_capability_metadata(harness) -> None:
    profiles = harness.client.get("/api/profiles").json()["profiles"]
    by_id = {profile["profile_id"]: profile for profile in profiles}
    assert set(by_id) == {"z-image", "z-image-turbo", "anima-turbo", "anima-2.9b"}
    assert by_id["z-image-turbo"]["guidance_fixed"] == 0.0
    assert by_id["z-image-turbo"]["negative_prompt_supported"] is False
    assert by_id["z-image"]["default_steps"] == 50


def test_runtime_status_shape(harness) -> None:
    status = harness.client.get("/api/runtime").json()
    assert status["implementation"] == "fake"
    assert status["state"] == "unloaded"
    assert status["resident"] is None
    assert status["queue_depth"] == 0


def test_second_instance_rejected(xdg_env: dict[str, str], tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    first = build_harness(xdg_env, FakeRuntime(), hub)
    with first.client:
        second = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "other-hub"))
        with pytest.raises(Exception, match="another image-studio instance"):
            with second.client:
                pass


def test_frontend_missing_returns_honest_503(
    xdg_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from image_studio import app as app_module

    monkeypatch.setattr(app_module, "_static_root", lambda: tmp_path / "no-assets")
    harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
    with harness.client:
        response = harness.client.get("/")
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "internal"
        assert "frontend assets are not built" in body["error"]["message"]
        deep = harness.client.get("/some/spa/route")
        assert deep.status_code == 503


def test_spa_served_with_built_assets(xdg_env: dict[str, str], tmp_path: Path) -> None:
    from image_studio import app as app_module

    if not (app_module._static_root() / "index.html").is_file():
        pytest.skip("frontend assets not built in this checkout")
    harness = build_harness(xdg_env, FakeRuntime(), FakeHub(tmp_path / "hub"))
    with harness.client:
        root = harness.client.get("/")
        assert root.status_code == 200
        assert "html" in root.headers["content-type"]
        deep = harness.client.get("/generate")
        assert deep.status_code == 200  # SPA fallback to index.html
        api = harness.client.get("/api/not-a-route")
        assert api.status_code == 404
        assert api.json()["error"]["code"] == "not_found"


def test_cli_parser_flags() -> None:
    from image_studio.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["serve", "--host", "0.0.0.0", "--port", "9000", "--fake-runtime", "--fake-hub"]
    )
    assert (args.host, args.port, args.fake_runtime, args.fake_hub) == (
        "0.0.0.0",
        9000,
        True,
        True,
    )
    defaults = parser.parse_args(["serve"])
    assert (defaults.host, defaults.port, defaults.fake_runtime, defaults.fake_hub) == (
        None,
        None,
        False,
        False,
    )


def test_real_runtime_default_factory_cpu_startup(xdg_env: dict[str, str], tmp_path: Path) -> None:
    """Without runtime injection the app builds the real supervisor.

    Startup stays CPU-only: the dispatch thread starts, no worker is spawned
    (no submission happened), and the runtime reports itself as ``real`` and
    unloaded. The fake hub keeps the test offline.
    """
    settings = load_settings(env=xdg_env)
    app = create_app(settings, hub=FakeHub(tmp_path / "hub").stack())
    with TestClient(app) as client:
        status = client.get("/api/runtime").json()
        assert status["implementation"] == "real"
        assert status["state"] == "unloaded"
        assert status["resident"] is None
        system = client.get("/api/system").json()
        assert system["development"]["fake_runtime"] is False
        assert isinstance(system["gpus"], list)


def test_method_not_allowed_is_client_error_with_allow_header(harness) -> None:
    response = harness.client.delete("/api/system")
    assert response.status_code == 405
    body = response.json()
    assert body["error"]["code"] == "validation"
    assert response.headers.get("allow") == "GET"
