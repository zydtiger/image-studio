"""Unit tests for XDG resolution, TOML precedence, and override validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from image_studio.config import ConfigError, load_settings


def test_default_paths_use_standard_xdg_locations(tmp_path: Path) -> None:
    env = {
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    settings = load_settings(env=env, home=tmp_path)
    assert settings.config_file == tmp_path / "cfg" / "image-studio" / "config.toml"
    assert settings.database_file == tmp_path / "data" / "image-studio" / "app.sqlite"
    assert settings.outputs_dir == tmp_path / "data" / "image-studio" / "outputs"
    assert settings.trash_dir == tmp_path / "data" / "image-studio" / "trash"
    assert settings.thumbnails_dir == tmp_path / "cache" / "image-studio" / "thumbnails"
    assert settings.log_file == tmp_path / "state" / "image-studio" / "logs" / "app.log"


def test_fallback_when_xdg_vars_missing_or_relative(tmp_path: Path) -> None:
    settings = load_settings(env={"XDG_DATA_HOME": "relative/path"}, home=tmp_path)
    assert settings.data_dir == tmp_path / ".local" / "share" / "image-studio"


def test_hf_cache_env_precedence(tmp_path: Path) -> None:
    explicit = load_settings(env={"HF_HUB_CACHE": "/explicit/hub"}, home=tmp_path)
    assert str(explicit.hub_cache_dir) == "/explicit/hub"
    home_only = load_settings(env={"HF_HOME": "/hf-home"}, home=tmp_path)
    assert str(home_only.hub_cache_dir) == "/hf-home/hub"
    default = load_settings(env={}, home=tmp_path)
    assert default.hub_cache_dir == tmp_path / ".cache" / "huggingface" / "hub"


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg" / "image-studio"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        '[server]\nhost = "0.0.0.0"\nport = 9000\n[storage]\noutputs_dir = "/data/images"\n'
    )
    settings = load_settings(
        env={
            "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
        },
        home=tmp_path,
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.outputs_dir == Path("/data/images")
    assert settings.trash_dir == Path("/data/images/trash")
    assert settings.database_file == tmp_path / "data" / "image-studio" / "app.sqlite"


def test_cli_beats_toml_beats_defaults(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg" / "image-studio"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[server]\nhost = "10.0.0.1"\nport = 9000\n')
    settings = load_settings(
        env={"XDG_CONFIG_HOME": str(tmp_path / "cfg")},
        home=tmp_path,
        host="192.168.1.2",
        port=8080,
    )
    assert settings.host == "192.168.1.2"
    assert settings.port == 8080


def test_invalid_port_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(env={}, home=tmp_path, port=70000)
    with pytest.raises(ConfigError):
        load_settings(env={}, home=tmp_path, port=0)


def test_relative_outputs_override_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(env={}, home=tmp_path, outputs_dir="relative/outputs")


def test_invalid_toml_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg" / "image-studio"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("not [ valid toml")
    with pytest.raises(ConfigError):
        load_settings(env={"XDG_CONFIG_HOME": str(tmp_path / "cfg")}, home=tmp_path)


def test_import_and_resolution_create_no_directories(tmp_path: Path) -> None:
    load_settings(env={"XDG_DATA_HOME": str(tmp_path / "fresh")}, home=tmp_path)
    assert not (tmp_path / "fresh").exists()


class TestHubCacheSdkParity:
    """The injected-environment mirror must match the installed SDK exactly.

    Each scenario runs an isolated subprocess so the SDK's import-time
    constants resolve under that exact environment, then compares
    ``huggingface_hub.constants.HF_HUB_CACHE`` with our ``load_settings``.
    """

    @staticmethod
    def _sdk_and_app(env: dict[str, str], home: str) -> tuple[str, str]:
        import subprocess
        import sys

        code = (
            "import os;"
            "from huggingface_hub.constants import HF_HUB_CACHE as S;"
            "from image_studio.config import load_settings;"
            # dict(os.environ) is NOT os.environ, so this exercises the
            # injected-environment mirror rather than the production branch
            "print(S); print(load_settings(env=dict(os.environ)).hub_cache_dir)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),  # repository root, any clone
            check=True,
        )
        sdk, app = result.stdout.strip().splitlines()
        return sdk, app

    @pytest.mark.parametrize(
        "scenario",
        [
            {"name": "default"},
            {"name": "xdg", "XDG_CACHE_HOME": "/tmp/xdg-hf/hub-probe"},
            {"name": "hf-home", "HF_HOME": "/tmp/hf-home-probe"},
            {"name": "legacy", "HUGGINGFACE_HUB_CACHE": "/tmp/legacy-hf-probe"},
            {"name": "current", "HF_HUB_CACHE": "/tmp/current-hf-probe"},
            {
                "name": "current-beats-legacy",
                "HF_HUB_CACHE": "/tmp/current-wins",
                "HUGGINGFACE_HUB_CACHE": "/tmp/legacy-loses",
            },
            {
                "name": "legacy-beats-hf-home",
                "HUGGINGFACE_HUB_CACHE": "/tmp/legacy-wins",
                "HF_HOME": "/tmp/hf-home-loses",
            },
            {
                "name": "hf-home-beats-xdg",
                "HF_HOME": "/tmp/hf-home-wins",
                "XDG_CACHE_HOME": "/tmp/xdg-loses",
            },
            {"name": "expandvars", "HF_HUB_CACHE": "$PARITY_VAR/nested"},
            {"name": "expanduser", "HF_HUB_CACHE": "~/hf-cache-probe"},
        ],
    )
    def test_hub_cache_matches_sdk(self, scenario: dict) -> None:
        import os

        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")
        }
        env.update({k: v for k, v in scenario.items() if k != "name"})
        env.setdefault("PARITY_VAR", "/tmp/expanded-probe")
        sdk, app = self._sdk_and_app(env, env.get("HOME", ""))
        assert app == sdk, f"{scenario['name']}: app={app} sdk={sdk}"


def test_boolean_port_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg" / "image-studio"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("[server]\nport = true\n")
    with pytest.raises(ConfigError, match="integer"):
        load_settings(env={"XDG_CONFIG_HOME": str(tmp_path / "cfg")}, home=tmp_path)


def test_explicit_zero_port_not_silently_defaulted(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg" / "image-studio"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("[server]\nport = 0\n")
    with pytest.raises(ConfigError, match="got 0"):
        load_settings(env={"XDG_CONFIG_HOME": str(tmp_path / "cfg")}, home=tmp_path)
