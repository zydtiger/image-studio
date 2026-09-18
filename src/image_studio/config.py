"""Application settings: XDG resolution, TOML configuration, precedence.

Importing this module never touches the filesystem. Paths are resolved by
:func:`load_settings`, and directories are created only by
:func:`ensure_runtime_dirs` during application startup. Precedence for
server values is CLI argument, then ``config.toml``, then defaults.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "image-studio"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


class ConfigError(ValueError):
    """Raised for unreadable configuration files or invalid values."""


def _env_dir(env: Mapping[str, str], name: str, fallback_home_subdir: str, home: Path) -> Path:
    """Resolve one XDG base directory, honoring only valid absolute paths."""
    raw = env.get(name)
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate
    return home / fallback_home_subdir


@dataclass(frozen=True)
class Settings:
    """Effective application settings with fully resolved paths."""

    host: str
    port: int
    config_file: Path
    data_dir: Path
    database_file: Path
    outputs_dir: Path
    trash_dir: Path
    thumbnails_dir: Path
    log_file: Path
    hub_cache_dir: Path

    @property
    def outputs_dir_is_override(self) -> bool:
        return self.outputs_dir.parent != self.data_dir


def _hub_cache_dir(env: Mapping[str, str], home: Path) -> Path:
    """The official Hugging Face cache location, mirroring the SDK exactly.

    Production (``env is os.environ``) uses the SDK's own import-time
    resolution so the app can never disagree with the installed
    ``huggingface_hub``. An injected environment (tests) cannot reuse that
    import-time constant, so it mirrors the SDK precedence instead —
    ``HF_HUB_CACHE`` > legacy ``HUGGINGFACE_HUB_CACHE`` > ``HF_HOME`` >
    ``XDG_CACHE_HOME/huggingface`` > ``~/.cache/huggingface``, always with
    the ``hub`` suffix and the SDK's ``expanduser``/``expandvars`` treatment.
    Subprocess parity tests keep the mirror honest against the real SDK.
    """
    import os

    if env is os.environ:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)

    default_home = os.path.join(str(home), ".cache")
    hf_home_env = env.get("HF_HOME")
    if hf_home_env is not None:
        hf_home = hf_home_env
    else:
        hf_home = os.path.join(env.get("XDG_CACHE_HOME", default_home), "huggingface")
    hf_home = os.path.expandvars(os.path.expanduser(hf_home))
    default_cache_path = os.path.join(hf_home, "hub")
    legacy = env.get("HUGGINGFACE_HUB_CACHE", default_cache_path)
    current = env.get("HF_HUB_CACHE", legacy)
    return Path(os.path.expandvars(os.path.expanduser(current)))


def _build(
    *,
    host: str,
    port: int,
    outputs_override: Path | None,
    env: Mapping[str, str],
    home: Path,
) -> Settings:
    config_root = _env_dir(env, "XDG_CONFIG_HOME", ".config", home)
    data_root = _env_dir(env, "XDG_DATA_HOME", ".local/share", home)
    cache_root = _env_dir(env, "XDG_CACHE_HOME", ".cache", home)
    state_root = _env_dir(env, "XDG_STATE_HOME", ".local/state", home)

    app_config = config_root / APP_NAME
    app_data = data_root / APP_NAME
    outputs_dir = outputs_override if outputs_override is not None else app_data / "outputs"
    trash_dir = outputs_override / "trash" if outputs_override is not None else app_data / "trash"
    return Settings(
        host=host,
        port=port,
        config_file=app_config / "config.toml",
        data_dir=app_data,
        database_file=app_data / "app.sqlite",
        outputs_dir=outputs_dir,
        trash_dir=trash_dir,
        thumbnails_dir=cache_root / APP_NAME / "thumbnails",
        log_file=state_root / APP_NAME / "logs" / "app.log",
        hub_cache_dir=_hub_cache_dir(env, home),
    )


def _validate_port(port: int) -> int:
    if not 1 <= port <= 65535:
        raise ConfigError(f"port must be 1-65535, got {port}")
    return port


def load_settings(
    *,
    host: str | None = None,
    port: int | None = None,
    outputs_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    config_file: str | Path | None = None,
) -> Settings:
    """Load settings with CLI > TOML > default precedence.

    ``env`` defaults to ``os.environ`` and ``home`` to the user's home
    directory; tests inject isolated values. Nothing is created on disk.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home

    file_host: str | None = None
    file_port: int | None = None
    file_outputs: str | None = None

    settings = _build(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        outputs_override=None,
        env=env,
        home=home,
    )
    cfg_path = Path(config_file).expanduser() if config_file is not None else settings.config_file
    if cfg_path.is_file():
        try:
            with cfg_path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid config file {cfg_path}: {exc}") from exc
        server = data.get("server", {})
        storage = data.get("storage", {})
        if not isinstance(server, dict) or not isinstance(storage, dict):
            raise ConfigError(f"invalid config file {cfg_path}: [server]/[storage] must be tables")
        file_host = server.get("host")
        file_port = server.get("port")
        file_outputs = storage.get("outputs_dir")
        if file_host is not None and not isinstance(file_host, str):
            raise ConfigError("server.host must be a string")
        if file_port is not None and (
            isinstance(file_port, bool) or not isinstance(file_port, int)
        ):
            raise ConfigError("server.port must be an integer")

    effective_host = host or file_host or DEFAULT_HOST
    # explicit values (including 0) reach the validator; defaults apply only
    # when a source is genuinely absent — never via truthiness
    selected_port = port if port is not None else (DEFAULT_PORT if file_port is None else file_port)
    effective_port = _validate_port(selected_port)
    raw_outputs = outputs_dir if outputs_dir is not None else file_outputs
    override = Path(str(raw_outputs)).expanduser() if raw_outputs else None
    if override is not None and not override.is_absolute():
        raise ConfigError(f"storage.outputs_dir must be absolute, got {raw_outputs!r}")

    return _build(
        host=effective_host,
        port=effective_port,
        outputs_override=override,
        env=env,
        home=home,
    )


def ensure_runtime_dirs(settings: Settings) -> None:
    """Create the runtime directory tree. Called only at application startup."""
    for directory in (
        settings.data_dir,
        settings.outputs_dir,
        settings.trash_dir,
        settings.thumbnails_dir,
        settings.log_file.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
