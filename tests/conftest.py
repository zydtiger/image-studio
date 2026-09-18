"""Shared fixtures: isolated XDG environments, fake hub, and app harnesses.

Ordinary tests never touch real XDG directories, never reach the network,
download no weights, and claim no GPUs. Every app instance gets a private
temporary data tree; restart scenarios reuse one environment across two
sequentially entered harnesses.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from image_studio.app import create_app
from image_studio.config import Settings, load_settings
from image_studio.testing import FakeHub, FakeRuntime

TERMINAL_STATUSES = ("completed", "partial", "failed", "cancelled", "interrupted")


@pytest.fixture
def xdg_env(tmp_path: Path) -> dict[str, str]:
    """Isolated XDG + Hugging Face environment values."""
    return {
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "HF_HOME": str(tmp_path / "hf"),
    }


@pytest.fixture
def settings(xdg_env: dict[str, str]) -> Settings:
    return load_settings(env=xdg_env)


class AppHarness:
    """A running app over temporary paths with an injected runtime and hub."""

    def __init__(self, client: TestClient, runtime: object, hub: FakeHub) -> None:
        self.client = client
        self.runtime = runtime
        self.hub = hub

    def register_model(self, repo_id: str = "Tongyi-MAI/Z-Image", profile: str = "z-image") -> dict:
        self.hub.seed_snapshot(repo_id)
        response = self.client.post("/api/models", json={"repo_id": repo_id, "profile": profile})
        assert response.status_code == 201, response.text
        return response.json()

    def submit(
        self,
        registration_id: str,
        *,
        gpu_uuid: str = "GPU-fake-0001",
        prompt: str = "a test prompt",
        **overrides: object,
    ) -> dict:
        body: dict[str, object] = {
            "registration_id": registration_id,
            "gpu_uuid": gpu_uuid,
            "prompt": prompt,
            "width": 256,
            "height": 256,
        }
        body.update(overrides)
        response = self.client.post("/api/generations", json=body)
        assert response.status_code == 202, response.text
        return response.json()

    def wait_terminal(self, run_id: str, timeout: float = 5.0) -> dict:
        import time

        deadline = time.monotonic() + timeout
        detail = {}
        while time.monotonic() < deadline:
            detail = self.client.get(f"/api/generations/{run_id}").json()
            if detail["status"] in TERMINAL_STATUSES:
                return detail
            time.sleep(0.01)
        raise AssertionError(f"run {run_id} did not finish: {detail}")


def build_harness(env: dict[str, str], runtime: object, hub: FakeHub) -> AppHarness:
    """Create an unentered harness; callers use ``with harness.client:``."""
    settings = load_settings(env=env)
    app = create_app(settings, runtime=runtime, hub=hub.stack())
    return AppHarness(TestClient(app), runtime, hub)


@pytest.fixture
def harness(xdg_env: dict[str, str], tmp_path: Path) -> Iterator[AppHarness]:
    runtime = FakeRuntime()
    hub = FakeHub(tmp_path / "fake-hub")
    harness = build_harness(xdg_env, runtime, hub)
    with harness.client:
        yield harness
