"""Check the built wheel through an isolated server and real-backend browser tests."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    wheel = ROOT / "dist" / f"image_studio-{version}-py3-none-any.whl"
    if not wheel.is_file():
        raise SystemExit("Build the frontend and wheel first: just e2e-check")
    logs = ROOT / "tests/e2e/test-results/integration-server.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / ".artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="integration-", dir=artifacts) as temporary:
        run_dir = Path(temporary)
        env = os.environ.copy()
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            env[name] = str(run_dir / name.lower())
        env.update(
            HF_HOME=str(run_dir / "hf"),
            HF_HUB_CACHE=str(run_dir / "hf/hub"),
            HUGGINGFACE_HUB_CACHE=str(run_dir / "hf/hub"),
            HF_HUB_OFFLINE="1",
            CUDA_VISIBLE_DEVICES="",
            TMPDIR=temporary,
        )
        env.pop("PYTHONPATH", None)
        # Install locked dependencies, then replace the editable package with the wheel.
        # Running outside the checkout prevents source files masking packaging defects.
        venv = run_dir / "venv"
        subprocess.run(
            ["uv", "sync", "--locked", "--no-dev"],
            cwd=ROOT,
            env={
                key: value
                for key, value in {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}.items()
                if key != "VIRTUAL_ENV"
            },
            check=True,
        )
        python = str(venv / "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", python, "--no-deps", "--reinstall", str(wheel)],
            check=True,
        )
        subprocess.run(
            [
                python,
                "-c",
                (
                    "from pathlib import Path; import image_studio; "
                    f"assert Path(image_studio.__file__).is_relative_to({str(venv)!r}); "
                    "assert (Path(image_studio.__file__).parent / 'web/static/index.html').is_file()"
                ),
            ],
            cwd=run_dir,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(venv / "bin/image-studio"), "serve", "--help"],
            cwd=run_dir,
            env=env,
            check=True,
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with logs.open("w") as log:
            server = subprocess.Popen(
                [
                    str(venv / "bin/image-studio"),
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--fake-runtime",
                    "--fake-hub",
                ],
                cwd=run_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 60
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited; see {logs}")
                    try:
                        with opener.open(f"{base}/api/runtime", timeout=1) as response:
                            assert json.load(response)["implementation"] == "fake"
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"Server readiness timed out; see {logs}") from None
                        time.sleep(0.2)
                subprocess.run(
                    ["pnpm", "--dir", "tests/e2e", "run", "test", "--workers=1"],
                    cwd=ROOT,
                    env={**os.environ, "E2E_BASE_URL": base},
                    check=True,
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
