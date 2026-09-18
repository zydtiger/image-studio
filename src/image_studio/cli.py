"""Command-line entry point: ``image-studio serve``."""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-studio",
        description="Local image generation interface backed by Diffusers",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default 7860)")
    serve.add_argument(
        "--fake-runtime",
        action="store_true",
        help="development only: use the deterministic fake inference runtime",
    )
    serve.add_argument(
        "--fake-hub",
        action="store_true",
        help="development only: use the fake Hugging Face catalog and an isolated temporary cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "serve":
        build_parser().error("unknown command")
        return 2

    from image_studio.app import create_app
    from image_studio.config import load_settings

    settings = load_settings(host=args.host, port=args.port)

    runtime = None
    hub = None
    if args.fake_runtime:
        from image_studio.testing import FakeRuntime

        runtime = FakeRuntime()
    if args.fake_hub:
        import tempfile
        from pathlib import Path

        from image_studio.testing import FakeHub

        fake_hub = FakeHub(Path(tempfile.mkdtemp(prefix="image-studio-fake-hub-")))
        hub = fake_hub.stack()

    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(settings.log_file, encoding="utf-8")],
    )

    app = create_app(settings, runtime=runtime, hub=hub)

    print(f"Image Studio listening on http://{settings.host}:{settings.port}")
    if args.fake_runtime or args.fake_hub:
        print(
            "DEVELOPMENT MODE: "
            + ("fake runtime " if args.fake_runtime else "")
            + ("fake hub" if args.fake_hub else "")
            + " — not a production inference stack"
        )

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
