"""Application factory: composition root for storage, hub, runtime, and API.

The API process never owns CUDA tensors; inference runs behind the
``schemas.Runtime`` protocol. Production defaults instantiate the real
inference supervisor; fake runtimes enter only through explicit
development/testing injection and are visibly identified in API responses.
"""

from __future__ import annotations

import fcntl
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from image_studio import schemas
from image_studio.config import Settings, ensure_runtime_dirs, load_settings
from image_studio.hub.client import HubStack
from image_studio.hub.downloads import DownloadEngine
from image_studio.runtime import RuntimeCoordinator
from image_studio.storage import database
from image_studio.storage.artifacts import ArtifactError, ArtifactStore
from image_studio.storage.repository import NotFoundError, Repository

logger = logging.getLogger(__name__)

_STATUS_BY_CODE = {
    schemas.ErrorCode.VALIDATION: 422,
    schemas.ErrorCode.CACHE_INCOMPLETE: 422,
    schemas.ErrorCode.UNSUPPORTED_MODEL: 422,
    schemas.ErrorCode.NOT_FOUND: 404,
    schemas.ErrorCode.REVISION_NOT_FOUND: 404,
    schemas.ErrorCode.CONFLICT: 409,
    schemas.ErrorCode.GATED_MODEL: 403,
    schemas.ErrorCode.HUB_AUTH_REQUIRED: 403,
    schemas.ErrorCode.HUB_UNREACHABLE: 502,
    schemas.ErrorCode.WORKER_ERROR: 500,
    schemas.ErrorCode.STORAGE_ERROR: 500,
    schemas.ErrorCode.INTERNAL: 500,
}


@dataclass
class AppState:
    settings: Settings
    repository: Repository
    artifacts: ArtifactStore
    coordinator: RuntimeCoordinator
    hub: HubStack
    downloads: DownloadEngine
    hf_logged_in: bool
    hf_username: str | None
    development: schemas.DevelopmentFlags


class SingleInstanceLock:
    """Prevents two server instances from sharing one application data directory."""

    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._handle = (data_dir / "app.lock").open("a+")

    def acquire(self) -> None:
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise schemas.ImageStudioError(
                schemas.ErrorCode.CONFLICT,
                "another image-studio instance already uses this data directory",
            ) from exc

    def release(self) -> None:
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()


def _build_real_runtime() -> schemas.Runtime:
    """Instantiate the real inference supervisor.

    ``create_runtime`` starts the dispatch thread immediately and defaults
    to the real worker spawn target; the heavy inference stack loads only
    inside the spawned worker process, so this stays CPU-safe at startup.
    """
    from image_studio.inference import create_runtime

    return create_runtime()


def create_app(
    settings: Settings | None = None,
    *,
    runtime: schemas.Runtime | None = None,
    hub: HubStack | None = None,
) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lock = SingleInstanceLock(settings.data_dir)
        lock.acquire()
        ensure_runtime_dirs(settings)
        connection = database.connect(settings.database_file)
        database.migrate(connection)
        hub_stack = hub if hub is not None else HubStack.real(settings.hub_cache_dir)
        repository = Repository(connection, hub_cache_dir=hub_stack.cache_dir)
        interrupted, paused = repository.reconcile_runs()
        repository.reconcile_downloads()
        if interrupted or paused:
            logger.info(
                "startup reconciliation: %d interrupted, %d paused runs",
                interrupted,
                paused,
            )
        logged_in, username = hub_stack.clients.whoami()
        active_runtime = runtime if runtime is not None else _build_real_runtime()
        artifacts = ArtifactStore(
            outputs_dir=settings.outputs_dir,
            trash_dir=settings.trash_dir,
            thumbnails_dir=settings.thumbnails_dir,
        )
        coordinator = RuntimeCoordinator(repository, artifacts, active_runtime)
        if paused:
            coordinator.mark_queue_paused()
        if interrupted:
            coordinator.rewrite_metadata_for_interrupted()
        engine = DownloadEngine(repository, hub_stack)
        engine.start()
        app.state.app_state = AppState(
            settings=settings,
            repository=repository,
            artifacts=artifacts,
            coordinator=coordinator,
            hub=hub_stack,
            downloads=engine,
            hf_logged_in=logged_in,
            hf_username=username,
            development=schemas.DevelopmentFlags(
                fake_runtime=_is_fake(active_runtime),
                fake_hub=hub is not None and _is_fake_hub(hub),
            ),
        )
        try:
            yield
        finally:
            engine.stop()
            coordinator.shutdown()
            connection.close()
            lock.release()

    app = FastAPI(title="Image Studio", version="0.1.0", lifespan=lifespan)
    _install_error_handlers(app)
    _include_routers(app)
    _install_static(app)
    return app


def _allowed_methods(request: Request) -> list[str]:
    """Methods the router accepts for this path (for the 405 Allow header).

    Derived from the (cached) OpenAPI schema: parameterized templates are
    matched against the concrete request path.
    """
    import re

    app = request.scope.get("app")
    if app is None:
        return []
    try:
        paths = (app.openapi().get("paths") or {}).items()
    except Exception:  # noqa: BLE001 - the header is best-effort
        return []
    allowed: set[str] = set()
    for template, operations in paths:
        pattern = re.sub(r"\{[^}]+\}", "[^/]+", template)
        if re.fullmatch(pattern, request.url.path):
            allowed |= {
                method.upper() for method in operations if method.upper() not in ("HEAD", "OPTIONS")
            }
    return sorted(allowed)


def _is_fake(runtime: schemas.Runtime) -> bool:
    from image_studio.testing import FakeRuntime

    return isinstance(runtime, FakeRuntime)


def _is_fake_hub(hub: HubStack) -> bool:
    from image_studio.testing import FakeHubApi

    return isinstance(hub.api, FakeHubApi)


def _error_response(code: schemas.ErrorCode, message: str, details, status: int) -> JSONResponse:
    body = schemas.ErrorBody(error=schemas.ErrorInfo(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(schemas.ImageStudioError)
    async def typed_error(request: Request, exc: schemas.ImageStudioError):
        status = _STATUS_BY_CODE.get(exc.code, 500)
        return _error_response(exc.code, exc.message, exc.details, status)

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError):
        return _error_response(schemas.ErrorCode.NOT_FOUND, str(exc), None, 404)

    @app.exception_handler(ArtifactError)
    async def artifact_error(request: Request, exc: ArtifactError):
        return _error_response(schemas.ErrorCode.NOT_FOUND, str(exc), None, 404)

    @app.exception_handler(FileNotFoundError)
    async def file_missing(request: Request, exc: FileNotFoundError):
        return _error_response(schemas.ErrorCode.NOT_FOUND, str(exc), None, 404)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return _error_response(
            schemas.ErrorCode.VALIDATION,
            "invalid request",
            {"errors": [error["msg"] for error in exc.errors()]},
            422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            code = schemas.ErrorCode.NOT_FOUND
        elif 400 <= exc.status_code < 500:
            # client-side HTTP failures (405 method mismatch, 413 payload,
            # ...): a real client error, never reported as an internal fault
            code = schemas.ErrorCode.VALIDATION
        else:
            code = schemas.ErrorCode.INTERNAL
        headers = getattr(exc, "headers", None)
        if exc.status_code == 405 and not headers:
            headers = {"Allow": ", ".join(_allowed_methods(request))}
        return JSONResponse(
            status_code=exc.status_code,
            content=schemas.ErrorBody(
                error=schemas.ErrorInfo(code=code, message=str(exc.detail))
            ).model_dump(mode="json"),
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error_response(schemas.ErrorCode.INTERNAL, "internal server error", None, 500)


def _include_routers(app: FastAPI) -> None:
    from image_studio.api import downloads, generations, history, models, system

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(downloads.router)
    app.include_router(generations.router)
    app.include_router(history.router)


class SPAStaticFiles(StaticFiles):
    """Serve built frontend assets with an index.html fallback for SPA routes."""

    async def get_response(self, path: str, scope):

        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            if scope.get("path", "").startswith("/api"):
                return _error_response(schemas.ErrorCode.NOT_FOUND, "unknown API path", None, 404)
            return await super().get_response("index.html", scope)


def _static_root() -> Path:
    """Built frontend assets; written by `just frontend`."""
    return Path(__file__).parent / "web" / "static"


def _install_static(app: FastAPI) -> None:
    static_root = _static_root()
    if (static_root / "index.html").is_file():
        app.mount("/", SPAStaticFiles(directory=static_root, html=True), name="spa")
        return

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend_missing(path: str):
        if path.startswith("api"):
            return _error_response(schemas.ErrorCode.NOT_FOUND, "unknown API path", None, 404)
        return _error_response(
            schemas.ErrorCode.INTERNAL,
            "frontend assets are not built; run `just frontend`",
            None,
            503,
        )
