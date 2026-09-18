"""System, runtime, and capability endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from image_studio import schemas
from image_studio.api import state

router = APIRouter()


@router.get("/api/profiles")
def profiles() -> schemas.ProfilesResponse:
    """Capability metadata driving form rendering."""
    return schemas.ProfilesResponse(profiles=list(schemas.PROFILES.values()))


@router.get("/api/system")
def system(request: Request) -> schemas.SystemInfo:
    app_state = state(request)
    settings = app_state.settings
    return schemas.SystemInfo(
        host=settings.host,
        port=settings.port,
        paths=schemas.SystemPaths(
            config_file=str(settings.config_file),
            data_dir=str(settings.data_dir),
            database_file=str(settings.database_file),
            outputs_dir=str(settings.outputs_dir),
            trash_dir=str(settings.trash_dir),
            thumbnails_dir=str(settings.thumbnails_dir),
            log_file=str(settings.log_file),
            hub_cache_dir=str(settings.hub_cache_dir),
        ),
        hf_logged_in=app_state.hf_logged_in,
        hf_username=app_state.hf_username,
        gpus=app_state.coordinator.list_gpus(),
        development=app_state.development,
    )


@router.get("/api/runtime")
def runtime_status(request: Request) -> schemas.RuntimeStatus:
    return state(request).coordinator.runtime_status()


@router.post("/api/runtime/eject", status_code=204)
def eject(request: Request) -> Response:
    state(request).coordinator.eject()
    return Response(status_code=204)
