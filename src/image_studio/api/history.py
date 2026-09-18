"""History read endpoints: listing, run detail, artifacts, thumbnails, metadata."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import FileResponse

from image_studio import schemas
from image_studio.api import state
from image_studio.schemas import ErrorCode, ImageStudioError, RunStatus
from image_studio.storage.repository import NotFoundError

router = APIRouter()


@router.get("/api/generations")
def list_runs(
    request: Request,
    q: str | None = None,
    status: RunStatus | None = None,
    model: str | None = None,
    favorite: bool | None = None,
    trashed: Literal["exclude", "only"] = "exclude",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    runs, total = state(request).coordinator.list_runs(
        q=q,
        status=status,
        repo_id=model,
        favorite=favorite,
        trashed=trashed,
        limit=limit,
        offset=offset,
    )
    return {"runs": [run.model_dump(mode="json") for run in runs], "total": total}


@router.get("/api/generations/queue")
def queue_state(request: Request) -> schemas.QueueState:
    return state(request).coordinator.queue_state()


@router.get("/api/generations/{run_id}")
def run_detail(request: Request, run_id: str) -> schemas.RunDetail:
    detail = state(request).coordinator.get_run(run_id)
    return _with_artifact_urls(detail)


@router.get("/api/generations/{run_id}/artifacts")
def list_artifacts(request: Request, run_id: str) -> dict[str, list[dict[str, object]]]:
    detail = state(request).coordinator.get_run(run_id)
    return {
        "artifacts": [view.model_dump(mode="json") for view in _with_artifact_urls(detail).images]
    }


@router.get("/api/generations/{run_id}/artifacts/{artifact_id}")
def get_artifact(request: Request, run_id: str, artifact_id: str, download: bool = False):
    app_state = state(request)
    row = _run_row(app_state, run_id)
    trashed = row["trashed_at"] is not None
    path = app_state.artifacts.artifact_path(_created_at(row), run_id, artifact_id, trashed=trashed)
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{run_id}-{artifact_id}.png"'
    return FileResponse(path, media_type="image/png", headers=headers)


@router.get("/api/generations/{run_id}/artifacts/{artifact_id}/thumbnail")
def get_thumbnail(request: Request, run_id: str, artifact_id: str) -> Response:
    app_state = state(request)
    row = _run_row(app_state, run_id)
    trashed = row["trashed_at"] is not None
    data = app_state.artifacts.thumbnail(_created_at(row), run_id, artifact_id, trashed=trashed)
    return Response(content=data, media_type="image/webp")


@router.get("/api/generations/{run_id}/metadata")
def get_metadata(request: Request, run_id: str) -> FileResponse:
    app_state = state(request)
    row = _run_row(app_state, run_id)
    trashed = row["trashed_at"] is not None
    path = app_state.artifacts.metadata_path(_created_at(row), run_id, trashed=trashed)
    return FileResponse(
        path,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-metadata.json"'},
    )


# ----- helpers --------------------------------------------------------------


def _run_row(app_state, run_id: str) -> dict[str, object]:
    try:
        return app_state.repository.get_run(run_id)
    except NotFoundError as exc:
        raise ImageStudioError(ErrorCode.NOT_FOUND, f"unknown run {run_id}") from exc


def _created_at(row: dict[str, object]):
    from datetime import datetime

    value = row["created_at"]
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _with_artifact_urls(detail: schemas.RunDetail) -> schemas.RunDetail:
    images = [
        view.model_copy(
            update={
                "url": f"/api/generations/{detail.run_id}/artifacts/{view.artifact_id}",
                "thumbnail_url": (
                    f"/api/generations/{detail.run_id}/artifacts/{view.artifact_id}/thumbnail"
                ),
            }
        )
        for view in detail.images
    ]
    return detail.model_copy(update={"images": images})
