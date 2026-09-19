"""Download job endpoints: create, list, retry, queued cancellation."""

from __future__ import annotations

from fastapi import APIRouter, Request

from image_studio import schemas
from image_studio.api import state
from image_studio.storage.repository import NotFoundError

router = APIRouter()


@router.post("/api/downloads", status_code=202)
def create_download(request: Request, body: schemas.DownloadCreate) -> schemas.DownloadJob:
    app_state = state(request)
    from image_studio.hub.anima import resolve_sources

    sources = resolve_sources(app_state.hub, body.repo_id, body.revision, body.profile)
    job = app_state.repository.create_download(
        repo_id=body.repo_id, revision=body.revision, profile=body.profile, sources=sources
    )
    app_state.downloads.enqueue(job.id)
    return app_state.repository.get_download(job.id)


@router.get("/api/downloads")
def list_downloads(request: Request) -> dict[str, list[schemas.DownloadJob]]:
    return {"jobs": state(request).repository.list_downloads()}


@router.post("/api/downloads/{job_id}/retry", status_code=202)
def retry_download(request: Request, job_id: str) -> schemas.DownloadJob:
    app_state = state(request)
    _require_job(app_state, job_id)
    app_state.downloads.retry(job_id)
    return app_state.repository.get_download(job_id)


@router.post("/api/downloads/{job_id}/cancel", status_code=202)
def cancel_download(request: Request, job_id: str) -> schemas.DownloadJob:
    app_state = state(request)
    _require_job(app_state, job_id)
    return app_state.downloads.cancel(job_id)


def _require_job(app_state, job_id: str) -> None:
    try:
        app_state.repository.get_download(job_id)
    except NotFoundError as exc:
        raise schemas.ImageStudioError(
            schemas.ErrorCode.NOT_FOUND, f"unknown download job {job_id}"
        ) from exc
