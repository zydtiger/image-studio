"""Generation mutation endpoints: submit, cancel, queue, favorites, trash."""

from __future__ import annotations

from fastapi import APIRouter, Request

from image_studio import schemas
from image_studio.api import state

router = APIRouter()


@router.post("/api/generations", status_code=202)
def submit_generation(request: Request, body: schemas.GenerationRequest) -> schemas.RunDetail:
    return state(request).coordinator.submit(body)


@router.post("/api/generations/queue/resume")
def resume_queue(request: Request) -> schemas.QueueState:
    return state(request).coordinator.resume_queue()


@router.post("/api/generations/{run_id}/cancel", status_code=202)
def cancel_generation(request: Request, run_id: str) -> schemas.RunDetail:
    return state(request).coordinator.cancel(run_id)


@router.patch("/api/generations/{run_id}")
def update_run(request: Request, run_id: str, body: schemas.FavoriteUpdate) -> schemas.RunSummary:
    return state(request).coordinator.favorite(run_id, body.favorite)


@router.post("/api/generations/{run_id}/trash")
def trash_run(request: Request, run_id: str) -> schemas.RunSummary:
    return state(request).coordinator.trash(run_id)


@router.post("/api/generations/{run_id}/restore")
def restore_run(request: Request, run_id: str) -> schemas.RunSummary:
    return state(request).coordinator.restore(run_id)
