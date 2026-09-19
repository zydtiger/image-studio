"""Model library endpoints: Hub discovery, cache inspection, registrations."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from image_studio import schemas
from image_studio.api import state
from image_studio.hub import cache as hub_cache
from image_studio.hub.anima import make_sources, model_problems
from image_studio.hub.client import DEFAULT_SEARCH_LIMIT
from image_studio.hub.compatibility import check_compatibility
from image_studio.schemas import ErrorCode, ImageStudioError

router = APIRouter()


@router.get("/api/hub/models")
def hub_search(
    request: Request, q: str = "", limit: int = DEFAULT_SEARCH_LIMIT
) -> dict[str, list[schemas.HubModelSummary]]:
    app_state = state(request)
    return {"results": app_state.hub.clients.search_models(q, limit)}


@router.get("/api/hub/models/{repo_id:path}/compatibility")
def hub_compatibility(
    request: Request, repo_id: str, revision: str | None = None
) -> schemas.CompatibilityReport:
    return check_compatibility(state(request).hub.api, repo_id, revision)


@router.get("/api/hub/models/{repo_id:path}")
def hub_model_detail(request: Request, repo_id: str) -> schemas.HubModelDetail:
    return state(request).hub.clients.model_detail(repo_id)


@router.get("/api/cache/models")
def cached_models(request: Request) -> dict[str, list[schemas.CachedRepo]]:
    app_state = state(request)
    return {"repos": hub_cache.scan(app_state.hub.cache_dir)}


@router.post("/api/models", status_code=201)
def register_model(request: Request, body: schemas.RegistrationCreate) -> schemas.ModelRegistration:
    app_state = state(request)
    hit = hub_cache.find_snapshot(body.repo_id, body.revision, app_state.hub.cache_dir)
    if hit is None:
        raise ImageStudioError(
            ErrorCode.CACHE_INCOMPLETE,
            f"no cached snapshot for {body.repo_id}; download it first",
            {"repo_id": body.repo_id},
        )
    sources = make_sources(app_state.hub.cache_dir, body.repo_id, hit.commit_sha, body.profile)
    problems = model_problems(hit.path, body.repo_id, body.profile, sources)
    if problems:
        raise ImageStudioError(
            hub_cache.problem_codes(problems),
            "cached model does not satisfy its component manifest",
            {**hub_cache.problem_details(problems), "snapshot_path": str(hit.path)},
        )
    try:
        return app_state.repository.create_registration(
            repo_id=body.repo_id,
            commit_sha=hit.commit_sha,
            profile=body.profile,
            snapshot_path=str(hit.path),
            display_name=body.display_name,
            sources=sources,
        )
    except ValueError as exc:
        raise ImageStudioError(ErrorCode.CONFLICT, str(exc)) from exc


@router.get("/api/models")
def list_models(request: Request) -> dict[str, list[schemas.ModelRegistration]]:
    from pathlib import Path

    repository = state(request).repository
    for registration in repository.list_registrations():
        if registration.sources:
            problems = model_problems(
                Path(registration.snapshot_path),
                registration.repo_id,
                registration.profile,
                registration.sources,
            )
            repository.set_registration_status(
                registration.id,
                schemas.RegistrationStatus.MISSING_FILES
                if problems
                else schemas.RegistrationStatus.READY,
                [problem.detail for problem in problems],
            )
    return {"registrations": repository.list_registrations()}


@router.patch("/api/models/{registration_id}")
def update_model(
    request: Request, registration_id: str, body: schemas.RegistrationUpdate
) -> schemas.ModelRegistration:
    app_state = state(request)
    if body.profile is not None:
        _guard_registration_unused(app_state, registration_id)
        from pathlib import Path

        registration = app_state.repository.get_registration(registration_id)
        if body.profile != registration.profile:
            problems = model_problems(
                Path(registration.snapshot_path or ""),
                registration.repo_id,
                body.profile,
                registration.sources,
            )
            if problems:
                raise ImageStudioError(
                    hub_cache.problem_codes(problems),
                    "profile does not match the cached model",
                    hub_cache.problem_details(problems),
                )
    try:
        return app_state.repository.update_registration(
            registration_id, display_name=body.display_name, profile=body.profile
        )
    except ValueError as exc:
        raise ImageStudioError(ErrorCode.CONFLICT, str(exc)) from exc


@router.delete("/api/models/{registration_id}", status_code=204)
def delete_model(request: Request, registration_id: str) -> Response:
    app_state = state(request)
    _guard_registration_unused(app_state, registration_id)
    app_state.repository.delete_registration(registration_id)
    return Response(status_code=204)


def _guard_registration_unused(app_state, registration_id: str) -> None:
    from image_studio.storage.repository import NotFoundError

    try:
        app_state.repository.get_registration(registration_id)
    except NotFoundError as exc:
        raise ImageStudioError(
            ErrorCode.NOT_FOUND, f"unknown registration {registration_id}"
        ) from exc
    resident = app_state.coordinator.runtime_status().resident
    if resident is not None and resident.registration_id == registration_id:
        raise ImageStudioError(
            ErrorCode.CONFLICT, "registration is resident on the worker; eject first"
        )
    if app_state.repository.registration_in_active_use(registration_id):
        raise ImageStudioError(ErrorCode.CONFLICT, "registration is referenced by unfinished runs")
