"""Bounded uploads and immutable input-image reads."""

from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser

from image_studio import schemas
from image_studio.api import state
from image_studio.storage.inputs import MAX_BYTES

router = APIRouter()


class UploadTooLarge(MultiPartException):
    pass


class ArtifactInput(BaseModel):
    run_id: str
    artifact_id: str


@router.post("/api/inputs/from-artifact", status_code=201)
def input_from_artifact(request: Request, body: ArtifactInput) -> schemas.InputImage:
    app = state(request)
    row = app.repository.get_run(body.run_id)
    images = app.repository.get_images(body.run_id)
    if not any(
        image["artifact_id"] == body.artifact_id and image["status"] == "completed"
        for image in images
    ):
        raise HTTPException(404, "completed image not found")
    path = app.artifacts.artifact_path(
        datetime.fromisoformat(row["created_at"]),
        body.run_id,
        body.artifact_id,
        trashed=row["trashed_at"] is not None,
    )
    with path.open("rb") as source:
        return app.inputs.save(source.read(MAX_BYTES + 1))


@router.post(
    "/api/inputs",
    status_code=201,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def upload_input(request: Request) -> schemas.InputImage:
    # Bound the whole multipart stream before the parser spools files to disk.
    total = 0

    async def bounded_stream():
        nonlocal total
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BYTES + 64 * 1024:
                raise UploadTooLarge("image exceeds 20 MiB")
            yield chunk

    try:
        form = await MultiPartParser(
            request.headers, bounded_stream(), max_files=1, max_fields=0
        ).parse()
    except UploadTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except MultiPartException as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(422, "upload one image in the file field")
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "image exceeds 20 MiB")
        return await run_in_threadpool(state(request).inputs.save, data)
    finally:
        await form.close()


@router.get("/api/inputs/{input_id}")
def get_input(request: Request, input_id: str) -> FileResponse:
    return FileResponse(state(request).inputs.path(input_id), media_type="image/png")
