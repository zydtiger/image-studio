"""Immutable input images, separate from generated outputs and their Trash."""

from __future__ import annotations

import io
import re
import threading
import uuid
import warnings
from datetime import timedelta
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from image_studio import schemas
from image_studio.storage.repository import Repository

MAX_BYTES = schemas.PROFILES[schemas.ProfileId.QWEN_IMAGE_21].max_input_bytes
MAX_PIXELS = schemas.PROFILES[schemas.ProfileId.QWEN_IMAGE_21].max_input_pixels


class InputStore:
    def __init__(self, root: Path, repository: Repository) -> None:
        self.root = root
        self.repository = repository
        # Submission holds this lock through create_run, so expiry cannot race attachment.
        self.lock = threading.RLock()

    def path(self, input_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", input_id):
            raise schemas.ImageStudioError(schemas.ErrorCode.NOT_FOUND, "unknown input image")
        self.repository.get_input(input_id)
        path = self.root / f"{input_id}.png"
        if not path.is_file():
            raise FileNotFoundError("input image is missing; upload it again")
        return path

    def cleanup(self) -> None:
        with self.lock:
            cutoff = schemas.utc_now() - timedelta(hours=24)
            for input_id in self.repository.expire_inputs(cutoff):
                (self.root / f"{input_id}.png").unlink(missing_ok=True)

    def save(self, data: bytes) -> schemas.InputImage:
        if len(data) > MAX_BYTES:
            raise schemas.ImageStudioError(schemas.ErrorCode.VALIDATION, "image exceeds 20 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("use a PNG, JPEG, or WebP image")
                    if source.width * source.height > MAX_PIXELS:
                        raise ValueError("image exceeds 16 megapixels")
                    if getattr(source, "n_frames", 1) != 1:
                        raise ValueError("animated images are not supported")
                    normalized = ImageOps.exif_transpose(source).convert("RGBA")
                    normalized.info.clear()
                    output = io.BytesIO()
                    normalized.save(output, format="PNG")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise schemas.ImageStudioError(
                schemas.ErrorCode.VALIDATION, f"invalid input image: {exc}"
            ) from exc
        png = output.getvalue()
        input_id = uuid.uuid4().hex
        image = schemas.InputImage(
            input_id=input_id,
            created_at=schemas.utc_now(),
            width=normalized.width,
            height=normalized.height,
            size_bytes=len(png),
            url=f"/api/inputs/{input_id}",
        )
        with self.lock:
            self.cleanup()
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / f"{input_id}.png"
            temporary = self.root / f"{input_id}.tmp"
            try:
                temporary.write_bytes(png)
                temporary.replace(target)
                self.repository.create_input(image)
            except Exception:
                temporary.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise
        return image
