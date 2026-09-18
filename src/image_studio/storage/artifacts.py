"""Durable artifact storage: outputs, metadata, thumbnails, and trash.

The artifact store is the only writer of the application data tree. Files
are written through temporary paths and atomic rename. Paths are always
re-derived from validated identifiers, never from client input.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ARTIFACT_ID_PATTERN = re.compile(r"image-\d{3}")
_THUMBNAIL_SIZE = (512, 512)


class ArtifactError(Exception):
    """Raised for confinement violations and missing artifacts."""


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ArtifactError(f"invalid run id {run_id!r}")


def _validate_artifact_id(artifact: str) -> None:
    if not _ARTIFACT_ID_PATTERN.fullmatch(artifact):
        raise ArtifactError(f"invalid artifact id {artifact!r}")


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)


def _confined(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ArtifactError(f"path escapes artifact root: {candidate}")
    return resolved


class ArtifactStore:
    def __init__(self, outputs_dir: Path, trash_dir: Path, thumbnails_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self.trash_dir = trash_dir
        self.thumbnails_dir = thumbnails_dir

    # ----- path derivation ---------------------------------------------------

    def _date_component(self, created_at: datetime) -> str:
        return created_at.date().isoformat()

    def output_dir(self, created_at: datetime, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self.outputs_dir / self._date_component(created_at) / run_id

    def trash_target(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self.trash_dir / run_id

    def _thumbnail_dir(self, created_at: datetime, run_id: str) -> Path:
        return self.thumbnails_dir / self._date_component(created_at) / run_id

    # ----- writes ---------------------------------------------------------------

    def write_image(self, created_at: datetime, run_id: str, artifact: str, png: bytes) -> int:
        _validate_artifact_id(artifact)
        target = self.output_dir(created_at, run_id) / f"{artifact}.png"
        _atomic_write(target, png)
        return len(png)

    def write_metadata(self, created_at: datetime, run_id: str, payload: dict[str, Any]) -> None:
        target = self.output_dir(created_at, run_id) / "metadata.json"
        _atomic_write(target, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))

    # ----- reads ----------------------------------------------------------------

    def artifact_path(
        self, created_at: datetime, run_id: str, artifact: str, *, trashed: bool
    ) -> Path:
        _validate_artifact_id(artifact)
        if trashed:
            base = _confined(self.trash_dir, self.trash_target(run_id))
        else:
            base = _confined(self.outputs_dir, self.output_dir(created_at, run_id))
        candidate = base / f"{artifact}.png"
        if not candidate.is_file():
            raise FileNotFoundError(f"artifact {run_id}/{artifact} not found")
        return candidate

    def metadata_path(self, created_at: datetime, run_id: str, *, trashed: bool) -> Path:
        if trashed:
            base = _confined(self.trash_dir, self.trash_target(run_id))
        else:
            base = _confined(self.outputs_dir, self.output_dir(created_at, run_id))
        candidate = base / "metadata.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"metadata for run {run_id} not found")
        return candidate

    def thumbnail(
        self, created_at: datetime, run_id: str, artifact: str, *, trashed: bool
    ) -> bytes:
        """Serve the WebP thumbnail, generating it lazily on first request."""
        from PIL import Image

        _validate_artifact_id(artifact)
        target = self._thumbnail_dir(created_at, run_id) / f"{artifact}.webp"
        if target.is_file():
            return target.read_bytes()
        source = self.artifact_path(created_at, run_id, artifact, trashed=trashed)
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail(_THUMBNAIL_SIZE)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=80)
        data = buffer.getvalue()
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, data)
        return data

    # ----- trash lifecycle ---------------------------------------------------

    def trash_run(self, created_at: datetime, run_id: str) -> None:
        source = _confined(self.outputs_dir, self.output_dir(created_at, run_id))
        if not source.is_dir():
            raise FileNotFoundError(f"run directory for {run_id} not found")
        target = _confined(self.trash_dir, self.trash_target(run_id))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ArtifactError(f"trash entry for {run_id} already exists")
        os.rename(source, target)
        self.drop_thumbnails(created_at, run_id)

    def restore_run(self, created_at: datetime, run_id: str) -> None:
        source = _confined(self.trash_dir, self.trash_target(run_id))
        if not source.is_dir():
            raise FileNotFoundError(f"trash entry for {run_id} not found")
        target = _confined(self.outputs_dir, self.output_dir(created_at, run_id))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ArtifactError(f"output directory for {run_id} already exists")
        os.rename(source, target)

    def drop_thumbnails(self, created_at: datetime, run_id: str) -> None:
        directory = self._thumbnail_dir(created_at, run_id)
        if directory.is_dir():
            shutil.rmtree(directory)
