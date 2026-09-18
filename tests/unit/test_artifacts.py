"""Unit tests for the artifact store: atomicity, mirroring, trash, confinement."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from image_studio.storage.artifacts import ArtifactError, ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        outputs_dir=tmp_path / "outputs",
        trash_dir=tmp_path / "trash",
        thumbnails_dir=tmp_path / "thumbs",
    )


RUN_ID = "ab" * 16
CREATED = datetime(2026, 9, 16, 12, 30, tzinfo=UTC)


def test_image_written_atomically_with_date_mirror(store: ArtifactStore) -> None:
    size = store.write_image(CREATED, RUN_ID, "image-001", b"png-bytes")
    target = store.outputs_dir / "2026-09-16" / RUN_ID / "image-001.png"
    assert target.read_bytes() == b"png-bytes"
    assert size == len(b"png-bytes")
    assert not list(target.parent.glob(".*.tmp"))


def test_metadata_atomic_rewrite(store: ArtifactStore) -> None:
    store.write_metadata(CREATED, RUN_ID, {"a": 1})
    store.write_metadata(CREATED, RUN_ID, {"a": 2})
    payload = json.loads((store.outputs_dir / "2026-09-16" / RUN_ID / "metadata.json").read_text())
    assert payload == {"a": 2}


def test_trash_and_restore_roundtrip(store: ArtifactStore) -> None:
    store.write_image(CREATED, RUN_ID, "image-001", b"png")
    store.write_metadata(CREATED, RUN_ID, {})
    store.trash_run(CREATED, RUN_ID)
    assert not (store.outputs_dir / "2026-09-16" / RUN_ID).exists()
    assert (store.trash_dir / RUN_ID / "image-001.png").is_file()
    store.restore_run(CREATED, RUN_ID)
    assert (store.outputs_dir / "2026-09-16" / RUN_ID / "image-001.png").is_file()


def test_trash_twice_rejected(store: ArtifactStore) -> None:
    store.write_image(CREATED, RUN_ID, "image-001", b"png")
    store.trash_run(CREATED, RUN_ID)
    with pytest.raises(FileNotFoundError):
        store.trash_run(CREATED, RUN_ID)


def test_thumbnail_lazy_generated_and_webp(store: ArtifactStore) -> None:
    from image_studio.testing import seed_png

    store.write_image(CREATED, RUN_ID, "image-001", seed_png(7, 256, 256))
    thumb_dir = store.thumbnails_dir / "2026-09-16" / RUN_ID
    assert not thumb_dir.exists()
    data = store.thumbnail(CREATED, RUN_ID, "image-001", trashed=False)
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert (thumb_dir / "image-001.webp").is_file()
    assert store.thumbnail(CREATED, RUN_ID, "image-001", trashed=False) == data


def test_trash_drops_thumbnails(store: ArtifactStore) -> None:
    from image_studio.testing import seed_png

    store.write_image(CREATED, RUN_ID, "image-001", seed_png(7, 256, 256))
    store.thumbnail(CREATED, RUN_ID, "image-001", trashed=False)
    store.trash_run(CREATED, RUN_ID)
    assert not (store.thumbnails_dir / "2026-09-16" / RUN_ID).exists()


class TestPathConfinement:
    def test_run_id_must_be_uuid_hex(self, store: ArtifactStore) -> None:
        with pytest.raises(ArtifactError):
            store.write_image(CREATED, "../escape", "image-001", b"x")

    def test_artifact_id_must_match_pattern(self, store: ArtifactStore) -> None:
        with pytest.raises(ArtifactError):
            store.artifact_path(CREATED, RUN_ID, "../../etc/passwd", trashed=False)
        with pytest.raises(ArtifactError):
            store.artifact_path(CREATED, RUN_ID, "image-9999", trashed=False)

    def test_missing_artifact_raises(self, store: ArtifactStore) -> None:
        with pytest.raises(FileNotFoundError):
            store.artifact_path(CREATED, RUN_ID, "image-001", trashed=False)
