"""Input upload, frozen edit requests, retention, and restart behavior."""

import io
from datetime import timedelta

import pytest
from PIL import Image

from image_studio import schemas
from image_studio.testing import FakeHub, FakeRuntime
from tests.conftest import build_harness


def png():
    buffer = io.BytesIO()
    Image.new("RGBA", (64, 96), (10, 20, 30, 80)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_real_supervisor_reuses_worker_and_transports_edit_input(xdg_env, tmp_path):
    import json

    from image_studio.inference.protocol import FakeScript
    from tests.integration.test_app_real_runtime import _real_harness

    trace = tmp_path / "trace.jsonl"
    harness = _real_harness(
        xdg_env, FakeHub(tmp_path / "hub"), FakeScript(step_delay=0, trace_path=str(trace))
    )
    with harness.client:
        reg = harness.register_model("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
        image = upload(harness)
        for kwargs in ({}, {"mode": "image-edit", "input_image_ids": [image["input_id"]]}, {}):
            run = harness.submit(reg["id"], steps=1, **kwargs)
            assert harness.wait_terminal(run["run_id"])["status"] == "completed"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len([event for event in events if event["event"] == "load_start"]) == 1
    runs = [event for event in events if event["event"] == "run_start"]
    assert runs[0]["input_paths"] == runs[2]["input_paths"] == []
    assert runs[1]["input_paths"][0].endswith(f"{image['input_id']}.png")


def upload(harness, data=None):
    response = harness.client.post(
        "/api/inputs", files={"file": ("input.png", data or png(), "image/png")}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_edit_retains_input_and_metadata_through_trash(harness):
    registration = harness.register_model("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
    image = upload(harness)
    received = []
    original = harness.runtime.submit

    def record(spec):
        received.append(spec)
        original(spec)

    harness.runtime.submit = record
    run = harness.submit(
        registration["id"], mode="image-edit", input_image_ids=[image["input_id"]], seed=3, count=2
    )
    finished = harness.wait_terminal(run["run_id"])
    assert finished["mode"] == "image-edit"
    assert finished["input_images"] == [image]
    assert received[0].input_image_ids == (image["input_id"],)
    assert received[0].input_paths[0].endswith(f"{image['input_id']}.png")
    metadata = harness.client.get(f"/api/generations/{run['run_id']}/metadata").json()
    assert metadata["mode"] == "image-edit" and metadata["input_images"] == [image]
    for action in ("trash", "restore"):
        assert harness.client.post(f"/api/generations/{run['run_id']}/{action}").status_code == 200
        with Image.open(io.BytesIO(harness.client.get(image["url"]).content)) as saved:
            assert saved.getpixel((0, 0)) == (10, 20, 30, 80)
    assert [item["seed"] for item in finished["images"]] == [3, 4]


@pytest.mark.parametrize(
    "mode,ids,profile,status",
    [
        ("image-edit", [], "qwen-image-2.1", 422),
        ("image-edit", ["missing"], "qwen-image-2.1", 404),
        ("image-edit", ["a", "b"], "qwen-image-2.1", 422),
        ("image-edit", ["a"], "z-image", 422),
        ("text-to-image", ["a"], "qwen-image-2.1", 422),
    ],
)
def test_invalid_edit_rejected_without_run(harness, mode, ids, profile, status):
    repo = "Qwen/Qwen-Image-2.1" if profile == "qwen-image-2.1" else "Tongyi-MAI/Z-Image"
    reg = harness.register_model(repo, profile)
    response = harness.client.post(
        "/api/generations",
        json={
            "registration_id": reg["id"],
            "gpu_uuid": "GPU-fake-0001",
            "prompt": "change background",
            "width": 256,
            "height": 256,
            "mode": mode,
            "input_image_ids": ids,
        },
    )
    assert response.status_code == status, response.text
    assert harness.client.get("/api/generations").json()["total"] == 0


def test_upload_validation_and_orientation(harness):
    for data in (b"not an image", png()[:40]):
        assert (
            harness.client.post("/api/inputs", files={"file": ("x.png", data)}).status_code == 422
        )
    assert (
        harness.client.post(
            "/api/inputs", files={"file": ("x.png", b"x" * (20 * 1024 * 1024 + 1))}
        ).status_code
        == 413
    )
    assert (
        harness.client.post(
            "/api/inputs", files=[("file", ("a", png())), ("file", ("b", png()))]
        ).status_code
        == 422
    )
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (30, 60)).save(buffer, format="JPEG", exif=exif)
    image = upload(harness, buffer.getvalue())
    assert (image["width"], image["height"]) == (60, 30)
    with Image.open(io.BytesIO(harness.client.get(image["url"]).content)) as saved:
        assert not saved.getexif()
    assert harness.client.get("/api/inputs/not-a-file").status_code == 404


def test_expiry_only_removes_unattached_inputs(harness, monkeypatch):
    reg = harness.register_model("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
    unused, used = upload(harness), upload(harness)
    run = harness.submit(reg["id"], mode="image-edit", input_image_ids=[used["input_id"]])
    harness.wait_terminal(run["run_id"])
    later = schemas.utc_now() + timedelta(hours=25)
    monkeypatch.setattr(schemas, "utc_now", lambda: later)
    harness.client.app.state.app_state.inputs.cleanup()
    assert harness.client.get(unused["url"]).status_code == 404
    assert harness.client.get(used["url"]).status_code == 200


def test_animation_and_decoded_pixel_limit(harness, monkeypatch):
    from image_studio.storage import inputs

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (32, 32), "blue")],
        duration=100,
    )
    response = harness.client.post(
        "/api/inputs", files={"file": ("animated.png", buffer.getvalue())}
    )
    assert response.status_code == 422
    monkeypatch.setattr(inputs, "MAX_PIXELS", 100)
    response = harness.client.post("/api/inputs", files={"file": ("large.png", png())})
    assert response.status_code == 422


def test_artifact_copy_survives_source_trash(harness):
    reg = harness.register_model()
    run = harness.submit(reg["id"])
    harness.wait_terminal(run["run_id"])
    response = harness.client.post(
        "/api/inputs/from-artifact", json={"run_id": run["run_id"], "artifact_id": "image-001"}
    )
    assert response.status_code == 201
    image = response.json()
    before = harness.client.get(image["url"]).content
    harness.client.post(f"/api/generations/{run['run_id']}/trash")
    assert harness.client.get(image["url"]).content == before


def test_queued_edit_resumes_with_same_input(xdg_env, tmp_path):
    hub = FakeHub(tmp_path / "hub")
    first = build_harness(xdg_env, FakeRuntime(gate_after_start=True), hub)
    with first.client:
        reg = first.register_model("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
        first.submit(reg["id"])
        image = upload(first)
        queued = first.submit(reg["id"], mode="image-edit", input_image_ids=[image["input_id"]])
    second = build_harness(xdg_env, FakeRuntime(), hub)
    with second.client:
        received = []
        original = second.runtime.submit

        def record(spec):
            received.append(spec)
            original(spec)

        second.runtime.submit = record
        assert second.client.post("/api/generations/queue/resume").status_code == 200
        assert second.wait_terminal(queued["run_id"])["input_images"] == [image]
        assert received[0].mode == "image-edit"
        assert received[0].input_paths[0].endswith(f"{image['input_id']}.png")
