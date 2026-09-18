"""nvidia-smi parsing for GPU enumeration (no CUDA, no torch)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from image_studio.inference.gpus import list_nvidia_gpus, normalize_gpu_uuid


def fake_run(stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_parses_gpu_rows(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: fake_run(
            "0, GPU-9f3a-1111-2222-3333-444444555555, NVIDIA GeForce RTX 5090, 32607\n"
            "1, GPU-77aa-bbbb-cccc-dddd-eeeeeeffffff, NVIDIA GeForce RTX 4090, 24564\n"
        ),
    )
    gpus = list_nvidia_gpus()
    assert [gpu.index for gpu in gpus] == [0, 1]
    assert gpus[0].uuid == "GPU-9f3a-1111-2222-3333-444444555555"
    assert gpus[0].name == "NVIDIA GeForce RTX 5090"
    assert gpus[0].memory_total_bytes == 32607 * 1024 * 1024


def test_missing_nvidia_smi_yields_empty(monkeypatch):
    def raise_missing(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", raise_missing)
    assert list_nvidia_gpus() == []


def test_failing_nvidia_smi_yields_empty(monkeypatch):
    def raise_failed(*a, **kw):
        raise subprocess.CalledProcessError(returncode=9, cmd=["nvidia-smi"])

    monkeypatch.setattr(subprocess, "run", raise_failed)
    assert list_nvidia_gpus() == []


def test_malformed_rows_are_skipped(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: fake_run(
            "not-a-number, , broken, not-a-number\n0, GPU-9f3a-1111-2222, Some GPU, 8192\n"
        ),
    )
    gpus = list_nvidia_gpus()
    assert len(gpus) == 1
    assert gpus[0].uuid == "GPU-9f3a-1111-2222"


def test_normalize_gpu_uuid_variants():
    assert normalize_gpu_uuid("GPU-9F3A-1111") == "9f3a1111"
    assert normalize_gpu_uuid("9f3a-1111") == "9f3a1111"
    assert normalize_gpu_uuid("  GPU-abc123  ") == "abc123"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GPU-9f3a-1111", "9f3a1111"),
        ("9f3a1111", "9f3a1111"),
    ],
)
def test_uuid_normalization_is_stable(raw, expected):
    assert normalize_gpu_uuid(raw) == expected


def test_query_uses_fixed_argv(monkeypatch):
    recorded = SimpleNamespace(argv=None)

    def spy_run(argv, **kwargs):
        recorded.argv = argv
        return fake_run()

    monkeypatch.setattr(subprocess, "run", spy_run)
    list_nvidia_gpus()
    assert recorded.argv[0] == "nvidia-smi"
    assert "--query-gpu=index,uuid,name,memory.total" in recorded.argv
