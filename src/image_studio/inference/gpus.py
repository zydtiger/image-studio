"""NVIDIA GPU enumeration without initializing CUDA or claiming devices.

The API process must never import torch (the worker child exclusively owns
CUDA), so ``list_gpus`` shells out to ``nvidia-smi``, which reads the driver
without creating CUDA contexts. An absent or failing ``nvidia-smi`` yields
an empty list; the backend then rejects submissions whose GPU UUID does not
match an enumerated entry.
"""

from __future__ import annotations

import subprocess

from image_studio.schemas import GpuInfo

_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total",
    "--format=csv,noheader,nounits",
]
_TIMEOUT_SECONDS = 10.0


def normalize_gpu_uuid(uuid: str) -> str:
    """Normalize GPU UUID spellings for reliable comparison.

    ``nvidia-smi`` reports ``GPU-xxxxxxxx-xxxx-...`` while ``torch`` exposes
    a bare ``uuid.UUID``; both reduce to the same lowercase hex digits.
    """

    return uuid.strip().lower().removeprefix("gpu-").replace("-", "")


def list_nvidia_gpus() -> list[GpuInfo]:
    """Enumerate NVIDIA GPUs; empty when the tool or driver is unavailable."""

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            _QUERY,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: list[GpuInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[1]:
            continue
        try:
            index = int(parts[0])
            memory_mib = int(parts[3])
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                uuid=parts[1],
                name=parts[2],
                index=index,
                memory_total_bytes=memory_mib * 1024 * 1024,
            )
        )
    return gpus
