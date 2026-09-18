"""HTTP endpoints for the model library, generation, history, and runtime."""

from __future__ import annotations

from fastapi import Request

from image_studio.app import AppState


def state(request: Request) -> AppState:
    return request.app.state.app_state
