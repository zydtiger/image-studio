"""Unit tests for huggingface_hub SDK error translation."""

from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from image_studio.hub.client import translate_hub_error
from image_studio.schemas import ErrorCode, ImageStudioError


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x"))


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (GatedRepoError("gated", response=_response(403)), ErrorCode.GATED_MODEL),
        (RepositoryNotFoundError("gone", response=_response(404)), ErrorCode.NOT_FOUND),
        (RevisionNotFoundError("nope", response=_response(404)), ErrorCode.REVISION_NOT_FOUND),
        (HFValidationError("bad id"), ErrorCode.VALIDATION),
        (HfHubHTTPError("401", response=_response(401)), ErrorCode.HUB_AUTH_REQUIRED),
        (HfHubHTTPError("500", response=_response(500)), ErrorCode.HUB_UNREACHABLE),
        (httpx.ConnectError("refused"), ErrorCode.HUB_UNREACHABLE),
        (TimeoutError("slow"), ErrorCode.HUB_UNREACHABLE),
        (RuntimeError("boom"), ErrorCode.HUB_UNREACHABLE),
    ],
)
def test_translate_maps_sdk_errors_to_contract_codes(exc: Exception, code: ErrorCode) -> None:
    translated = translate_hub_error(exc)
    assert isinstance(translated, ImageStudioError)
    assert translated.code is code
    assert translated.message


def test_translate_preserves_cause_and_passes_typed_through() -> None:
    original = GatedRepoError("gated", response=_response(403))
    translated = translate_hub_error(original)
    assert translated.code is ErrorCode.GATED_MODEL
    # callers must raise ... from exc so the cause is inspectable
    with pytest.raises(ImageStudioError) as excinfo:
        raise translated from original
    assert excinfo.value.__cause__ is original

    already = ImageStudioError(ErrorCode.CONFLICT, "already typed")
    assert translate_hub_error(already) is already


def test_translate_never_includes_response_headers() -> None:
    original = HfHubHTTPError("boom", response=_response(500))
    translated = translate_hub_error(original)
    assert "authorization" not in translated.message.lower()
