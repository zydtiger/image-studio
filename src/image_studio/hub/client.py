"""Hugging Face clients: search, model detail, login state, file listings.

All network access happens behind an injected ``HfApi``-shaped object so
tests can substitute recorded fakes. The Hub integration never executes
remote code and never passes ``local_dir``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from image_studio import schemas
from image_studio.schemas import (
    ErrorCode,
    HubFileEntry,
    HubModelSummary,
    HubRevision,
    ImageStudioError,
)

DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 100

type FileLister = Callable[[str, str | None], tuple[str | None, list[tuple[str, int | None]]]]
type Downloader = Callable[[str, str, str], str]


def translate_hub_error(exc: Exception) -> ImageStudioError:
    """Map a huggingface_hub SDK failure onto a typed contract error.

    Order matters: specific subclasses are checked before ``HfHubHTTPError``.
    The caller preserves the original exception as ``__cause__``; messages
    come from the SDK text only and never include credentials or headers.
    """
    import httpx
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalTokenNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, ImageStudioError):
        return exc
    message = str(exc).strip()
    if isinstance(exc, GatedRepoError):
        return ImageStudioError(ErrorCode.GATED_MODEL, message or "gated repository access denied")
    if isinstance(exc, LocalTokenNotFoundError):
        return ImageStudioError(ErrorCode.HUB_AUTH_REQUIRED, message or "Hub login required")
    if isinstance(exc, RepositoryNotFoundError):
        return ImageStudioError(ErrorCode.NOT_FOUND, message or "repository not found")
    if isinstance(exc, RevisionNotFoundError):
        return ImageStudioError(ErrorCode.REVISION_NOT_FOUND, message or "revision not found")
    if isinstance(exc, HFValidationError):
        return ImageStudioError(ErrorCode.VALIDATION, message or "invalid repository identifier")
    if isinstance(exc, HfHubHTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        text = message or f"Hub request failed with HTTP {status_code}"
        if status_code == 401:
            return ImageStudioError(ErrorCode.HUB_AUTH_REQUIRED, text)
        if status_code == 403:
            return ImageStudioError(ErrorCode.GATED_MODEL, text)
        if status_code == 404:
            return ImageStudioError(ErrorCode.NOT_FOUND, text)
        return ImageStudioError(ErrorCode.HUB_UNREACHABLE, text)
    if isinstance(exc, FileNotFoundError):
        return ImageStudioError(ErrorCode.NOT_FOUND, message or "not found")
    if isinstance(exc, (httpx.HTTPError, ConnectionError, TimeoutError)):
        return ImageStudioError(
            ErrorCode.HUB_UNREACHABLE, message or "cannot reach the Hugging Face Hub"
        )
    return ImageStudioError(ErrorCode.HUB_UNREACHABLE, f"{type(exc).__name__}: {exc}")


def _as_bool(value: Any) -> bool:
    return bool(value) if not isinstance(value, str) else True


class HubStack:
    """Everything the backend needs from the Hub: API plus download hooks."""

    def __init__(
        self,
        api: Any,
        *,
        file_lister: FileLister,
        downloader: Downloader,
        cache_dir: Path,
    ) -> None:
        self.api = api
        self.clients = HubClients(api)
        self.file_lister = file_lister
        self.downloader = downloader
        self.cache_dir = cache_dir

    @classmethod
    def real(cls, cache_dir: Path) -> HubStack:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()

        def real_lister(
            repo_id: str, revision: str | None
        ) -> tuple[str | None, list[tuple[str, int | None]]]:
            info = api.model_info(repo_id, revision=revision, files_metadata=True)
            files = [(s.rfilename, getattr(s, "size", None)) for s in info.siblings]
            return getattr(info, "sha", None), files

        def real_downloader(repo_id: str, filename: str, revision: str) -> str:
            return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

        return cls(api, file_lister=real_lister, downloader=real_downloader, cache_dir=cache_dir)


class HubClients:
    """Read-only Hub queries used by the Models and Settings pages."""

    def __init__(self, api: Any) -> None:
        self._api = api

    def search_models(self, query: str, limit: int) -> list[HubModelSummary]:
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        try:
            # The SDK's list_models is a lazy generator: auth and network
            # failures surface during iteration, so consume it fully inside
            # the error boundary before any translation escapes.
            results = list(self._api.list_models(search=query or None, limit=limit))
        except Exception as exc:
            raise translate_hub_error(exc) from exc
        return [_model_to_summary(item) for item in results]

    def model_detail(self, repo_id: str) -> schemas.HubModelDetail:
        try:
            info = self._api.model_info(repo_id, files_metadata=True)
            refs = self._api.list_repo_refs(repo_id)
        except Exception as exc:
            raise translate_hub_error(exc) from exc
        siblings = getattr(info, "siblings", None) or []
        # Installed-SDK shape (GitRefs/GitRefInfo): fields are
        # name/ref/target_commit; there is no sha and no default. The ref
        # name is what Hub APIs accept as a revision; target_commit is the
        # fixed commit that name resolves to.
        revisions = [
            HubRevision(revision=ref.name, commit_sha=ref.target_commit)
            for group in ("branches", "tags", "converts")
            for ref in getattr(refs, group, None) or []
        ]
        # Honest default-branch derivation: model_info reports the default
        # branch's commit. Return a branch name only when exactly one
        # branch targets that commit; when several branches share it,
        # picking any single one as "the" default would be unjustified.
        default_revision = None
        default_sha = getattr(info, "sha", None)
        if default_sha is not None:
            matching = [
                ref.name
                for ref in getattr(refs, "branches", None) or []
                if getattr(ref, "target_commit", None) == default_sha
            ]
            if len(matching) == 1:
                default_revision = matching[0]
        return schemas.HubModelDetail(
            repo_id=info.id,
            author=getattr(info, "author", None),
            private=_as_bool(getattr(info, "private", False)),
            gated=_as_bool(getattr(info, "gated", False)),
            downloads=getattr(info, "downloads", None),
            likes=getattr(info, "likes", None),
            last_modified=getattr(info, "lastModified", None),
            pipeline_tag=getattr(info, "pipeline_tag", None),
            license=_card_license(info),
            default_revision=default_revision,
            revisions=revisions,
            files=[
                HubFileEntry(path=sibling.rfilename, size=getattr(sibling, "size", None))
                for sibling in siblings
            ],
        )

    def whoami(self) -> tuple[bool, str | None]:
        try:
            result = self._api.whoami()
        except Exception:
            return False, None
        if isinstance(result, dict) and result.get("name"):
            return True, str(result["name"])
        name = getattr(result, "name", None)
        return (True, str(name)) if name else (False, None)


def _model_to_summary(item: Any) -> HubModelSummary:
    return HubModelSummary(
        repo_id=item.id,
        author=getattr(item, "author", None),
        private=_as_bool(getattr(item, "private", False)),
        gated=_as_bool(getattr(item, "gated", False)),
        downloads=getattr(item, "downloads", None),
        likes=getattr(item, "likes", None),
        last_modified=getattr(item, "lastModified", None),
        pipeline_tag=getattr(item, "pipeline_tag", None),
        license=_card_license(item),
    )


def _card_license(item: Any) -> str | None:
    card = getattr(item, "card_data", None)
    if card is None:
        return None
    if isinstance(card, dict):
        value = card.get("license")
        return str(value) if value else None
    value = getattr(card, "license", None)
    return str(value) if value else None
