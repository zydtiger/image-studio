"""Dispatch the supported multi-source recipes without importing inference code."""

from image_studio.hub import anima, qwen_fp8


def _recipe(repo_id):
    return qwen_fp8 if repo_id == qwen_fp8.REPO else anima


def make_sources(cache_dir, repo_id, commit, profile):
    return _recipe(repo_id).make_sources(cache_dir, repo_id, commit, profile)


def resolve_sources(hub, repo_id, revision, profile):
    return _recipe(repo_id).resolve_sources(hub, repo_id, revision, profile)


def model_problems(root, repo_id, profile, sources=()):
    return _recipe(repo_id).model_problems(root, repo_id, profile, sources)
