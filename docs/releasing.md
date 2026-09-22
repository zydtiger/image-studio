# Manual source releases

Image Studio ships GitHub source releases from `main`, tagged `vX.Y.Z`.
Install from the tagged checkout using uv and the committed lockfile, or build
the Podman image from that checkout. Wheels and sdists are local packaging
validation artifacts, not independently supported release downloads: a plain
pip installation does not apply the repository's uv Git and CUDA source settings.
GitHub Actions validates changes on `ubuntu-latest`; it does not publish releases,
PyPI packages, or container images.

## Version and history

`project.version` in `pyproject.toml` is authoritative. When changing it, run
`uv lock` and include both `pyproject.toml` and `uv.lock` in the release preparation
commit. Keep the HTTP API's version in `src/image_studio/app.py` in sync.
For 0.1.0 these values already match; no dependency update is needed.

Public compatibility surfaces are the documented CLI commands, flags and exit
behavior, HTTP API, configuration keys, and persisted user data. Python modules
are implementation details. Before 1.0, breaking changes to those surfaces bump
the minor version; other changes bump the patch version. Identify breaking
changes and data migration requirements in the notes. Adopting 1.0 requires an
explicit compatibility commitment. From 1.0, use major versions for breaking
changes, minor for compatible features, and patch for compatible fixes; prefer
deprecation and a migration period before removing a public surface.

[GitHub Releases](https://github.com/zydtiger/image-studio/releases) is the sole
version history. Keep each entry brief: summarize user-visible changes and any
breaking changes or material limitations, and link to existing setup documentation.
Do not maintain local release notes or a separate changelog.

## Prepare and validate

1. Start from `main` synchronized with `origin/main`. Keep release preparation
   separate from feature changes; preserve any unrelated local work.
2. Prepare the version metadata and a brief GitHub release entry. Link to setup
   instructions and state material validation limitations. Do not imply GPU verification
   from fake-runtime tests or download models as part of release preparation.
3. Run the local gates from the prepared tree:

   ```sh
   prek run --all-files
   prek run --all-files --stage pre-push
   ```

   Include new files explicitly with `--files` until they are tracked. These
   gates cover backend and frontend checks, browser tests, and `just build`.
   Inspect the resulting wheel and sdist for version metadata and built frontend
   assets. Record which browser tests were skipped; mocked browser tests do not
   establish real-backend or GPU behavior. GPU tests remain separately opt-in.
4. After commit and push authorization, create a focused release preparation
   commit, push it, and verify a clean tree and the same local and remote `main`
   commit. If the prepared content changes, repeat the affected checks.

## Publish after approval

Release approval must identify the exact version and commit. Preparation, commit,
or push approval alone does not authorize a release. One release approval covers
the tag push and the GitHub release publication for that version and commit.

Before creating the first release tag, configure a GitHub tag ruleset for
`refs/tags/v*` with active enforcement and `deletion`, `non_fast_forward`, and
`update` rules, without routine bypass. Verify the ruleset reported by GitHub and
record the configured pattern in `AGENTS.md`. Do not test it with a disposable
tag, because a protected tag cannot subsequently be removed.

Create an annotated tag on the approved commit, push that tag, and publish the
GitHub release with the prepared brief entry and the existing tag. Verify
the remote tag's peeled commit, release body, and installation from the exact
tag in an isolated checkout. Keep verification data isolated from existing user
data; starting the application does not require running GPU inference.

Published release tags are immutable. Never move, replace, force-push, or delete
one; correct a mistaken release with a new version.
