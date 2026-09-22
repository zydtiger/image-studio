# Development

## Setup and scope

Install the prerequisites listed in the root README, then install the hook
runner and set up the development environment:

```sh
uv tool install prek==0.4.14
just setup
```

`just setup` (owned by the root
`justfile`) installs everything: `uv sync --locked` installs the application,
its inference dependencies and development tools, the frontend and
e2e Node dependencies install from their frozen pnpm lockfiles, and `prek install`
activates the hooks. Run `just setup` before development or tests. The
`frontend-deps` recipe synchronizes frontend dependencies before `just frontend`,
`just build`, and `just serve`; those commands do not re-run the full setup or
install e2e dependencies and hooks. `just serve` builds the frontend before
starting the backend. The default environment includes PyTorch, torchvision,
Diffusers, Transformers and Accelerate; no extra is needed. The first setup can
download substantial GPU runtime dependencies, but it does not download model
weights or execute inference. Later `uv sync --locked` calls retain this stack.

Node is pinned in the root `.node-version`. Activate it using your preferred
version manager before `just setup`. Both Node projects pin pnpm 12.4.2 through
`packageManager` in their `package.json`. The separate `frontend/pnpm-lock.yaml`
and `tests/e2e/pnpm-lock.yaml` files are authoritative; this is not a pnpm workspace.
Normal setup uses `pnpm install --frozen-lockfile`, which reuses installed
dependencies and fails when a lockfile is missing or inconsistent with its
manifest. When intentionally changing dependencies, use `pnpm --dir frontend add`
or `pnpm --dir tests/e2e add` with the package name and commit the manifest and
lockfile changes together.
Python targets Linux x86_64 and Python 3.12; other platforms are not yet supported.

The backend server, storage, Hub integration, HTTP API, the real inference
supervisor/worker, and the four-page frontend application are implemented,
integrated, and covered by CPU-only tests (fake Hub, fake worker process and
small CPU tensor checks), including a live browser flow against
the built frontend served by the real API. Real generation additionally
needs model weights and a GPU. No runtime data directories are
created during setup, and no model download or GPU execution is part of
setup.

## Validation

Install prek per machine if needed, then run `prek install` per clone. Hook
commands and scope live only in `.pre-commit-config.yaml`.

For committed/tracked files:

```sh
prek run --all-files
prek run --all-files --stage pre-push
git diff --check
```

Before the initial commit, `--all-files` cannot see untracked files. Validate
them without staging by passing `--files`, for example from zsh or bash:

```sh
git ls-files --others --exclude-standard -z |
  xargs -0 prek run --files
git ls-files --others --exclude-standard -z |
  xargs -0 prek run --stage pre-push --files
```

The commit stage includes structural and whitespace checks, the uv lock check,
Python lint/format, frontend lint/format, and frontend/browser TypeScript checks.
The push stage runs backend and frontend unit tests, mocked browser tests, and
`just e2e-check`. That recipe builds fresh frontend assets and Python
packages, then runs browser integration against the installed wheel. The frontend scripts own Node
commands; hooks invoke them through the package, and the root justfile owns
the build orchestration. Commit subjects are validated at the commit-msg
stage, not by `--all-files`.

Backend behavioral tests run with `uv run --locked pytest` and are wired into
the push stage through the `backend-tests` hook. They inject the fake
runtime and fake Hub from `image_studio.testing` over isolated temporary XDG
paths and never touch the network, download weights, or claim a GPU. Frontend
unit tests use Vitest; browser tests use Playwright. See the
[browser test setup](../tests/e2e/README.md) for installing Chromium and running
the two browser test modes. GPU tests
stay opt-in and must not claim a GPU or download weights during discovery.

## Implementation sequence

1. Implement configuration, SQLite storage, server CLI, and static serving.
2. Implement Hub discovery, profiles, cache inspection, and downloads.
3. Implement the global queue, sole model worker, per-request GPU selection,
   persistent residency, automatic replacement, and Eject.
4. Implement generation, durable artifacts, gallery, parameter reuse, favorites,
   and recoverable deletion.
5. Validate the complete workflow with fake providers, browser checks, and
   explicitly authorized real-GPU checks.

## Acceptance checks to add with behavior

- XDG defaults and overrides, isolated test paths, no import-time filesystem writes.
- Fixed revisions, complete/partial/external cache states, no hidden downloads
  during generation, and no remote model Python execution.
- Same-model reuse; complete unload before model/GPU replacement; one worker
  globally; no idle timeout; Eject serialization and busy-state rejection.
- Cancellation, OOM, crash, restart, paused queue restoration, and second-instance
  exclusion without disturbing unrelated processes.
- Unique run directories, per-image seeds, partial success, metadata consistency,
  safe file serving, favorites, Trash, and restore.
- Browser flow from discovery through download and generation to history,
  including refresh recovery, GPU selection, model switch notices, and Eject.

Real-GPU acceptance verifies Base and Turbo, reuse, replacement, persistence, and
manual Eject. Cross-GPU replacement is tested only when both devices are available
and execution is authorized. Do not interrupt unrelated GPU workloads.

## Anima GPU validation

The runtime dependencies include `torchvision` from the same CUDA index as `torch`.
Cosmos uses its transforms during denoising even for text-to-image. CPU conversion
tests also execute the padding-mask forward path to catch missing dependencies.

On 2026-09-19, both original Anima recipes generated 1024 × 1024 PNGs through
the live API on an NVIDIA GeForce RTX 5090, using bfloat16 and seed 12345:

- Anima-Turbo v1.1: 10 steps, CFG 1, no negative prompt.
- Anima 2.9B Preview v1: 40 steps, CFG 4, with a negative prompt.

The observed outputs were nonblank illustrations. The API runs also exercised
switching between the models, PNG persistence and history records. This is
functional verification, not a quality benchmark or verification of other GPUs.
The stack was torch 2.14.0+cu130, torchvision 0.29.0+cu130, Diffusers 0.40.0,
Transformers 5.17.0 and Accelerate 1.15.0. No model weights were downloaded.

With both recipes already cached, explicitly select an idle GPU and run:

```sh
IMAGE_STUDIO_TEST_GPU=GPU-<uuid> uv run --locked \
  pytest -q -m gpu tests/gpu/test_real_anima.py
```

This opt-in test loads only local snapshots, generates both profiles at their
defaults, checks PNG dimensions and variation, per-step progress, model switching,
residency and Eject. It releases its worker afterward and writes images into the
pytest temporary directory. Ordinary test runs exclude it.

## Qwen-Image-2.1 validation

The inference dependency is pinned in `pyproject.toml` and `uv.lock` to Diffusers
commit `8b3c707ebd3ec4881f4190cf42931da07eaf3b65` (0.41.0.dev0), which includes
`QwenImage21Pipeline`; Diffusers 0.40.0 does not provide that class. Transformers
5.17 or later provides its Qwen3-VL processor and text encoder. Use the locked uv
environment for inference.

CPU tests cover pipeline/component compatibility, processor file selection,
fixed-revision downloads, automatic registration, profile mismatch rejection,
generation defaults and dimensions, offline loader arguments, per-image seeds,
step progress/cancellation and RGBA PNG encoding. Frontend tests cover its fixed
guidance controls and 32-pixel dimension increments. These tests use fake providers
and do not verify real model loading, output quality or GPU memory requirements.
No Qwen weights or GPU inference are required for ordinary validation.

## Contributions and publication

Read the repository's [contribution contract](../AGENTS.md) before making
changes. Use a short-lived task branch, keep changes focused, and run the
checks appropriate to the files you change. Include validation results and
any limitations when submitting a pull request. Documentation-only changes
need the pre-commit stage on the changed documents; a full local gate uses
the two all-files commands above.

Source publication targets a public GitHub repository. Review both the current
tree and the Git history before the first push: removing a file in a later
commit does not remove it from earlier commits. Keep credentials, personal
configuration, prompts, generated media, model weights, and machine-specific
deployment settings outside Git. Git author and committer identities are also
part of the published history.

Repository creation, remote configuration, commits, and pushes require explicit
authorization. Source releases follow the [manual release process](releasing.md).
GitHub Actions runs validation on `ubuntu-latest`. There is no PyPI or
container registry publication.
Building a wheel is local packaging validation, not authorization to publish it.

## GitHub Actions

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
requests. One `ubuntu-latest` job installs dependencies once and runs the full
pre-commit and pre-push stages through SHA-pinned `j178/prek-action`. Python is
3.12; Node/pnpm follow `.node-version` and the frontend's `packageManager` field.
New runs cancel older runs for the same ref. Actions have read-only permissions.

The hook configuration is the single list of checks for local use and CI:

| Stage | Checks |
| --- | --- |
| `pre-commit` | File hygiene, actionlint, lock consistency, Python/frontend lint and format checks, frontend/browser TypeScript checks |
| `commit-msg` | Commit subject convention; runs locally when committing |
| `pre-push` | Backend tests, frontend unit tests, mocked Chromium E2E, fresh frontend/package build and installed-wheel browser integration |

There are no manual-stage hooks or cross-job build artifacts. `just build`
builds distributable packages without starting a server. `just e2e-check`
depends on that build and synchronizes browser-test dependencies before checking
those packages, so it can be run directly without preparing static assets first.
Install Playwright Chromium once as described in the browser test setup.

```sh
just e2e-check
```

The package check installs the wheel into a temporary environment with the locked
runtime dependencies. It verifies the imported package comes from that
installation, checks bundled frontend assets and the CLI, then runs browser tests
against the installed server with fake providers and isolated XDG/Hugging Face
paths. The server uses a temporary loopback port and stops after the check,
including on failure. No model downloads or GPU inference occur.

Browser failures retain traces, screenshots, HTML reports, and the integration
server log under `tests/e2e/`; CI uploads those diagnostics. Successful packages
are CI artifacts, not published releases. Mocked and integration reports use
separate directories. The frontend is built once per push-stage run.

uv and pnpm downloads are cached; Playwright installs its pinned Chromium and
Linux dependencies on the runner. Dependabot checks Actions weekly and Python,
Node, and remote hook dependencies monthly. The action selects prek `0.4.x`;
this version range and the rust-just installer version are maintained manually.
Keep prek at or above the hook configuration's minimum version.
