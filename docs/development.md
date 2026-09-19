# Development

## Setup and scope

Use the setup commands in the root README. `just setup` (owned by the root
`justfile`) installs everything: `uv sync --locked` installs the core package
and development tools without the optional inference stack, the frontend and
e2e Node dependencies install from their frozen pnpm lockfiles, and `prek install`
activates the hooks. Run `just setup` before development or tests. The
`frontend-deps` recipe synchronizes frontend dependencies before `just frontend`,
`just build`, and `just serve`; those commands do not re-run the full setup or
install e2e dependencies and hooks. `just serve` builds the frontend before
starting the backend. Use `uv sync --locked --extra inference` when GPU
implementation is required. Subsequent uv commands should include that extra
if they need to retain it.

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
integrated, and covered by CPU-only tests (fake Hub, fake worker process;
no torch in the test environment), including a live browser flow against
the built frontend served by the real API. Real generation additionally
needs the opt-in inference extra and a GPU. No runtime data directories are
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
Python lint/format, and frontend lint/format. The push stage builds the Python
distribution through `just build`, which type-checks and builds the frontend
first and includes the generated web assets. The frontend scripts own Node
commands; hooks invoke them through the package, and the root justfile owns
the build orchestration. Commit subjects are validated at the commit-msg
stage, not by `--all-files`.

Backend behavioral tests run with `uv run --locked pytest` and are wired into
the commit stage through the `backend-tests` hook. They inject the fake
runtime and fake Hub from `image_studio.testing` over isolated temporary XDG
paths and never touch the network, download weights, or claim a GPU. Add
Vitest/Playwright and their gates when frontend behavior exists. GPU tests
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

The inference extra includes `torchvision` from the same CUDA index as `torch`.
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
IMAGE_STUDIO_TEST_GPU=GPU-<uuid> uv run --locked --extra inference \
  pytest -q -m gpu tests/gpu/test_real_anima.py
```

This opt-in test loads only local snapshots, generates both profiles at their
defaults, checks PNG dimensions and variation, per-step progress, model switching,
residency and Eject. It releases its worker afterward and writes images into the
pytest temporary directory. Ordinary test runs exclude it.

## Publication

Keep Git local. There is no remote, hosted CI, or release workflow. Building a
wheel is local packaging validation, not authorization to publish it.
