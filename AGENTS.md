# Repository contract

## Purpose and ownership

Image Studio is a local, private, single-user image generation application.
Read `README.md` for the implemented state, `docs/architecture.md` for the
accepted product and storage design, and `docs/development.md` for tooling.
These documents are repository-owned and must remain self-contained.

The backend, the inference runtime, and the frontend application are
implemented and integrated: server CLI, configuration, SQLite storage with
migrations, Hub discovery and fixed-commit downloads, cache inspection, the
generation queue with restart reconciliation, artifact storage with
thumbnails and recoverable Trash, the HTTP API, the real
supervisor/worker/adapter behind `schemas.Runtime`, and the four-page
React frontend served by the Python server. All of it is validated by
CPU-only tests through the injected fake Hub and the inference fake worker
process, including live browser flows against the built frontend served by
the real API. Real generation additionally requires the opt-in inference
extra and a GPU. GPU tests remain opt-in. Anima-Turbo and Anima 2.9B have
produced real default-parameter 1024 × 1024 images on an RTX 5090; scope
verification claims to the evidence in `docs/development.md`. Fake modes
stay visibly identified as development-only.

## Product boundaries

- Support text-to-image with Z-Image/Z-Image-Turbo through `ZImagePipeline`
  and original Anima-Turbo v1.1/Anima 2.9B Preview v1 through
  `AnimaModularPipeline`. The Anima recipes pin every checkpoint and shared
  component revision; convert original denoiser weights only in memory.
  Generic single-file imports, LoRA, quantization and image-to-image remain
  outside the supported scope.
- Every generation request selects its GPU. Across the application, keep at
  most one model worker and one resident model. Reuse an identical model on
  the same GPU; otherwise finish the current task and fully unload the old
  worker before loading its replacement.
- Keep the model resident after generation. Never add automatic idle ejection.
  Provide explicit Eject; do not interrupt busy work as an implicit ejection.
- Use a global FIFO generation queue. Downloads have a separate queue.
- Use the official Hugging Face cache without `local_dir`, duplicate weight
  storage, automatic model updates, or remote Python code execution.
- Store user data outside the checkout using XDG locations. The thumbnail
  tree mirrors output dates and generation IDs. One generation run can contain
  several images with individually recorded seeds.
- Use English interface text and Unicode-safe prompts. Provide configurable
  host and port; default to loopback, without an account system.

## Source layout

- `src/image_studio/` owns the backend Python package; there is no uv workspace.
- `frontend/` owns React/TypeScript sources. Its build produces ignored assets
  under `src/image_studio/web/static/` for the Python wheel.
- `tests/` separates unit, integration, browser, and opt-in GPU checks.
- Keep model weights, private prompts, generated media, credentials, databases,
  caches, and local configuration out of Git.
- Do not create runtime files under XDG directories during imports, packaging,
  or ordinary tests. Tests use isolated temporary directories.

## Development and validation

Use Python 3.12, uv, just, and the Node version in `.node-version`. Change
Python dependencies through uv and Node dependencies through pnpm; update
their lockfiles together with manifests. Keep `frontend/` and `tests/e2e/`
as separate Node projects with their own `pnpm-lock.yaml`; both pin pnpm
through `packageManager`. Inference dependencies are an opt-in extra.

The root `justfile` owns command orchestration for setup and builds. `just
setup` installs base and development dependencies and activates the hooks;
`just build` builds the frontend and then the Python distribution. Both
`just setup` and `just frontend` depend on `frontend-deps`, which synchronizes
frontend dependencies with a frozen lockfile. `just serve` builds the frontend
before starting the server with the inference extra. Build recipes do not
re-run the full setup, install e2e dependencies or hooks, download weights, or
start services.

`.pre-commit-config.yaml` owns mechanical commands and stage scopes. Activate
hooks with `prek install`. The `package-build` hook runs `just build` at the
pre-push stage. Run `prek run --all-files` and
`prek run --all-files --stage pre-push` for a full local gate. For untracked
scaffold files, pass the explicit file list with `--files`; `--all-files` only
checks tracked files. For documentation-only edits, run the pre-commit stage
on the changed documents. For targeted checks, select the appropriate hook
and changed files rather than repeating all passing gates.

Backend behavioral tests live in `tests/unit/` and `tests/integration/` and
run through the `backend-tests` hook. They use the fake runtime and fake Hub
with isolated temporary data paths: no network, no weight downloads, no GPU.
Frontend unit tests run through the `frontend-tests` hook and browser tests through the `e2e-typecheck` and `e2e-test` hooks.
Keep GPU tests opt-in. Do not download models, run inference, claim GPUs, or
start background services as part of setup or ordinary validation.

## Git and publication

- The base branch is `main`; this repository is local-only and has no remote.
- Scaffolding may be prepared on the unborn `main`. Once initialized with a
  commit, use short-lived task branches for implementation. Use sibling
  worktrees for substantial or concurrent changes, with one writer per worktree.
- Branch names use `<prefix>/<lowercase-hyphenated-topic>` with a prefix from
  the commit categories below. Include an issue number only when one exists.
- Commit subjects use `prefix(scope): concise imperative summary` or
  `prefix: concise imperative summary`, starting the summary with a lowercase
  letter and without a trailing period. Scopes use lowercase letters, digits,
  and hyphens.
- Allowed prefixes: `feat` (behavior), `fix` (correctness), `docs`
  (documentation), `refactor` (structure), `test` (validation), `build`
  (dependencies/packaging), and `chore` (maintenance).
- Editing does not authorize staging, committing, creating further branches,
  merging, creating remotes, pushing, tagging, or releasing. Obtain explicit
  authorization for each Git lifecycle action unless already bundled by the user.
- Preserve unrelated changes, stage exact authorized paths, and inspect the
  staged diff before any approved commit. Merge locally only when authorized.
- No hosted CI, package publication, or releases are planned. The package
  version is development metadata, not a published release commitment.
- Prefer recoverable deletion for user data. Never delete shared model cache
  files as a side effect of removing a library entry.
