# Architecture and implementation contract

This is the accepted design, now implemented. The backend server, storage,
Hub integration, HTTP API, the real inference runtime, and the four-page
frontend application are implemented and validated by CPU-only tests (fake
Hub, fake worker process, live browser flows against the built frontend);
real generation requires the opt-in inference extra and a GPU, and the
real-GPU acceptance suite is opt-in and not yet executed.

## Product

Use four English-language pages:

- **Generate:** registered model, GPU, Unicode prompt, supported negative
  prompt, size, seed, image count, steps, and applicable guidance. Show the
  global queue, loaded model, Eject, progress, and results. A form change does
  not load a model; submission freezes the requested settings. The results
  gallery lists the selected model's runs newest first and only those that
  already saved a completed image (completed, partial, or active with saved
  images). The Queue panel continues to show pending and running tasks. The
  followed run's detail retains progress, cancellation, and failure feedback
  even when it has no images.
- **Models:** Discover searches Hugging Face and shows model cards, licenses,
  revisions, access restrictions, and compatibility. My Models contains
  application registrations. Local Cache discovers all cached model repos,
  including unsupported and incomplete ones. Adding an existing snapshot only
  registers it; removing a registration never deletes shared weights.
- **History:** image gallery, prompt search, model/favorite filters, run details,
  parameter reuse, downloads, favorites, and recoverable Trash. Retain failed,
  cancelled, and partially completed runs: every record is stored. The default
  Library view hides cancelled runs without completed images; an explicit
  status filter (Cancelled) and the Trash view expose them again, and failed
  runs without images stay visible with their errors.
- **Settings:** effective paths, host/port, Hugging Face login status without
  token values, GPU information, and resident-model status. Configuration
  changes require restart; no live data migration in v1.

V1 is text-to-image only. Exclude FLUX, arbitrary standalone checkpoints, LoRA, masks,
video, CPU offload, ComfyUI integration, account management, automatic updates,
and shared cache deletion. Add image-to-image later as a distinct capability.

## Models and residency

Use explicit built-in adapters: `ZImagePipeline` for Z-Image and
`AnimaModularPipeline` for the two original-checkpoint Anima recipes, and
`QwenImage21Pipeline` for Qwen-Image-2.1 text-to-image:

| Profile | Default repo | Steps | Guidance | Negative prompt |
| --- | --- | --- | --- | --- |
| Z-Image | `Tongyi-MAI/Z-Image` | 50 | 4.0 | Supported |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | 9 | Fixed at 0.0 | Hidden |
| Anima-Turbo | `circlestone-labs/Anima` (Turbo v1.1) | 10 | Fixed at 1.0 | Hidden |
| Anima 2.9B | `Gazingstars123/Anima-2.9B` (Preview v1) | 40 | 4.0 | Supported |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` | 40 | Fixed at 1.0 | Hidden |

Default to bfloat16 and 1024 x 1024. Accept dimensions from 256 to 2048 in
multiples of 16 (32 for Qwen-Image-2.1) and 1-4 sequential images per request. Resolve an empty seed
at submission, use increasing seeds within a run, and record each actual seed.
Use real text-to-image; do not manufacture a blank image for an img2img pipeline.

For compatible derivative repos, inspect pipeline/component declarations and
let the user explicitly choose Base or Turbo. A shared pipeline class does not
identify distillation type. Do not imply that structural compatibility guarantees
output quality. No remote Python code execution or `trust_remote_code`.

Qwen-Image-2.1 declares a Qwen3-VL text encoder and processor, its own transformer
and VAE, and a FlowMatch scheduler. Validate those declarations and the processor's
tokenizer, image/video configuration and chat template in remote listings and local
snapshots. Its profile fixes `true_cfg_scale=1.0` and enables the pipeline's KV
cache. Reject mismatched profiles and dimensions before inference. Preserve the
returned PNG's alpha channel; image editing remains outside the application scope.

Run one API process, one global FIFO generation queue, and at most one resident
inference worker. Every request specifies a GPU UUID. The API process never owns
CUDA model tensors. Downloads use a separate single-task queue.

- Same model, fixed revision, component sources, profile, dtype, and GPU: reuse the worker.
- Different model or GPU: finish current work, stop the previous worker,
  confirm its exit, then start and load the replacement. Never overlap workers.
- After generation: keep the model resident indefinitely; no idle unload.
- Eject while idle: stop the worker and confirm release, leaving files and
  registrations intact. While loading, generating, or switching, Eject is
  disabled and the API returns a conflict. Task cancellation is a separate action.
- Serialize Eject and task dispatch under the same supervisor lock.
- OOM/crash: fail the task and clean up its worker without silently changing
  GPU, precision, or settings. Do not terminate another application's processes.
- Browser closure does not affect tasks or residency. Server shutdown releases
  the worker. On restart, mark active work interrupted, retain finished images,
  and pause queued work until the user resumes it; do not preload a model.

Prevent two server instances from using the same app data directory. Actual
worker state is authoritative for residency; never restore a stale database flag
as if a model were still in VRAM. A healthy worker remains loaded after a normal
generation cancellation. Check cancellation at safe inference boundaries.

Keep Blackwell-specific workarounds out of defaults until a supported runtime
reproduces the failure. Record any explicitly enabled workaround in metadata.

## Hugging Face integration

Use `HfApi` for search and metadata and `scan_cache_dir()` for local discovery.
Downloads resolve the revision to a fixed commit first, then fetch each
selected file with `hf_hub_download()` into the SDK-resolved cache.
Honor existing Hugging Face environment configuration and cache symlinks. Never
pass `local_dir` or modify the user's global cache configuration.

Persist model identity as repo ID, fixed commit and profile, with each Anima
component's repo ID, fixed commit and relative file selection. Do not persist
absolute snapshot paths in SQLite or new generation metadata. Resolve paths
under the current SDK-configured cache when reading registrations, retrying
downloads and reconstructing paused runs. Revalidate all profiles for library
status, submission and queue resume; a missing pinned revision must not fall
back to a different cached revision. The runtime handoff carries resolved local
paths, and inference remains local-only. Migration 003 removes legacy stored
paths while preserving registration IDs, component revisions, history and queue
order. Existing metadata exports are retained and are not used to locate models.

Resolve branches/tags to a commit before downloading. Choose configs, tokenizer
files, safetensors, and shard indexes required by the selected profile; omit
duplicate weight formats and unrelated training artifacts. Validate required
files after download. Cache presence alone does not mean a model is runnable.
Generation uses the verified snapshot with `local_files_only=True`.

Anima recipes select the exact original checkpoint and pinned shared components
from the official CircleStone Diffusers export. A download freezes both repos
before enqueueing and reports combined progress. Registration and each new
submission validate both cached snapshots. Missing shared files make the
registration unavailable; repair reuses the same fixed revisions. Source
manifests persist through restarts and accompany generation metadata. Inference
loads the checkpoint's own conditioner, overrides transformer depth to 28 or 40,
and converts tensor keys in memory using Diffusers' Cosmos mapping. It explicitly
loads local component paths instead of following remote modular component URLs.

Downloads started through Image Studio automatically register the verified snapshot
with their selected profile and fixed commit. Registration and download completion
are committed atomically; repeated downloads reuse the existing library entry
and preserve its display name. External cache entries still require manual
registration, and previously completed jobs are not backfilled.

Show measured download progress, errors, and retry. Permit cancellation of queued
downloads; active pause is out of scope. A retry reuses available cache files.
Reuse `HF_TOKEN` or existing Hub login. Do not save tokens in SQLite or logs.
Model registration survives external cache deletion and becomes missing-files.
Do not update registered revisions automatically.

## Code tree

Existing package directories reserve responsibility boundaries. Named behavior
modules below are the implementation map; they are not empty executable stubs.

```text
image-studio/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .node-version
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── docs/
│   ├── architecture.md
│   └── development.md
├── src/image_studio/
│   ├── __init__.py
│   ├── cli.py, config.py, app.py, runtime.py,
│   │   schemas.py, testing.py
│   ├── api/
│   │   └── models.py, downloads.py, generations.py,
│   │       history.py, system.py
│   ├── hub/
│   │   └── client.py, cache.py, compatibility.py,
│   │       downloads.py
│   ├── inference/
│   │   └── supervisor.py, worker.py, z_image.py,
│   │       profiles.py, gpus.py, protocol.py, fake.py
│   ├── storage/
│   │   ├── database.py, repository.py, artifacts.py
│   │   └── migrations/
│   └── web/static/                                     (generated)
├── frontend/
│   ├── package.json, pnpm-lock.yaml
│   ├── index.html, tsconfig.json, vite.config.ts, vitest.config.ts
│   ├── eslint.config.js, .prettierrc.json, .prettierignore
│   └── src/
│       ├── main.tsx, App.tsx
│       ├── api/, pages/, components/, hooks/, state/, lib/
│       └── styles/
└── tests/
    ├── README.md
    ├── unit/, integration/, gpu/
    └── e2e/                       # Playwright: mocked flows + real-backend
```

Use one Python package, not a uv workspace. The frontend owns rendering and
typed API access; the backend owns compatibility, validation, GPU lifecycle,
downloads, and persistence. Use a small explicit adapter registry for the supported architectures;
there is no dynamic plugin framework.

## Application data tree

These locations are a runtime design. Scaffolding and imports must not create
them. Respect the corresponding XDG variable when set to a valid absolute path;
otherwise use its standard default. `~` is the backend user's home directory.

```text
~/.config/image-studio/                     # XDG_CONFIG_HOME
└── config.toml

~/.local/share/image-studio/                # XDG_DATA_HOME
├── app.sqlite
├── outputs/YYYY-MM-DD/<generation-id>/
│   ├── metadata.json
│   ├── image-001.png
│   └── image-002.png
└── trash/<generation-id>/
    ├── metadata.json
    └── image-001.png

~/.cache/image-studio/                      # XDG_CACHE_HOME
└── thumbnails/YYYY-MM-DD/<generation-id>/
    ├── image-001.webp
    └── image-002.webp

~/.local/state/image-studio/                # XDG_STATE_HOME
└── logs/app.log

~/.cache/huggingface/hub/                   # SDK default; honor HF overrides
└── ...official repositories, snapshots, and blobs...
```

Dates for outputs and thumbnails come from the same run creation timestamp,
even when a thumbnail is generated later. Cache thumbnails lazily. They are
disposable and may be removed when a run enters Trash.

One generation ID identifies one submission, not one image. The run shares its
model, GPU, prompt, and dimensions, and contains 1-4 images with individual seeds
and completion records. Partial success retains completed files and records the
failed/cancelled remainder. Every submission gets a new ID, even with identical
settings. `metadata.json` contains shared run settings and per-image results.

SQLite owns registrations, tasks, favorites, trash status, and artifact indexing.
Store each run's resolved repo/commit/profile, effective parameters, seeds, GPU,
dtype, pipeline, dependency versions, timestamps, status, and enabled compatibility
settings in metadata. Conditions are reproducible; bitwise equality across
hardware or dependency changes is not promised.

Write files through temporary paths and atomic rename, then record success in
SQLite. Reconcile unfinished runs after restart. Serve registered artifact IDs,
not arbitrary client-provided filesystem paths. Keep tokens out of metadata.

Allow an `outputs_dir` override while retaining the local SQLite database.
For an overridden output root, place Trash within that root to permit same-filesystem
moves. Keep artifact paths relative to their configured root. Provide restore,
but no permanent-empty action in v1. Add `inputs/` only when image-to-image exists.

## Planned interfaces

`image-studio serve --host 127.0.0.1 --port 7860` is the server entry point.
Precedence is CLI, then TOML configuration, then defaults. `--fake-runtime`
and `--fake-hub` are explicit development-only flags and are visibly
identified in the runtime and system APIs; the production default composes
the real inference supervisor, and real generation additionally requires the
opt-in inference extra and a GPU. No login/account system: anyone able to reach
an explicitly exposed listener can operate it. Default to loopback and
same-origin frontend/API; do not automatically configure public ingress or a
background service.

| Endpoint family | Purpose |
| --- | --- |
| `/api/hub/models` | Search, inspect, check compatibility |
| `/api/cache/models` | Discover cached model repositories |
| `/api/models` | Register, list, configure, remove library entries |
| `/api/downloads` | Download jobs, progress, retry, queued cancellation |
| `/api/generations` | Submit, list, inspect, cancel, resume paused queue |
| `/api/generations/{id}/artifacts` | Images and metadata |
| `/api/generations/{id}/trash` | Recoverable deletion and restoration |
| `GET /api/runtime` | Resident model, GPU, worker state |
| `POST /api/runtime/eject` | Eject idle worker; conflict when busy |
| `/api/system` | Effective config, GPUs, health |

Generation requests include model registration, GPU UUID, prompt, applicable
negative prompt, dimensions, steps, guidance, seed, and count. Submission returns
a task ID; the active frontend polls status once per second and reduces polling
outside active views. Freeze task settings at submission. Do not remove model
registrations used by resident or unfinished work. Historical GPU unavailability
requires a new explicit selection when reusing settings.

## References

- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Hugging Face downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Hugging Face cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir/latest/)
