# Image Studio

A local WebUI for Hugging Face model discovery and text-to-image generation
with Diffusers. The supported profiles are Z-Image, Z-Image-Turbo, Anima-Turbo, and Anima 2.9B.

**Status: implemented and integrated (CPU-validated).** The backend server
(`image-studio serve`), SQLite storage with migrations, Hub discovery, cache
inspection, fixed-commit downloads, the generation queue with restart
reconciliation, artifact storage with thumbnails and Trash, the complete
HTTP API, the real inference supervisor/worker/adapter, and the four-page
frontend application (Generate, Models, History, Settings) are implemented
and integrated. CPU-only tests cover the stack end to end using the
injected fake Hub and the inference fake worker process, including a live
browser flow against the built frontend served by the real Python API; no
fake mode claims to be a working inference backend. Real generation
additionally requires the optional inference extra
(`uv sync --locked --extra inference`) and an NVIDIA GPU; the opt-in
real-GPU acceptance suite exists but has not been executed.

## Setup

Requirements: Linux x86_64, Python 3.12, uv, Node as pinned in `.node-version`,
npm, just, and prek. A GPU is not required to develop or validate the backend.
Install prek as a machine-level tool if it is unavailable, for example with
`uv tool install prek==0.4.14`; it is not a project dependency.

```sh
just setup
```

The root `justfile` owns the orchestration commands. `just setup` runs
`uv sync --locked` for the base and development Python dependencies (no
inference extra), installs the frontend and e2e Node dependencies from their
lockfiles, and activates the prek hooks. Build recipes do not re-run setup
or reinstall project dependencies; run `just setup` first.

The optional inference environment is declared separately. It is required
for real generation (the inference subsystem ships with the application;
development and tests run without it):

```sh
uv sync --locked --extra inference
```

That extra uses PyTorch's CUDA 13.0 index. Setup never downloads model weights.

## Running the backend

```sh
just serve --host 127.0.0.1 --port 7860
```

`just serve` starts the server with the inference extra. Host and port
default to `127.0.0.1` and `7860`; `config.toml` under the XDG config
directory can override them, and command-line flags win over both
(CLI > `config.toml` > defaults). Two explicit development flags are
available and visibly identified in `/api/system` and `/api/runtime`:

```sh
uv run --locked image-studio serve --fake-runtime --fake-hub
```

`--fake-runtime` substitutes a deterministic fake inference runtime, and
`--fake-hub` serves a fake Hub catalog from an isolated temporary cache.
Both are for development and tests only; production defaults always use the
real supervisor and the real Hugging Face Hub.

## Frontend

```sh
npm --prefix frontend run dev
```

The Vite development server binds to loopback and proxies `/api` to a
backend on `127.0.0.1:7860`; start the backend first (the fake flags give a
CPU-only development stack). Building writes ignored assets to
`src/image_studio/web/static/`, which the backend serves with SPA routing:

```sh
just build
```

`just build` runs the frontend build and then `uv build`. The frontend build
requires the Node dependencies from a prior `just setup`. A plain `uv build`
does not build the frontend and requires already-built assets under
`src/image_studio/web/static/`.

## Design

- Choose a GPU for each generation request.
- Keep one model resident across the entire application, until explicit Eject,
  a model/GPU change, worker failure, or shutdown. No idle timeout.
- Reuse the official Hugging Face cache; keep app registration metadata in SQLite.
- Keep generated images and per-run metadata outside the repo in XDG data storage.

See [architecture and file trees](docs/architecture.md) for the accepted design,
[the implementation contract](docs/implementation-contract.md) for subsystem
boundaries and HTTP shapes, and [development](docs/development.md) for
validation and implementation order. This is a local-only Git repository with
no remote or release workflow.

## Anima models

Search these original repositories in Models → Discover, select the matching
profile, and download. Once all files pass validation, the model is automatically
added to My Models with the selected profile:

| Profile | Repository and checkpoint | Default settings |
| --- | --- | --- |
| Anima-Turbo | `circlestone-labs/Anima`, `split_files/diffusion_models/anima-turbo-v1.1.safetensors` | 10 steps, fixed CFG 1, no negative prompt |
| Anima 2.9B | `Gazingstars123/Anima-2.9B`, `Anima-2.9B-preview-v1.safetensors` | 40 steps, CFG 4, negative prompt supported |

Both default to bfloat16 and 1024 × 1024. Downloads also fetch the required
shared components from CircleStone's official `Anima-Base-v1.0-Diffusers`
export at commit `073c3a9db359c31ad0e8aa268d15775473c2176c`. The component
sources and fixed revisions remain attached to registrations and history.
The shared export's Base denoiser and text conditioner weights are not downloaded.

Inference uses native Diffusers Anima blocks and Euler flow matching with
shift 3. Checkpoint conversion happens in memory; there is no converted weight
copy, ComfyUI dependency, or remote Python execution. Only the two named
checkpoint recipes are supported, not arbitrary single-file or SD.Next exports.
The models retain the CircleStone Labs Non-Commercial License.

Anima validation covers fake end-to-end flows and, when the inference extra
is installed, tiny synthetic CPU tensor conversion with the real Diffusers
classes. Real-GPU output quality and performance have not been verified.
