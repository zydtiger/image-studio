# Implementation contract

Binding interface contract between the three implementation areas: the
backend subsystem (configuration, schemas, storage, hub, HTTP, CLI, and
backend tests), the inference subsystem (`src/image_studio/inference/` and
its tests), and the frontend (`frontend/` and browser tests). The backend
subsystem owns integration: it composes the areas together and owns the
schemas that define their boundaries. Changes to this file or to
`src/image_studio/schemas.py` require backend review and explicit user
approval before any area may rely on the change.

`src/image_studio/schemas.py` is the executable form of this contract. Where
prose and code disagree, the code is authoritative for shapes and the prose
for sequencing and side effects.

## 1. Ownership and import rules

| Area | Owner |
| --- | --- |
| `schemas.py`, `config.py`, `cli.py`, `app.py`, `runtime.py`, `testing.py` | Backend |
| `src/image_studio/api/`, `hub/`, `storage/`, `storage/migrations/` | Backend |
| `tests/unit/`, `tests/integration/` (backend behavior) | Backend |
| `src/image_studio/inference/` (entire package, including internal fake worker-process test helpers) | Inference |
| `tests/gpu/` | Inference |
| `frontend/`, `tests/e2e/` | Frontend |

Import and side-effect rules:

- `schemas.py` imports stdlib and pydantic only. No torch, diffusers,
  FastAPI, huggingface_hub, filesystem access, or network access, ever.
- The inference package may import `schemas` (and its own stack). It must
  never import `storage`, `api`, `app`, `config`, FastAPI, or sqlite3, and
  must never write to application data locations. All persistence flows
  through `EventSink` callbacks implemented by the backend.
- The backend never imports `inference` modules except through the
  `schemas.Runtime` protocol and one composition point in `app.py` that
  constructs the real runtime. Fake runtimes used by tests and development
  live in the backend-owned `testing.py` module, never inside `inference/`.
- No root global configuration and no writes outside the application's XDG
  locations (plus the optional `outputs_dir` override). The repository
  checkout is never written at runtime.
- `app.py` composes: settings, storage, hub clients, the runtime, and the
  API. No other module wires these together.

## 2. Shared semantics: ids, timestamps, normalization, seeds

The backend owns identifiers, creation timestamps, profile normalization,
and seed resolution, in that order, before persisting anything and before
`Runtime.submit`. The runtime performs no product-rule validation and no
seed arithmetic.

- `run_id` is a UUID4 hex string assigned at submission. Every submission
  gets a new id, even with identical settings. Download and registration ids
  are UUID4 hex strings.
- Artifact ids are `image-001` through `image-00N` (1-based, three digits),
  derived from the frozen image count.
- All timestamps are timezone-aware UTC, serialized as ISO 8601.
- Normalization pipeline for `POST /api/generations`:
  1. Resolve the registration; it must have status `ready` with a verified
     snapshot path.
  2. Resolve the GPU: `gpu_uuid` must match an entry in
     `Runtime.list_gpus()`; otherwise 422 `validation`. `count` defaults to
     one image when omitted.
  3. Apply profile defaults from `schemas.PROFILES` and validate with
     `schemas.validate_generation`. Unsupported inputs for a fixed profile
     (Turbo guidance other than 0.0, any Turbo negative prompt) are
     rejected with 422 and never silently ignored or dropped.
  4. Resolve seeds with `schemas.resolve_seeds`. Within a run, seeds
     increase by one per image and never wrap: an explicit seed whose run
     would pass `MAX_SEED` is rejected with reason `seed_overflow`;
     randomly drawn first seeds are sampled from a range that makes
     wrap-around impossible. The actual seed of every image is recorded.
  5. Freeze a `schemas.FrozenRunSpec`. Effective parameters (steps,
     guidance, dimensions, prompt texts, dtype) are explicit in the spec;
     the runtime recomputes nothing. The seed and artifact-id sequences are
     immutable tuples, so the frozen handoff cannot be mutated in place.
  6. Persist the run and its per-image rows as `queued`, then call
     `Runtime.submit(spec)`. A submission failure after persistence leaves
     the run recorded as failed, never silently dropped.
- `FrozenRunSpec.model.snapshot_path` is the absolute, verified local
  snapshot directory inside the Hugging Face cache. Generation loads with
  `local_files_only=True` from that path. The runtime never re-resolves
  repo or revision against the Hub, eliminating repo+revision lookup
  ambiguity when a cache holds multiple revisions.
- Profile metadata (`schemas.PROFILES`) is the single truth for validation
  bounds, defaults, field visibility, and UI capability metadata. The
  inference subsystem's `profiles.py` consumes these constants; it does not
  redefine them.

## 3. Runtime protocol

Implemented by the inference subsystem against `schemas.Runtime`; consumed
by the backend API layer. Method semantics:

- `attach(sink)` — register the single `EventSink`, once, at startup before
  any `submit`. Submitting without an attached sink is an internal error.
- `list_gpus()` — enumerate NVIDIA GPUs (stable hardware UUIDs) without
  initializing CUDA or claiming devices. The backend validates submission
  GPU choices against this list; the frontend uses it for selection.
- `status()` — cheap, consistent snapshot (`schemas.RuntimeStatus`) for
  1-second polling. `implementation` is `"real"` or `"fake"` so a stub can
  never masquerade as a complete inference stack.
- `submit(spec)` — append to the tail of the global FIFO generation queue.
  Non-blocking; never loads a model inline. Accepts any queued run
  regardless of current worker state; replacement scheduling is internal.
- `cancel(run_id)` — returns `removed_from_queue` (queued run removed
  immediately), `cancelling` (running run, cooperative cancel requested at
  the next safe inference boundary), or `already_finished`. A healthy
  worker remains resident after a cancelled run.
- `eject()` — permitted while idle, and an idempotent no-op (204) while
  unloaded. Stops the worker, confirms process exit and release, and leaves
  files and registrations untouched. Raises
  `schemas.RuntimeConflictError` (HTTP 409) while loading, generating,
  switching, or ejecting. Eject and task dispatch are serialized under the
  same supervisor lock.
- `shutdown()` — stop any worker and confirm exit; used on server shutdown.

Worker invariants (normative):

- Globally at most one worker process and one resident model, ever. The
  invariant includes GPU changes: the same model on a different GPU is a
  full worker replacement, not a mutation.
- Worker reuse requires identical repo id, commit, component source manifests,
  profile, dtype, and GPU.
  Otherwise: finish the current task, fully stop the previous worker,
  confirm its exit, then start and load the replacement. Workers never
  overlap in time or memory.
- The model stays resident indefinitely after generation. No idle timeout,
  no automatic unload, under any condition. Liveness observation is not
  unloading: a worker that exits while idle (external kill, OOM) is reaped,
  residency clears to `unloaded` with a `worker_error` `last_error`, and
  the next run spawns a fresh worker. A live idle worker is never ejected
  by this path, and no completed run is retried.
- On OOM or worker crash during a run: fail the active task (`run_failed`
  with `worker_error`), clean up the worker's own process only, never
  change GPU, precision, or settings silently, never terminate unrelated
  processes, and transition to `unloaded` with `last_error` recorded.
- On server restart the runtime starts `unloaded` and preloads nothing.
  Actual worker state is authoritative; a database flag never implies
  residency.

Worker state machine (`schemas.WorkerState`): `unloaded` is the explicit
no-worker state; `unloaded -> loading -> idle <-> generating`; replacement
passes through `switching -> loading`; explicit eject passes through
`ejecting -> unloaded`; crash or OOM from any state lands in `unloaded`
with `last_error` set.

## 4. Event sink contract

The runtime reports through the single `EventSink.on_event` callback with
`schemas.RuntimeEvent` payloads (discriminated union). Events are Python
objects; PNG bytes travel in-process in `ImageCompleted.png`. How the
supervisor bridges events and bytes out of the worker process is an
inference-internal implementation detail.

Ordering and threading:

- Events are delivered from a single dispatch thread, strictly serialized:
  the next event is delivered only after the previous handler returns. This
  gives the backend natural backpressure and guarantees that durable
  writes for one image complete before further inference proceeds.
- Per run: `run_started`, then any interleaving of `run_progress` and
  `image_completed`, then exactly one terminal event (`run_completed`,
  `run_failed`, or `run_cancelled`). `worker_state_changed` may interleave;
  a `generating` state always precedes `run_started` for the dispatched run.
  A queued run cancelled before dispatch receives only `run_cancelled`
  (no `run_started`); sinks must treat that as a complete terminal.
- `run_started` carries the serving worker's reported runtime identity
  (`pipeline_class`, `dependency_versions`), so every run — including runs
  on a reused worker and runs that later fail — records the stack that
  produced it. `ResidentModel` carries the same fields for the resident
  worker. Both are optional with empty defaults for compatibility.
- The sink may be read concurrently from API threads; the backend
  implementation must be thread-safe. Events themselves arrive only from
  the dispatch thread.

Exception propagation (binding): a storage failure must fail the task,
never report success. The backend coordinator wraps every event's durable
work (artifact write, metadata rewrite, database update) so that a failure
persists `run_failed` for that run or marks it failed during startup
reconciliation. If a sink handler raises despite this, the supervisor
treats the exception as run failure: stop remaining images, emit
`run_failed`, never emit `run_completed` afterwards. If delivering
`run_failed` itself raises, the supervisor records the failure in
`status.last_error` and keeps serving; startup reconciliation closes any
run left non-terminal.

Worker-state notifications split into two classes. The `generating`
transition that immediately precedes `run_started`, and `run_started`
itself, are run-scoped: a sink raise there fails that run with a storage
error before dispatch, preserves the healthy resident worker, and keeps
the queue serving. All other worker-state notifications — `loading`,
`switching`, `idle`, `unloaded`, `ejecting` — are observational: a sink
raise never mutates or fails any run; it is recorded in
`status.last_error` and serving continues (a subsequent healthy load
clears `last_error` as with worker faults).

## 5. Artifacts and persistence boundary

The backend is the sole writer of application data. The runtime never
touches the data tree.

- `storage/artifacts.py` owns `outputs/YYYY-MM-DD/<run_id>/` with
  `image-NNN.png` and `metadata.json`, written through temporary paths and
  atomic rename, then recorded in SQLite (file first, database second).
  `metadata.json` carries shared run settings and per-image results and is
  atomically rewritten after each `image_completed`, so partial success and
  crashes retain consistent on-disk records. Directory dates come from the
  run creation timestamp.
- `GET /api/generations/{run_id}/metadata` serves this exact file. No ZIP
  archives in v1.
- Thumbnails are WebP, generated lazily on first request under
  `thumbnails/YYYY-MM-DD/<run_id>/`, mirroring output dates and run ids.
  They are disposable and are dropped when a run enters Trash.
- Trash is a same-filesystem move of the run directory to
  `trash/<run_id>/`. Restore reverses it. v1 provides restore only: no
  permanent-empty and no per-item permanent deletion.
- Registered artifact ids are the only servable paths. Clients never
  supply filesystem paths.
- Tokens are never persisted in SQLite, metadata, or logs.

## 6. HTTP API surface

Every non-2xx response uses `schemas.ErrorBody`:
`{"error": {"code", "message", "details?"}}`. Status mapping: 400/422
validation (`validation`, `cache_incomplete`), 403 (`gated_model`,
`hub_auth_required`), 404 (`not_found`, `revision_not_found`), 409 (`conflict`), 502
(`hub_unreachable`), 500 (`internal`). Success responses use the named
schema models. Unknown `/api/*` paths return the JSON envelope 404; other
unknown paths fall back to the SPA.

| Method | Path | Request | Success | Notes |
| --- | --- | --- | --- | --- |
| GET | `/api/profiles` | - | `ProfilesResponse` | Capability metadata: defaults, bounds, fixed/hidden fields per profile |
| GET | `/api/hub/models` | `?q=&limit=` | `{results: [HubModelSummary]}` | Hub search |
| GET | `/api/hub/models/{repo_id}` | - | `HubModelDetail` | Card, license, revisions, gated, files |
| GET | `/api/hub/models/{repo_id}/compatibility` | `?revision=` | `CompatibilityReport` | Structural check; profile is an explicit user choice |
| GET | `/api/cache/models` | - | `{repos: [CachedRepo]}` | All cached repos incl. unsupported/incomplete |
| POST | `/api/models` | `RegistrationCreate` | 201 `ModelRegistration` | Registers an existing snapshot only; never downloads; missing files -> 422 `cache_incomplete` |
| GET | `/api/models` | - | `{registrations: [...]}` | |
| PATCH | `/api/models/{id}` | `RegistrationUpdate` | 200 `ModelRegistration` | Profile change 409 while resident or referenced by unfinished runs |
| DELETE | `/api/models/{id}` | - | 204 | 409 while resident or referenced by unfinished runs; never deletes files |
| POST | `/api/downloads` | `DownloadCreate` | 202 `DownloadJob` | Separate single-task queue; revision resolved to commit before transfer; validated completion atomically registers the selected profile |
| GET | `/api/downloads` | - | `{jobs: [...]}` | |
| POST | `/api/downloads/{id}/retry` | - | 202 `DownloadJob` | Reuses cached files |
| POST | `/api/downloads/{id}/cancel` | - | 202 `DownloadJob` | Queued only; running -> 409 |
| POST | `/api/generations` | `GenerationRequest` | 202 `RunDetail` | Normalization per §2; returns the frozen run. 409 while the queue is paused after a restart: resume it first |
| GET | `/api/generations` | `?q=&status=&model=&favorite=&has_images=&exclude_empty_cancelled=&trashed=&limit=&offset=` | `{runs: [RunSummary], total}` | Prompt search, filters, pagination (`limit` 1-200, `offset` >= 0, `trashed` `exclude`/`only`, boolean filters per the semantics below; invalid values 422); `trashed=only` for the Trash view |
| GET | `/api/generations/queue` | - | `QueueState` | `paused` flag and pending order after restart |
| POST | `/api/generations/queue/resume` | - | `QueueState` | Resumes paused queue in original FIFO order |
| GET | `/api/generations/{run_id}` | - | `RunDetail` | 1 Hz polling while active; includes progress snapshot and queue position |
| PATCH | `/api/generations/{run_id}` | `FavoriteUpdate` | 200 `RunSummary` | Favorite toggle |
| POST | `/api/generations/{run_id}/cancel` | - | 202 `RunDetail` | Queued -> cancelled now; running -> `cancelling` |
| GET | `/api/generations/{run_id}/artifacts` | - | `{artifacts: [ArtifactView]}` | Per-image status with explicit URLs |
| GET | `/api/generations/{run_id}/artifacts/{artifact_id}` | `?download=1` | `image/png` | Registered ids only; `download=1` sets attachment disposition |
| GET | `/api/generations/{run_id}/artifacts/{artifact_id}/thumbnail` | - | `image/webp` | Lazily generated |
| GET | `/api/generations/{run_id}/metadata` | - | `application/json` | Serves the run's `metadata.json` as a download |
| POST | `/api/generations/{run_id}/trash` | - | 200 `RunSummary` | Same-filesystem move; retains records |
| POST | `/api/generations/{run_id}/restore` | - | 200 `RunSummary` | |
| GET | `/api/runtime` | - | `RuntimeStatus` | Resident model including its GPU, pipeline class, and dependency versions; worker state, current run, queue depth |
| POST | `/api/runtime/eject` | - | 204 | Idle only; 409 otherwise |
| GET | `/api/system` | - | `SystemInfo` | Effective paths, host/port, HF login state (no token values), GPUs, development flags |

Run-listing filter semantics:

- `has_images` is tri-state: omitted applies no image filter, `true`
  returns only runs with at least one completed image, and `false` only
  runs with none. A completed image is an `images` row with
  `status = 'completed'`, independent of run status, so active runs with
  saved images and partial runs both count.
- `exclude_empty_cancelled=true` hides cancelled runs without completed
  images. It defaults to `false` — omitting it changes nothing — and an
  explicit `status` filter always wins, so `status=cancelled` exposes
  the empty cancellations regardless.
- All filters compose into one WHERE clause applied before `COUNT` and
  `LIMIT`/`OFFSET`: `total` and the paginated page always describe the
  filtered set.

Frontend consumption rules:

- Default GPU selection: the resident worker's GPU when one is loaded,
  otherwise the first entry of the GPU list. Reusing settings from history
  with a GPU uuid no longer present requires an explicit new selection
  before submission.
- Only form submission affects residency. Editing a form never loads a
  model; submitted settings are frozen.
- The active Generate view polls run detail and runtime status at 1 Hz and
  reduces polling outside active views.
- Configuration changes require a server restart; `/api/system` reports
  effective values only.

## 7. Data records (outline)

SQLite owns registrations, downloads, runs, images, favorites, and trash
state; schema versioning via `PRAGMA user_version` with SQL files under
`storage/migrations/`.

- `registrations(id, repo_id, commit_sha, profile, display_name, status,
  missing_files, snapshot_path, created_at, last_used_at)` — status flips to
  `missing_files` when the cache snapshot disappears externally; removing a
  registration never deletes shared cache files.
- `downloads(id, repo_id, requested_revision, resolved_commit, profile,
  status, error, progress counters, created_at, started_at, finished_at)`.
- `runs(run_id, queue_seq, created_at, started_at, finished_at, status,
  favorite, trashed_at, registration_id, repo_id, commit_sha, profile,
  gpu_uuid, gpu_name, prompt, negative_prompt, width, height, steps,
  guidance, initial_seed, image_count, dtype, pipeline_class,
  dependency_versions, runtime_meta, error)` — records resolved model,
  effective parameters, environment, and any explicitly enabled
  compatibility workaround.
- `images(run_id, artifact_id, idx, seed, status, width, height, size_bytes,
  error, completed_at)`.

Startup reconciliation: runs `running` become `interrupted` (finished
images retained); queued runs become `paused` pending explicit queue
resume; interrupted downloads become failed and retryable. A single-instance
lock on the data directory prevents two servers from sharing it.

## 8. Fakes and injection points

- `create_app(settings, *, runtime=None, hub=None)` constructs the server;
  omitted arguments select the production implementations (real runtime,
  real Hub clients). Production defaults never use fakes.
- `image-studio serve --fake-runtime --fake-hub` are explicit
  development/testing flags only. When active, they are visibly identified
  in `RuntimeStatus.implementation = "fake"` and
  `SystemInfo.development`. A fake runtime never masquerades as a complete
  implementation, and no standalone fake application is a deliverable.
- The backend-owned `testing.py` provides `FakeRuntime` (implements
  `schemas.Runtime`; two deterministic fake GPUs; configurable delays,
  failure injection, and seed-derived placeholder PNGs using Pillow) and a
  fake Hub catalog. The inference subsystem keeps any internal
  worker-process test helpers inside its own area; no fake is shared
  across areas.

## 9. Test responsibilities

- Backend: unit tests for configuration, XDG resolution, normalization and
  seed rules, compatibility verdicts, artifact naming/atomicity/metadata,
  Trash/restore, thumbnails, and the download state machine; integration
  tests driving the full API with `FakeRuntime` and fake Hub over ASGI,
  including cancellation, partial success, queue pause/resume across app
  recreation, eject conflicts, registration guards, single-instance
  refusal, and static serving.
- Inference: worker lifecycle and supervisor semantics with its own fakes;
  opt-in `tests/gpu/` with local weights and explicit authorization.
- Frontend: component/unit tests under `frontend/src` (Vitest + Testing
  Library) and browser tests under `tests/e2e` (Playwright). The default
  e2e mode runs mocked transport flows against the Vite dev server;
  integration mode (`E2E_BASE_URL`) runs `real-backend.spec.ts` against the
  built frontend served by the real Python API with
  `--fake-runtime --fake-hub`, covering onboarding
  (Discover → download → automatic My Models registration), explicit GPU selection
  with a Unicode prompt, two same-page runs, decoded media with exact
  metadata (status/seeds/GPU), browser download events, History
  favorite/Trash/restore, and Eject.

Ordinary tests use isolated temporary XDG directories, perform no network
access, download no weights, and claim no GPUs.

## Original Anima checkpoint recipes

`ProfileId` additionally accepts `anima-turbo` and `anima-2.9b`; profile defaults
and field visibility remain owned by `schemas.PROFILES`. Existing Z-Image
request bodies and registrations remain valid.

Downloads, registrations, frozen models and run details carry `sources`: an
immutable list of `ModelSource` records (`repo_id`, `commit_sha`, `files`,
`snapshot_path`). It is server-owned; clients cannot choose arbitrary files or
component paths. Anima downloads resolve the original checkpoint revision and
use the pinned official shared-component revision before queueing. Retrying
keeps this manifest, and registration requires every selected file locally.
Z-Image retains its existing single-snapshot loading and an empty source list.
Migration 002 adds JSON source columns to downloads, registrations and runs,
with an empty-list default preserving existing records. Metadata exports and
restart reconstruction retain all recorded sources.

The Anima adapter uses native Diffusers modular blocks, original checkpoint
weights and explicit local component loaders. It never follows model-card
Python or remote component loading instructions. Depth is 28 for Turbo and 40
for 2.9B. CFG is set through the modular guider for every image; the native
step loop reports progress and propagates cooperative cancellation. Successful
cancellation retains the healthy model. CPU tests use placeholder Hub files
and synthetic tensors; they are not evidence of verified GPU image quality.
