# Tests

- `unit/`: configuration, contract schemas, artifact storage, repository
  persistence, Hub cache/download/compatibility logic, and the inference
  profiles, GPU enumeration, adapter kwargs, and supervisor semantics (real
  spawn processes running the CPU-only fake worker). Frontend unit tests
  live with their sources under `frontend/src` (Vitest).
- `integration/`: the public HTTP API and coordinator over the injected fake
  runtime and fake Hub, runtime lifecycle and concurrency with real spawn
  processes, and full app flows against the real supervisor with a fake
  worker process.
- `e2e/`: Playwright browser tests. Default mode drives the Vite dev
  server with a transport-level fake API (generate/models/history/runtime/
  queue/settings/shell flows plus screenshot captures); integration mode
  (`E2E_BASE_URL`, no route interception) drives the built frontend served
  by the real Python API with `--fake-runtime --fake-hub`.
- `gpu/`: explicitly opted-in inference checks with local weights and a
  selected NVIDIA GPU (`pytest.mark.gpu`, deselected by default).

All ordinary tests use isolated temporary data paths and fake providers: no
network access, no weight downloads, no GPU claims, and no writes to real
XDG directories. Backend suites run through the `backend-tests` pre-commit
hook (`uv run --locked pytest -q`); the default pytest selection excludes
`gpu` via `-m 'not gpu'`. GPU tests collect without side effects and execute
only under explicit opt-in.
