# End-to-end browser tests

Playwright specs for the Image Studio frontend. The default mode runs the
Vite dev server with a transport-level fake API (route interception) that
implements the exact contract shapes from `src/image_studio/schemas.py`.
Nothing here downloads models, claims GPUs, or talks to a real backend in
default mode.

## Running

Requires the frontend dependencies and this package's dependencies:

```sh
npm --prefix frontend ci
npm --prefix tests/e2e ci
npm --prefix tests/e2e typecheck   # Playwright only transpiles; tsc catches
                                   # unsupported options (e.g. route.fulfill's
                                   # removed isBase64) at type level
npm --prefix tests/e2e test
```

`@playwright/test` stays pinned at 1.62.0 because that release matches the
Chromium build in the shared machine cache (`chromium-1234`); do not
upgrade it without checking `~/.cache/ms-playwright`.

## Two modes

- **Transport-mocked (default):** `fixtures/fakeApi.ts` serves every
  contract endpoint through Playwright route interception, with a
  pathname-prefix matcher (a naive `**/api/**` glob also swallows Vite
  module URLs like `/src/api/profiles.ts`). Tests receive a `fakeApi`
  control object from `fixtures/app.ts` to seed registrations, runs, and
  downloads, and to inject progress, failures, cancellation, and a
  simulated restart. Polling the app performs at ~1 Hz drives the fake's
  state machine.
- **Integration (real backend):** set `E2E_BASE_URL` to a server running
  the built frontend with `image-studio serve --fake-runtime
--fake-hub`:

  ```sh
  npm --prefix frontend run build
  uv run image-studio serve --fake-runtime --fake-hub --port 7860
  E2E_BASE_URL=http://127.0.0.1:7860 npm --prefix tests/e2e test
  ```

  Injection-dependent tests skip themselves; `real-backend.spec.ts` runs
  structural smoke checks against the served API. Transport-mocked
  browser tests alone are not final end-to-end proof; integration mode
  is the final gate for the composed stack.

## Layout

- `fixtures/fakeApi.ts` — the contract-shaped fake backend and controls.
- `fixtures/app.ts` — the `fakeApi` fixture (null in integration mode).
- `shell.spec.ts`, `generate-flow.spec.ts`, `runtime.spec.ts`
  (switch notice, Eject, cancellation, failures),
  `queue-restart.spec.ts` (pause/resume), `models-flow.spec.ts`
  (discover/download/cache/registrations), `history-flow.spec.ts`
  (filters/favorites/trash/reuse), `settings.spec.ts`.
- `screenshots.spec.ts` writes representative desktop/narrow captures to
  `test-results/` (gitignored).
