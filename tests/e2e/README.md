# End-to-end browser tests

Playwright specs for the Image Studio frontend. The default mode runs the
Vite dev server with a transport-level fake API (route interception) that
implements the exact contract shapes from `src/image_studio/schemas.py`.
Nothing here downloads models, claims GPUs, or talks to a real backend in
default mode.

## Running

Requires the frontend dependencies and this package's dependencies:

```sh
pnpm --dir frontend install --frozen-lockfile
pnpm --dir tests/e2e install --frozen-lockfile
# Install the Chromium revision required by the pinned Playwright package.
pnpm --dir tests/e2e exec playwright install chromium
# Playwright only transpiles; typecheck catches unsupported options.
pnpm --dir tests/e2e run typecheck
pnpm --dir tests/e2e run test
```

`@playwright/test` is pinned in this package's manifest and lockfile. After
upgrading it, rerun the browser installation command to install its matching
Chromium revision. If Playwright reports missing Linux system libraries,
install the reported prerequisites before running browser tests.

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
  pnpm --dir frontend run build
  uv run image-studio serve --fake-runtime --fake-hub --port 7860
  E2E_BASE_URL=http://127.0.0.1:7860 pnpm --dir tests/e2e run test
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

## Automated integration

From the repository root, `just e2e-check` builds the frontend and wheel,
installs the wheel into a temporary environment, and runs integration-mode
browser tests against an isolated fake-provider server. The runner owns server
startup, readiness, and cleanup; it never uses the regular application's data.
The recipe installs frontend and browser-test dependencies; install Chromium
first as described above. Both local pre-push hooks and CI run this through the
`e2e-check` hook, including a fresh build. Failure traces, screenshots, and server
logs are retained under `test-results/`; CI also creates HTML reports under
`playwright-report/`, with separate subdirectories for the two browser modes.
