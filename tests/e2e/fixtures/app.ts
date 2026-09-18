import { test as base, expect } from "@playwright/test";

import { installFakeApi, type FakeApi } from "./fakeApi";

/**
 * Shared fixtures. In the default transport-mocked mode route interception
 * is installed automatically for every test, so specs that never request
 * `fakeApi` still get a working /api transport instead of requests proxied
 * to an absent backend. The `fakeApi` control object remains available for
 * tests that steer the fake. When E2E_BASE_URL points at a real backend
 * (`image-studio serve --fake-runtime --fake-hub`), `fakeApi` is null,
 * nothing is intercepted, and injection-dependent tests skip themselves so
 * the same specs can serve as integration evidence.
 */
export const test = base.extend<{ fakeApi: FakeApi | null }>({
  fakeApi: [
    async ({ context }, use) => {
      if (process.env.E2E_BASE_URL !== undefined) {
        await use(null);
        return;
      }
      await use(await installFakeApi(context));
    },
    { auto: true },
  ],
});

export { expect };
