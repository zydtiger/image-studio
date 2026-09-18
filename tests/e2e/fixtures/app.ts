import { test as base, expect } from "@playwright/test";

import { installFakeApi, type FakeApi } from "./fakeApi";

/**
 * Shared fixtures. In the default transport-mocked mode every test gets a
 * `fakeApi` control object driving route interception. When E2E_BASE_URL
 * points at a real backend (`image-studio serve --fake-runtime
 * --fake-hub`), `fakeApi` is null and injection-dependent tests skip
 * themselves so the same specs can serve as integration evidence.
 */
export const test = base.extend<{ fakeApi: FakeApi | null }>({
  fakeApi: async ({ context }, use) => {
    if (process.env.E2E_BASE_URL !== undefined) {
      await use(null);
      return;
    }
    const api = await installFakeApi(context);
    await use(api);
  },
});

export { expect };
