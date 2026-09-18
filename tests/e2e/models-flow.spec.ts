import { expect, test } from "./fixtures/app";

/**
 * Model management flows: Hub discovery with compatibility and profile
 * choice, downloads with retry, cache registration, and registration
 * guards. The fake Hub catalog is defined in fixtures/fakeApi.ts.
 */

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  await page.goto("/#/models");
});

test("discovers a model, checks compatibility, and downloads it", async ({
  page,
}) => {
  await page.getByLabel("Search Hugging Face").fill("z-image-derived");
  await page.getByRole("button", { name: "Search" }).click();

  await page.getByRole("button", { name: "Details" }).click();
  await expect(page.getByText("Structurally compatible")).toBeVisible();
  await expect(
    page.getByText(/Pipeline class declares ZImagePipeline/),
  ).toBeVisible();

  await page.getByRole("button", { name: "Download…" }).click();

  // The download lands on the Downloads tab and progresses to completion.
  await expect(
    page.getByRole("tabpanel").getByText("someone/z-image-derived"),
  ).toBeVisible({ timeout: 5_000 });
  await expect(page.getByText("Downloading").first()).toBeVisible();
  await expect(page.getByText("Completed").first()).toBeVisible({
    timeout: 15_000,
  });
  await expect(
    page.getByText(/Register the snapshot from Local Cache/i),
  ).toBeVisible();
});

test("shows incompatible repositories without download actions", async ({
  page,
}) => {
  await page.getByLabel("Search Hugging Face").fill("unrelated");
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByRole("button", { name: "Details" }).click();

  await expect(page.getByText("Not compatible")).toBeVisible();
  await expect(page.getByRole("button", { name: "Download…" })).toBeDisabled();
});

test("registers a cached snapshot from Local Cache", async ({
  page,
  fakeApi,
}) => {
  test.skip(fakeApi === null, "transport fixtures only");
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });

  await page.getByRole("tab", { name: "Local Cache" }).click();
  await expect(page.getByText("Tongyi-MAI/Z-Image").first()).toBeVisible();

  // The orphan snapshot of an unregistered repo offers registration.
  await page.getByRole("button", { name: "Register" }).first().click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Register" })
    .click();

  await expect(page.getByText(/registered/i).first()).toBeVisible({
    timeout: 5_000,
  });
});

test("marks missing-file registrations and keeps cache files on removal", async ({
  page,
  fakeApi,
}) => {
  test.skip(fakeApi === null, "transport fixtures only");
  fakeApi!.addRegistration({
    repo_id: "broken/model",
    status: "missing_files",
  });

  await page.getByRole("tab", { name: "My Models" }).click();
  await expect(page.getByText("broken/model")).toBeVisible();
  await expect(page.getByText("Missing files").first()).toBeVisible();
  await expect(
    page.getByText(/Missing from the cache: model_index\.json/i),
  ).toBeVisible();

  await page
    .getByRole("listitem")
    .filter({ hasText: "broken/model" })
    .getByRole("button", { name: "Remove" })
    .click();
  await expect(
    page.getByText(/Files in the shared Hugging Face cache are never deleted/i),
  ).toBeVisible();
  await page.getByRole("button", { name: "Remove registration" }).click();

  await expect(page.getByText("No models registered")).toBeVisible({
    timeout: 5_000,
  });
});

test("refuses to remove a resident registration with a clear conflict", async ({
  page,
  fakeApi,
}) => {
  test.skip(fakeApi === null, "transport fixtures only");
  const registration = fakeApi!.addRegistration({
    repo_id: "resident/model",
  });
  fakeApi!.setResidentIdle(registration.id);

  await page.getByRole("tab", { name: "My Models" }).click();
  await page
    .getByRole("listitem")
    .filter({ hasText: "resident/model" })
    .getByRole("button", { name: "Remove" })
    .click();
  await page.getByRole("button", { name: "Remove registration" }).click();

  await expect(page.getByText(/currently resident/i).first()).toBeVisible({
    timeout: 5_000,
  });
  await expect(page.getByText("resident/model", { exact: true })).toBeVisible();
});

test("refuses to remove a registration referenced by unfinished runs", async ({
  page,
  fakeApi,
}) => {
  test.skip(fakeApi === null, "transport fixtures only");
  const registration = fakeApi!.addRegistration({
    repo_id: "queued/model",
  });
  fakeApi!.setAutoStart(false);
  fakeApi!.seedRun({
    prompt: "queued run holding the registration",
    status: "queued",
    registrationId: registration.id,
  });

  await page.getByRole("tab", { name: "My Models" }).click();
  await page
    .getByRole("listitem")
    .filter({ hasText: "queued/model" })
    .getByRole("button", { name: "Remove" })
    .click();
  await page.getByRole("button", { name: "Remove registration" }).click();

  await expect(
    page.getByText(/referenced by unfinished runs/i).first(),
  ).toBeVisible({ timeout: 5_000 });
  await expect(page.getByText("queued/model", { exact: true })).toBeVisible();
  expect(
    fakeApi!
      .state()
      .registrations.some((entry) => entry.repo_id === "queued/model"),
  ).toBe(true);
});

test("retries a failed download", async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  await page.getByRole("tab", { name: "Discover" }).click();
  await page.getByLabel("Search Hugging Face").fill("Z-Image");
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByRole("button", { name: "Details" }).first().click();

  const [response] = await Promise.all([
    page.waitForResponse((res) => res.url().includes("/api/downloads")),
    page.getByRole("button", { name: "Download…" }).click(),
  ]);
  expect(response.ok()).toBeTruthy();
  const jobId = (await response.json()).id as string;

  await page.getByRole("tab", { name: "Downloads" }).click();
  await expect(page.getByText("Downloading").first()).toBeVisible();
  fakeApi!.failDownload(jobId, "Connection reset.");
  await expect(page.getByText("Connection reset.")).toBeVisible({
    timeout: 10_000,
  });

  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("Downloading").first()).toBeVisible({
    timeout: 5_000,
  });
});
