import { expect, test } from "./fixtures/app";

/**
 * Model-scoped result browsing on the Generate page: the selected model's
 * recent runs stay browsable after a reload, switching models swaps the
 * list, older runs open in the detail view, and finished submissions
 * appear without a reload.
 */

test.beforeEach(async ({ fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });
  fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    profile: "z-image-turbo",
  });
});

test("keeps a model's recent results browsable across reloads", async ({
  page,
  fakeApi,
}) => {
  const base = fakeApi!
    .state()
    .registrations.find((entry) => entry.repo_id === "Tongyi-MAI/Z-Image")!;
  fakeApi!.seedRun({
    prompt: "older harbor",
    registrationId: base.id,
    completed: 2,
  });
  const newest = fakeApi!.seedRun({
    prompt: "newest harbor",
    registrationId: base.id,
    completed: 2,
  });

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.getByTitle("older harbor")).toBeVisible({
    timeout: 10_000,
  });
  await expect(results.getByTitle("newest harbor")).toBeVisible();

  // The newest run of the selected model is followed by default.
  await expect(
    results.getByText(`run ${newest.run_id.slice(0, 8)}`).first(),
  ).toBeVisible();
  await expect(
    results.getByText(`seed ${newest.initial_seed}`).first(),
  ).toBeVisible();

  // List thumbnails are real decodable media.
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const images = Array.from(
            document.querySelectorAll<HTMLImageElement>(
              ".model-run__preview img",
            ),
          );
          return {
            count: images.length,
            allDecoded: images.every(
              (img) => img.complete && img.naturalWidth > 0,
            ),
          };
        }),
      { message: "run list thumbnails must decode" },
    )
    .toEqual({ count: 2, allDecoded: true });

  // A reload restores the selected model and the same default view.
  await page.reload();
  await expect(page.getByLabel("Model", { exact: true })).toHaveValue(base.id, {
    timeout: 10_000,
  });
  await expect(results.getByTitle("newest harbor")).toBeVisible({
    timeout: 10_000,
  });
  await expect(results.getByTitle("older harbor")).toBeVisible();
  await expect(
    results.getByText(`seed ${newest.initial_seed}`).first(),
  ).toBeVisible();
});

test("switching models swaps the run list and resets the selection", async ({
  page,
  fakeApi,
}) => {
  const registrations = fakeApi!.state().registrations;
  const base = registrations.find(
    (entry) => entry.repo_id === "Tongyi-MAI/Z-Image",
  )!;
  const turbo = registrations.find(
    (entry) => entry.repo_id === "Tongyi-MAI/Z-Image-Turbo",
  )!;
  const baseRun = fakeApi!.seedRun({
    prompt: "base model run",
    registrationId: base.id,
  });
  const turboRun = fakeApi!.seedRun({
    prompt: "turbo model run",
    registrationId: turbo.id,
  });

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.getByTitle("base model run")).toBeVisible({
    timeout: 10_000,
  });
  await expect(
    results.getByText(`seed ${baseRun.initial_seed}`).first(),
  ).toBeVisible();

  await page.getByLabel("Model", { exact: true }).selectOption(turbo.id);
  await expect(results.getByTitle("turbo model run")).toBeVisible();
  await expect(results.getByTitle("base model run")).toBeHidden();
  await expect(
    results.getByText(`seed ${turboRun.initial_seed}`).first(),
  ).toBeVisible();

  // Switching back restores the base model's list and newest run.
  await page.getByLabel("Model", { exact: true }).selectOption(base.id);
  await expect(results.getByTitle("base model run")).toBeVisible();
  await expect(
    results.getByText(`seed ${baseRun.initial_seed}`).first(),
  ).toBeVisible();
});

test("opening an older run shows its images in the detail view", async ({
  page,
  fakeApi,
}) => {
  const base = fakeApi!
    .state()
    .registrations.find((entry) => entry.repo_id === "Tongyi-MAI/Z-Image")!;
  const older = fakeApi!.seedRun({
    prompt: "older run detail",
    registrationId: base.id,
    completed: 2,
  });
  fakeApi!.seedRun({ prompt: "newest run detail", registrationId: base.id });

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.getByTitle("newest run detail")).toBeVisible({
    timeout: 10_000,
  });

  await results.getByTitle("older run detail").click();
  const chip = results.getByTitle("older run detail");
  await expect(chip).toHaveAttribute("aria-pressed", "true");
  await expect(
    results.getByText(`run ${older.run_id.slice(0, 8)}`).first(),
  ).toBeVisible();
  await expect(
    results.getByText(`seed ${older.initial_seed}`).first(),
  ).toBeVisible();
});

test("finished submissions join the list and Clear keeps the history", async ({
  page,
  fakeApi,
}) => {
  const base = fakeApi!
    .state()
    .registrations.find((entry) => entry.repo_id === "Tongyi-MAI/Z-Image")!;
  fakeApi!.seedRun({ prompt: "earlier run", registrationId: base.id });

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.getByTitle("earlier run")).toBeVisible({
    timeout: 10_000,
  });

  await page.getByLabel("Prompt", { exact: true }).fill("fresh session run");
  await page.getByLabel("Seed", { exact: true }).fill("500");
  await page.getByRole("button", { name: "Generate" }).click();

  // The submitted run completes and appears in the list without a reload.
  await expect(results.getByText("Completed").first()).toBeVisible({
    timeout: 20_000,
  });
  const freshChip = results.getByTitle("fresh session run");
  await expect(freshChip).toBeVisible({ timeout: 10_000 });
  await expect(freshChip.getByText("Completed")).toBeVisible();

  // Clearing the detail keeps the browsable history and does not reselect.
  await results.getByRole("button", { name: "Clear" }).click();
  await expect(results.getByText("No run selected")).toBeVisible();
  await expect(results.getByTitle("fresh session run")).toBeVisible();
  await expect(results.getByTitle("earlier run")).toBeVisible();

  // Picking a listed run restores its detail.
  await results.getByTitle("fresh session run").click();
  await expect(results.getByText("seed 500").first()).toBeVisible({
    timeout: 10_000,
  });
});

test("loads older runs with bounded page requests beyond the API ceiling", async ({
  page,
  fakeApi,
}) => {
  const base = fakeApi!
    .state()
    .registrations.find((entry) => entry.repo_id === "Tongyi-MAI/Z-Image")!;
  // More records than the API's maximum limit: paging must request by
  // offset in page-size slices, never by a growing limit that would be
  // rejected with 422.
  const total = 205;
  for (let index = 0; index < total; index += 1) {
    fakeApi!.seedRun({
      prompt: `history run ${index}`,
      registrationId: base.id,
    });
  }

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.getByTitle("history run 204")).toBeVisible({
    timeout: 10_000,
  });
  await expect(results.locator(".model-run")).toHaveCount(8);

  for (let click = 0; click < 25; click += 1) {
    const response = page.waitForResponse(
      (request) =>
        request.request().method() === "GET" &&
        new URL(request.url()).pathname === "/api/generations",
    );
    await results.getByRole("button", { name: /load more/i }).click();
    expect((await response).status()).toBe(200);
    await expect(results.locator(".model-run")).toHaveCount(
      Math.min(8 * (click + 2), total),
    );
  }

  // Every archived record is reachable and no pagination remains.
  await expect(results.getByTitle("history run 0")).toBeVisible();
  await expect(
    results.getByRole("button", { name: /load more/i }),
  ).toBeHidden();
  expect(await results.locator(".model-run").count()).toBe(total);
});

test("a completion between runtime polls updates the run list", async ({
  page,
  fakeApi,
}) => {
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });
  fakeApi!.setAutoStart(false);
  // A queued run on an already-loaded deeper page is followed, then the
  // fake dispatches and finishes it: the refresh must re-read the loaded
  // depth, not only the first page, so the deep chip updates too.
  const base = fakeApi!
    .state()
    .registrations.find((entry) => entry.repo_id === "Tongyi-MAI/Z-Image")!;
  fakeApi!.seedRun({
    registrationId: base.id,
    prompt: "old queued run",
    status: "queued",
  });
  for (let index = 0; index < 8; index += 1) {
    fakeApi!.seedRun({
      registrationId: base.id,
      prompt: `later queued ${index}`,
      status: "queued",
    });
  }

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.locator(".model-run")).toHaveCount(8, {
    timeout: 10_000,
  });
  await results.getByRole("button", { name: /load more/i }).click();
  await expect(results.locator(".model-run")).toHaveCount(9, {
    timeout: 10_000,
  });
  const deepChip = results.getByTitle("old queued run", { exact: true });
  await deepChip.click();
  await expect(deepChip).toHaveAttribute("aria-pressed", "true");

  fakeApi!.startNextRun();
  fakeApi!.finishCurrentRun();

  await expect(
    results.getByText("Completed", { exact: true }).first(),
  ).toBeVisible({ timeout: 10_000 });
  await expect(deepChip).toContainText("Completed", { timeout: 5_000 });
});

test("cancelling an unfollowed run updates its older-page chip", async ({
  page,
  fakeApi,
}) => {
  const base = fakeApi!
    .state()
    .registrations.find((entry) => entry.repo_id === "Tongyi-MAI/Z-Image")!;
  fakeApi!.setAutoStart(false);
  fakeApi!.seedRun({
    registrationId: base.id,
    prompt: "old cancellation",
    status: "queued",
  });
  for (let index = 0; index < 8; index += 1) {
    fakeApi!.seedRun({
      registrationId: base.id,
      prompt: `newer pending ${index}`,
      status: "queued",
    });
  }

  await page.goto("/#/generate");
  const results = page.getByLabel("Run results");
  await expect(results.locator(".model-run")).toHaveCount(8, {
    timeout: 10_000,
  });
  await results.getByRole("button", { name: /load more/i }).click();
  const deepChip = results.getByTitle("old cancellation", { exact: true });
  await expect(deepChip).toContainText("Queued");

  // The run is never followed; cancelling it from the queue must still
  // refresh the deep loaded page.
  await page
    .getByLabel("Generation queue")
    .getByRole("listitem")
    .filter({ hasText: "old cancellation" })
    .getByRole("button", { name: "Cancel", exact: true })
    .click();
  await expect(deepChip).toContainText("Cancelled", { timeout: 5_000 });
});

test("a completion hidden from runtime polls still updates the run list", async ({
  page,
  fakeApi,
}) => {
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });
  fakeApi!.setAutoStart(false);
  // Gate every run-detail request until the fake run has synchronously
  // started and finished: the first detail read must already be terminal
  // while the runtime never observed the run at all.
  let releaseDetail = () => {};
  const detailGate = new Promise<void>((resolve) => {
    releaseDetail = resolve;
  });
  await page.route(/\/api\/generations\/run[^/]+$/, async (route) => {
    await detailGate;
    await route.fallback();
  });
  await page.goto("/#/generate");
  await page.getByLabel("Prompt", { exact: true }).fill("quick completion");
  await page
    .getByRole("button", { name: "Generate", exact: true })
    .click();

  const results = page.getByLabel("Run results");
  const chip = results.getByTitle("quick completion", { exact: true });
  await expect(chip).toContainText("Queued");

  fakeApi!.startNextRun();
  fakeApi!.finishCurrentRun();
  releaseDetail();

  // The detail view and the list chip both reach the terminal state even
  // though no poll ever observed the run as active.
  await expect(results.getByText("Completed", { exact: true }).first()).toBeVisible();
  await expect(chip).toContainText("Completed", { timeout: 5_000 });
});

test("a late submission response respects a newer model selection", async ({
  page,
  fakeApi,
}) => {
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });
  const turbo = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    profile: "z-image-turbo",
  });
  const saved = fakeApi!.seedRun({
    registrationId: turbo.id,
    profile: "z-image-turbo",
    prompt: "turbo saved",
  });

  let release = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let posted = false;
  await page.route("**/api/generations", async (route) => {
    if (route.request().method() === "POST") {
      posted = true;
      await gate;
    }
    await route.fallback();
  });
  await page.goto("/#/generate");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("delayed base submission");
  await page
    .getByRole("button", { name: "Generate", exact: true })
    .click();
  await expect.poll(() => posted).toBe(true);

  // While the POST is still in flight, the user switches models and the
  // panel follows the new model's saved run.
  await page.getByLabel("Model", { exact: true }).selectOption(turbo.id);
  const results = page.getByLabel("Run results");
  await expect(
    results.getByText(`run ${saved.run_id.slice(0, 8)}`).first(),
  ).toBeVisible();

  const response = page.waitForResponse(
    (request) =>
      request.request().method() === "POST" &&
      new URL(request.url()).pathname === "/api/generations",
  );
  release();
  await response;

  // The late response belongs to the previous model: the newer model's
  // selection must survive it.
  await expect(
    page.getByRole("button", { name: "Generate", exact: true }),
  ).toBeEnabled();
  await expect(
    results.getByText(`run ${saved.run_id.slice(0, 8)}`).first(),
  ).toBeVisible();
});
