import { expect, test } from "./fixtures/app";

/**
 * Model/GPU replacement notice, Eject idle vs busy, and cancellation of
 * queued and running work. Task cancellation is separate from Eject.
 */

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  const registration = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image",
  });
  fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    profile: "z-image-turbo",
  });
  fakeApi!.setResidentIdle(registration.id, "gpu-fake-0");
  await page.goto("/#/generate");
});

test("announces a cross-GPU replacement before submission", async ({
  page,
  fakeApi,
}) => {
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Reuses the resident model on its current GPU."),
  ).toBeVisible();

  await page.getByLabel("GPU", { exact: true }).selectOption("gpu-fake-1");
  await expect(
    page.getByText(/is fully unloaded and .* loads on Fake GPU 1/),
  ).toBeVisible();

  // Submit and let the fake complete the switch; the banner follows.
  await page.getByLabel("Prompt", { exact: true }).fill("cross-gpu run");
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(page.getByText("Fake GPU 1").first()).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByText("Idle").first()).toBeVisible({
    timeout: 15_000,
  });
});

test("eject is idle-only: disabled while generating, working when idle", async ({
  page,
  fakeApi,
}) => {
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();

  // Select the Turbo model and slow the fake down so the generating
  // state is observable before the run finishes.
  const turboValue = await page
    .locator("#generation-model option", { hasText: "Z-Image-Turbo" })
    .getAttribute("value");
  await page.getByLabel("Model").selectOption(turboValue!);
  fakeApi!.setStepsPerTick(2);

  const eject = page.getByRole("button", { name: "Eject model" });
  await expect(eject).toBeEnabled();

  // Start a run; the button must disable with a reason while busy.
  await page.getByLabel("Prompt", { exact: true }).fill("busy run");
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(page.getByText("Generating").first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(eject).toBeDisabled();
  await expect(eject).toHaveAttribute(
    "title",
    "Eject is unavailable while the worker is generating.",
  );

  fakeApi!.finishCurrentRun();
  await expect(page.getByText("Idle").first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(eject).toBeEnabled();

  await eject.click();
  await page.getByRole("button", { name: /^Eject$/ }).click();
  await expect(
    page.getByText(/Files and registrations are untouched/i),
  ).toBeVisible();
  await expect(page.getByText("No model loaded").first()).toBeVisible();
  expect(fakeApi!.state().runtimeState).toBe("unloaded");
});

test("cancels a queued run immediately", async ({ page, fakeApi }) => {
  fakeApi!.setAutoStart(false);
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();

  await page.getByLabel("Prompt", { exact: true }).fill("queued one");
  await page.getByRole("button", { name: "Generate" }).click();
  await page.getByLabel("Prompt", { exact: true }).fill("queued two");
  await page.getByRole("button", { name: "Generate" }).click();

  const queuePanel = page.getByLabel("Generation queue");
  await expect(queuePanel.getByText("queued one")).toBeVisible({
    timeout: 5_000,
  });
  await expect(queuePanel.getByText("queued two")).toBeVisible();

  await page
    .getByRole("listitem")
    .filter({ hasText: "queued one" })
    .getByRole("button", { name: "Cancel" })
    .click();

  await expect(queuePanel.getByText("queued one")).toBeHidden({
    timeout: 5_000,
  });
  await expect(queuePanel.getByText("queued two")).toBeVisible();
});

test("cancels a running run cooperatively and keeps completed images", async ({
  page,
  fakeApi,
}) => {
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  // The fake advances per HTTP poll, and submission triggers a burst of
  // polls: one-image-per-tick would finish all four images inside the
  // burst (a completed run, not a cooperative cancel). Ten steps per tick
  // complete the first image only after the burst settles.
  fakeApi!.setStepsPerTick(10);
  await page.getByLabel("Prompt", { exact: true }).fill("cancel me");
  await page.getByLabel("Images", { exact: true }).selectOption("4");
  await page.getByRole("button", { name: "Generate" }).click();

  const results = page.getByLabel("Run results");
  // At least one image completed while later images are still pending.
  await expect(results.getByText(/[12]\/4 images/)).toBeVisible({
    timeout: 10_000,
  });

  // Now the remaining images advance barely at all: the cancel below lands
  // while the run is genuinely running (a bounded race — the slow pace
  // leaves many seconds of margin). The exact match keeps the queue's
  // Cancel button distinguishable from result-list chips whose accessible
  // names embed the prompt ("cancel me").
  fakeApi!.setStepsPerTick(1);
  await page
    .getByRole("button", { name: "Cancel", exact: true })
    .last()
    .click();

  await expect(page.getByText(/Cancellation requested/i)).toBeVisible();
  await expect(page.getByText("Partial").first()).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByText("Idle").first()).toBeVisible({
    timeout: 15_000,
  });
  // The completed image(s) are kept in the partial run.
  await expect(results.getByText(/[12]\/4 images/)).toBeVisible();
});

test("fails a run visibly on a worker error", async ({ page, fakeApi }) => {
  fakeApi!.failCurrentRun("worker_error", "Simulated OOM.");
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  await page.getByLabel("Prompt", { exact: true }).fill("doomed run");
  await page.getByRole("button", { name: "Generate" }).click();

  const results = page.getByLabel("Run results");
  await expect(results.getByText(/Simulated OOM./).first()).toBeVisible({
    timeout: 15_000,
  });
  await expect(results.getByText("Failed").first()).toBeVisible();
  await expect(page.getByText("No model loaded").first()).toBeVisible({
    timeout: 5_000,
  });
});

test("viewing another queued run fetches it without waiting for a stalled detail", async ({
  page,
  fakeApi,
}) => {
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });
  fakeApi!.setAutoStart(false);
  const first = fakeApi!.seedRun({
    prompt: "stalled detail A",
    status: "queued",
  });
  const second = fakeApi!.seedRun({
    prompt: "stalled detail B",
    status: "queued",
  });

  // Stall every detail request for run A behind a manual gate.
  let release: () => void = () => undefined;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let firstRequested = false;
  let secondRequested = false;
  await page.route(`**/api/generations/${first.run_id}`, async (route) => {
    firstRequested = true;
    await gate;
    await route.fallback();
  });
  page.on("request", (request) => {
    if (
      new URL(request.url()).pathname === `/api/generations/${second.run_id}`
    ) {
      secondRequested = true;
    }
  });

  try {
    await page.goto("/#/generate");
    const queuePanel = page.getByLabel("Generation queue");
    await expect(queuePanel.getByText("stalled detail A")).toBeVisible({
      timeout: 5_000,
    });

    await queuePanel
      .getByRole("listitem")
      .filter({ hasText: "stalled detail A" })
      .getByRole("button", { name: "View" })
      .click();
    await expect.poll(() => firstRequested).toBe(true);

    // Switch to run B while A's detail is still stalled: B must be
    // requested immediately, not after A completes.
    await queuePanel
      .getByRole("listitem")
      .filter({ hasText: "stalled detail B" })
      .getByRole("button", { name: "View" })
      .click();
    await expect
      .poll(() => secondRequested, {
        timeout: 3_000,
        message: "new selected run must fetch independently",
      })
      .toBe(true);

    // Once released, the late A response must not replace B's view.
    release();
    const results = page.getByLabel("Run results");
    await expect(
      results.getByText(`run ${second.run_id.slice(0, 8)}`).first(),
    ).toBeVisible({ timeout: 10_000 });
  } finally {
    release();
  }
});
