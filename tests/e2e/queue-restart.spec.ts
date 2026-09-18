import { expect, test } from "./fixtures/app";

/**
 * Queue pause after a simulated server restart and explicit resume in
 * original FIFO order.
 */

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    profile: "z-image-turbo",
  });
  // Advance two steps per poll so the running phase stays observable.
  fakeApi!.setStepsPerTick(2);
  await page.goto("/#/generate");
});

test("restart pauses the queue; resume continues the paused run", async ({
  page,
  fakeApi,
}) => {
  fakeApi!.setAutoStart(false);
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();

  await page.getByLabel("Prompt", { exact: true }).fill("paused run");
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(page.getByText("paused run")).toBeVisible({ timeout: 5_000 });

  fakeApi!.simulateRestart();
  // A restarted dispatcher accepts work again.
  fakeApi!.setAutoStart(true);

  await expect(
    page.getByText(/queue is paused after a server restart/i),
  ).toBeVisible({ timeout: 5_000 });

  // While paused, new submissions are refused with the contract's 409
  // and no run is persisted.
  const runsBefore = fakeApi!.state().runs.length;
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("rejected while paused");
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(
    page.getByText(/Resume it before submitting new runs/i),
  ).toBeVisible({ timeout: 5_000 });
  expect(fakeApi!.state().runs.length).toBe(runsBefore);

  await page.getByRole("button", { name: "Resume queue" }).click();

  await expect(page.getByText("Running").first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText("paused run").first()).toBeVisible();
  await expect(page.getByText("Completed").first()).toBeVisible({
    timeout: 15_000,
  });
});
