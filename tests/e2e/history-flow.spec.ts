import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "./fixtures/app";

/**
 * History flows: statuses retained, favorites, prompt search, trash and
 * restore, parameter reuse (including forced GPU reselection), and
 * missing previews.
 */

/** Gitignored local visual-review shots for the default-view change. */
const REVIEW_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../.artifacts/ui-review/history-view",
);

function reviewShot(name: string): string {
  mkdirSync(REVIEW_DIR, { recursive: true });
  return path.join(REVIEW_DIR, name);
}

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  const registration = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image",
  });
  fakeApi!.seedRun({ prompt: "a quiet harbor at dawn" });
  fakeApi!.seedRun({
    prompt: "a stormy sea, oil painting",
    status: "partial",
    completed: 1,
  });
  fakeApi!.seedRun({
    prompt: "a broken render",
    status: "failed",
    imageCount: 2,
  });
  fakeApi!.seedRun({ prompt: "a favorite mountain", favorite: true });
  fakeApi!.seedRun({ prompt: "a retired rig", gpuUuid: "gpu-retired-xyz" });
  fakeApi!.seedRun({ prompt: "a deleted valley", trashed: true });
  void registration;
  await page.goto("/#/history");
});

test("retains completed, partial, and failed runs with clear states", async ({
  page,
}) => {
  await expect(page.getByText("a quiet harbor at dawn")).toBeVisible();
  const partialCard = page.locator(".run-card").filter({
    hasText: "a stormy sea",
  });
  const failedCard = page.locator(".run-card").filter({
    hasText: "a broken render",
  });
  await expect(partialCard.getByText("Partial")).toBeVisible();
  await expect(failedCard.getByText("Failed")).toBeVisible();
  await expect(page.getByText("a deleted valley")).toBeHidden();

  // Preview thumbnails are real decodable media served by the transport.
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const images = Array.from(
            document.querySelectorAll<HTMLImageElement>(".run-card img"),
          );
          return (
            images.length > 0 &&
            images.every((img) => img.complete && img.naturalWidth > 0)
          );
        }),
      { message: "history previews must decode" },
    )
    .toBe(true);
});

test("searches prompts and filters favorites", async ({ page }) => {
  await page.getByLabel("Search prompts").fill("harbor");
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByText("a quiet harbor at dawn")).toBeVisible();
  await expect(page.getByText("a stormy sea, oil painting")).toBeHidden();

  await page.getByLabel("Search prompts").fill("");
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByLabel("Favorites only").check();
  await expect(page.getByText("a favorite mountain")).toBeVisible();
  await expect(page.getByText("a quiet harbor at dawn")).toBeHidden();
});

test("toggles favorites from the grid", async ({ page }) => {
  const card = page.locator(".run-card").filter({
    hasText: "a quiet harbor at dawn",
  });
  await card.getByRole("button", { name: "Mark favorite" }).click();
  await expect(
    card.getByRole("button", { name: "Remove favorite" }),
  ).toBeVisible({ timeout: 5_000 });
});

test("moves a run to recoverable Trash directly from its History card", async ({
  page,
}) => {
  const card = page
    .locator(".run-card")
    .filter({ hasText: "a quiet harbor at dawn" });
  await card.getByRole("button", { name: "Move run to Trash" }).click();
  const confirm = page.getByRole("dialog", { name: "Move run to Trash?" });
  await expect(confirm).toContainText("entire run (2 images)");
  await confirm
    .getByRole("button", { name: "Move to Trash", exact: true })
    .click();
  await expect(card).toHaveCount(0);
  await page.getByRole("button", { name: "Trash", exact: true }).click();
  await expect(card).toBeVisible();
  await expect(
    card.getByRole("button", { name: "Move run to Trash" }),
  ).toHaveCount(0);
  await card.getByRole("button", { name: /Open run from/ }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Restore", exact: true })
    .click();
  await page.getByRole("button", { name: "Close dialog" }).click();
  await page.getByRole("button", { name: "Library", exact: true }).click();
  await expect(card).toBeVisible();
});

test("keeps the scrolled history grid and focus while saving a favorite", async ({
  page,
  fakeApi,
}) => {
  for (let index = 0; index < 18; index += 1) {
    fakeApi!.seedRun({ prompt: `scroll test run ${index}` });
  }
  await page.reload();
  await expect(page.locator(".run-card")).toHaveCount(23);
  const card = page.locator(".run-card").last();
  const star = card.getByRole("button", { name: "Mark favorite" });
  await star.scrollIntoViewIfNeeded();
  const scrollBefore = await page.evaluate(() => window.scrollY);
  expect(scrollBefore).toBeGreaterThan(500);

  let releaseRefresh!: () => void;
  const refreshGate = new Promise<void>((resolve) => {
    releaseRefresh = resolve;
  });
  let refreshStarted!: () => void;
  const refreshing = new Promise<void>((resolve) => {
    refreshStarted = resolve;
  });
  await page.route("**/api/generations?*", async (route) => {
    refreshStarted();
    await refreshGate;
    await route.fallback();
  });

  await star.click();
  await refreshing;
  try {
    await expect(page.locator(".run-card")).toHaveCount(23);
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);
    await expect(star).toBeFocused();
    await expect(page).toHaveURL(/#\/history$/);
  } finally {
    releaseRefresh();
  }
  await expect(
    card.getByRole("button", { name: "Remove favorite" }),
  ).toBeFocused();
  expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);
});

test("trashes and restores a run", async ({ page }) => {
  const card = page.locator(".run-card").filter({
    hasText: "a quiet harbor at dawn",
  });
  await card.getByRole("button", { name: /Open run from/i }).click();

  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Move to Trash" })
    .first()
    .click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Move to Trash" })
    .last()
    .click();

  // Close the drawer (Escape closed only the nested confirm), then the
  // card must be gone from the library view.
  await page.getByRole("button", { name: "Close dialog" }).first().click();
  await expect(
    page.locator(".run-card").filter({ hasText: "a quiet harbor at dawn" }),
  ).toBeHidden({ timeout: 5_000 });

  await page.getByRole("button", { name: "Trash", exact: true }).click();
  await expect(page.getByText("a quiet harbor at dawn")).toBeVisible({
    timeout: 5_000,
  });
  await expect(page.getByText("a deleted valley")).toBeVisible();

  const trashedCard = page.locator(".run-card").filter({
    hasText: "a quiet harbor at dawn",
  });
  await trashedCard.getByRole("button", { name: /Open run from/i }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Restore" })
    .click();

  // The restored run leaves the Trash view; other trashed runs stay.
  await expect(
    page.locator(".run-card").filter({ hasText: "a quiet harbor at dawn" }),
  ).toBeHidden({ timeout: 5_000 });
  await expect(page.getByText("a deleted valley")).toBeVisible();
});

test("hides cancelled runs without images in the default view only", async ({
  page,
  fakeApi,
}) => {
  fakeApi!.seedRun({ prompt: "a cancelled ghost", status: "cancelled" });
  fakeApi!.seedRun({
    prompt: "a cancelled discard",
    status: "cancelled",
    trashed: true,
  });
  await page.goto("/#/history");

  // The default view keeps the failed run with its error but hides the
  // cancellation that produced nothing.
  const failedCard = page.locator(".run-card").filter({
    hasText: "a broken render",
  });
  await expect(failedCard.getByText("Failed")).toBeVisible();
  await expect(page.getByText("a cancelled ghost")).toBeHidden();
  await page.screenshot({
    path: reviewShot("history-default-view.png"),
    fullPage: true,
  });

  // The explicit Cancelled filter exposes the hidden record.
  await page.getByLabel("Status", { exact: true }).selectOption("cancelled");
  await expect(page.getByText("a cancelled ghost")).toBeVisible();
  await page.screenshot({
    path: reviewShot("history-cancelled-filter.png"),
    fullPage: true,
  });

  // The Trash view remains fully inspectable, cancellations included.
  await page.getByRole("button", { name: "Trash", exact: true }).click();
  await expect(page.getByText("a cancelled discard")).toBeVisible({
    timeout: 5_000,
  });
});

test("reuses parameters and forces GPU reselection when the GPU is gone", async ({
  page,
  fakeApi,
}) => {
  const turbo = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    profile: "z-image-turbo",
  });
  const card = page.locator(".run-card").filter({
    hasText: "a retired rig",
  });
  await card.getByRole("button", { name: /Open run from/i }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Reuse parameters" })
    .click();

  await expect(page).toHaveURL(/#\/generate$/);
  await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
    "a retired rig",
  );

  // The historical GPU no longer exists: explicit reselection required.
  await expect(page.getByLabel("GPU", { exact: true })).toHaveValue("");
  await expect(
    page.getByText(/The GPU used by this run .* is not available/),
  ).toBeVisible();

  // Choosing a DIFFERENT model must not satisfy the GPU requirement: the
  // selection stays blank and the hint stays until the user picks a GPU.
  await page.getByLabel("Model", { exact: true }).selectOption(turbo.id);
  await expect(page.getByLabel("GPU", { exact: true })).toHaveValue("");
  await expect(
    page.getByText(/The GPU used by this run .* is not available/),
  ).toBeVisible();
});
