import type { Page } from "@playwright/test";

import { expect, test } from "./fixtures/app";

/**
 * Representative screenshots of the implemented pages for visual review,
 * captured into the gitignored test-results directory. Desktop and narrow
 * viewports are both covered. Captures wait for lazy media to decode and
 * scroll back to the top so sticky elements render in their real resting
 * position; the app's CSS is never adjusted for screenshots.
 */

async function captureDecoded(
  page: Page,
  path: string,
  imageSelector: string,
  fullPage = true,
) {
  if (imageSelector !== "") {
    // Wait until every matching image finished loading and decoded.
    await expect
      .poll(
        async () =>
          page.evaluate((selector) => {
            const images = Array.from(
              document.querySelectorAll<HTMLImageElement>(selector),
            );
            return (
              images.length > 0 &&
              images.every((img) => img.complete && img.naturalWidth > 0)
            );
          }, imageSelector),
        { message: `images matching ${imageSelector} must decode` },
      )
      .toBe(true);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(150);
  await page.screenshot({ path, fullPage });
}

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "seeded fixtures only");
  const registration = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image",
  });
  fakeApi!.setResidentIdle(registration.id, "gpu-fake-0");
  fakeApi!.seedRun({ prompt: "a quiet harbor at dawn, soft light" });
  fakeApi!.seedRun({
    prompt: "a stormy sea in oil painting style",
    status: "partial",
    completed: 1,
  });
});

test("captures the four pages on desktop and narrow viewports", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.goto("/#/generate");
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("a quiet harbor at dawn, soft light");
  await captureDecoded(page, "test-results/generate-desktop.png", "");

  await page.goto("/#/models");
  await page.getByRole("tab", { name: "My Models" }).click();
  await expect(page.getByText("Tongyi-MAI/Z-Image").first()).toBeVisible();
  await captureDecoded(page, "test-results/models-desktop.png", "");

  await page.goto("/#/history");
  await expect(
    page.getByText("a quiet harbor at dawn, soft light"),
  ).toBeVisible();
  await captureDecoded(
    page,
    "test-results/history-desktop.png",
    ".run-card img",
  );

  await page.goto("/#/settings");
  await expect(page.getByText(/127\.0\.0\.1:7860/)).toBeVisible();
  await captureDecoded(page, "test-results/settings-desktop.png", "");

  await page.setViewportSize({ width: 390, height: 844 });

  await page.goto("/#/generate");
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("a quiet harbor at dawn, soft light");
  await captureDecoded(page, "test-results/generate-narrow.png", "");

  await page.goto("/#/history");
  await expect(
    page.getByText("a quiet harbor at dawn, soft light"),
  ).toBeVisible();
  await captureDecoded(
    page,
    "test-results/history-narrow.png",
    ".run-card img",
  );

  await page.getByRole("button", { name: "Open navigation menu" }).click();
  await expect(page.locator("#app-sidebar")).toHaveAttribute(
    "data-open",
    "true",
  );
  await captureDecoded(
    page,
    "test-results/history-narrow-menu.png",
    ".run-card img",
    false,
  );
});

test("captures a completed run's results panel", async ({ page, fakeApi }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/#/generate");
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("a completed run for the screenshot");
  await page.getByLabel("Images", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Generate" }).click();

  await expect(page.getByText("2/2 images")).toBeVisible({ timeout: 15_000 });
  void fakeApi;
  await captureDecoded(
    page,
    "test-results/results-desktop.png",
    ".image-tile__button img",
  );
});
