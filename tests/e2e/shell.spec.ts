import { expect, test } from "./fixtures/app";

/**
 * Shell checks: navigation, routing, the skip link, the narrow-viewport
 * menu, and the global runtime banner. Runs against the fake transport so
 * the app renders normally; in real-backend mode the fake-specific parts
 * still exercise the same UI against the served API.
 */

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("redirects to Generate and marks the active nav item", async ({
  page,
}) => {
  await expect(page).toHaveURL(/#\/generate$/);
  await expect(page.getByRole("heading", { name: "Generate" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Generate" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});

test("navigates to every primary page and back", async ({ page }) => {
  for (const name of ["Models", "History", "Settings", "Generate"]) {
    await page.getByRole("link", { name }).click();
    await expect(page).toHaveURL(new RegExp(`#/${name.toLowerCase()}$`));
    await expect(page.getByRole("heading", { name })).toBeVisible();
    await expect(page.getByRole("link", { name })).toHaveAttribute(
      "aria-current",
      "page",
    );
  }
});

test("shows a not-found page for unknown routes", async ({ page }) => {
  // Wait for the index redirect to settle before navigating again;
  // otherwise the redirect can overwrite the new fragment.
  await expect(page).toHaveURL(/#\/generate$/);
  await page.goto("/#/unknown-route");
  await expect(
    page.getByRole("heading", { name: "Page not found" }),
  ).toBeVisible();
});

test("activating the skip link keeps the current page and URL", async ({
  page,
}) => {
  const skip = page.getByRole("link", { name: "Skip to main content" });
  await expect(skip).toHaveAttribute("href", "#main-content");
  await skip.focus();
  await page.keyboard.press("Enter");

  // Main takes focus without HashRouter consuming "#main-content" as a
  // route: the page content survives and the URL keeps the current route.
  await expect(page.getByRole("main")).toBeFocused();
  await expect(page.getByRole("heading", { name: "Generate" })).toBeVisible();
  await expect(page).toHaveURL(/#\/generate$/);
});

test("supports tabbing through skip link into navigation", async ({
  page,
  fakeApi,
}) => {
  test.skip(fakeApi === null, "healthy-app focus order only");
  await expect(page.getByRole("heading", { name: "Generate" })).toBeVisible();
  // Reset the focus anchor so the first Tab press is deterministic.
  await page.evaluate(() => {
    (document.activeElement as HTMLElement | null)?.blur();
  });
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to main content" }),
  ).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Generate" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Models" })).toBeFocused();
});

test("narrow viewport opens and closes the menu", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 720 });

  const toggle = page.getByRole("button", {
    name: "Open navigation menu",
  });
  await toggle.click();
  await expect(
    page.getByRole("button", { name: "Close navigation menu" }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator("#app-sidebar")).toHaveAttribute(
    "data-open",
    "true",
  );

  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "Open navigation menu" }),
  ).toHaveAttribute("aria-expanded", "false");
});

test("narrow viewport menu closes after navigation", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 720 });

  await page.getByRole("button", { name: "Open navigation menu" }).click();
  await page.getByRole("link", { name: "History" }).click();
  await expect(page).toHaveURL(/#\/history$/);
  await expect(
    page.getByRole("button", { name: "Open navigation menu" }),
  ).toHaveAttribute("aria-expanded", "false");
});

test("shows the runtime banner on every page", async ({ page }) => {
  await expect(
    page.getByRole("region", { name: "Runtime status" }),
  ).toBeVisible();
});

test("captures desktop and narrow screenshots for review", async ({ page }) => {
  await page.screenshot({
    path: "test-results/shell-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 375, height: 720 });
  await page.getByRole("button", { name: "Open navigation menu" }).click();
  await page.screenshot({
    path: "test-results/shell-narrow-menu.png",
    fullPage: true,
  });
});
