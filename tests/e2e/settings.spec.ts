import { expect, test } from "./fixtures/app";

/** Settings renders the served system information verbatim. */

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  await page.goto("/#/settings");
});

test("shows paths, GPUs, development flags, and restart note", async ({
  page,
}) => {
  await expect(page.getByText("Server")).toBeVisible();
  await expect(page.getByText(/127\.0\.0\.1:7860/)).toBeVisible();
  await expect(page.getByText("fake hub")).toBeVisible();
  await expect(page.getByText(/require a restart/i)).toBeVisible();

  await expect(page.getByText("Hugging Face")).toBeVisible();
  await expect(page.getByText("Fake GPU 0").first()).toBeVisible();
  await expect(
    page.getByText("/fake/.config/image-studio/config.toml"),
  ).toBeVisible();
  await expect(
    page.getByText("/fake/.local/share/image-studio/app.sqlite"),
  ).toBeVisible();
});

test("reports the resident model from the runtime status", async ({
  page,
  fakeApi,
}) => {
  const registration = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image",
  });
  fakeApi!.setResidentIdle(registration.id, "gpu-fake-1");

  await expect(page.getByText(/Tongyi-MAI\/Z-Image/).first()).toBeVisible({
    timeout: 5_000,
  });
  await expect(page.getByText("Fake GPU 1").first()).toBeVisible();
  await expect(page.getByText("bfloat16").first()).toBeVisible();
});
