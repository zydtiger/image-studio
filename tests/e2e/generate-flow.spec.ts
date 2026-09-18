import { expect, test } from "./fixtures/app";

/**
 * Core generation flow against the transport-level fake API: choose model
 * and GPU, write a prompt, submit, follow queue progress to per-image
 * results with seeds, and verify the banner keeps the resident model and
 * GPU after the run clears.
 */

test.beforeEach(async ({ page, fakeApi }) => {
  test.skip(fakeApi === null, "transport fixtures only");
  fakeApi!.addRegistration({ repo_id: "Tongyi-MAI/Z-Image" });
  fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image-Turbo",
    profile: "z-image-turbo",
  });
  await page.goto("/#/generate");
});

test("generates images with recorded seeds and keeps the resident model visible", async ({
  page,
  fakeApi,
}) => {
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Model")).toHaveValue(/reg/);

  // Turbo with throttled progress keeps the generating phase observable.
  const turboValue = await page
    .locator("#generation-model option", { hasText: "Z-Image-Turbo" })
    .getAttribute("value");
  await page.getByLabel("Model").selectOption(turboValue!);
  fakeApi!.setStepsPerTick(2);

  await page
    .getByLabel("Prompt", { exact: true })
    .fill("a quiet harbor at dawn, 你好");
  await page.getByLabel("Seed", { exact: true }).fill("500");
  await page.getByLabel("Images", { exact: true }).selectOption("2");
  await page.getByRole("button", { name: "Generate" }).click();

  // The banner shows the fake runtime and then the running worker.
  await expect(page.getByText("Fake runtime (development)")).toBeVisible();
  await expect(page.getByText("Generating").first()).toBeVisible({
    timeout: 10_000,
  });

  // Results appear with per-image seeds once the run completes.
  const results = page.getByLabel("Run results");
  await expect(results.getByText("Completed").first()).toBeVisible({
    timeout: 20_000,
  });
  await expect(results.getByText("seed 500").first()).toBeVisible();
  await expect(results.getByText("seed 501").first()).toBeVisible();
  await expect(results.getByText("2/2 images")).toBeVisible();

  // Images are real decodable media, not just status text.
  const decoded = await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const images = Array.from(
            document.querySelectorAll<HTMLImageElement>(".image-tile__button img"),
          );
          return {
            count: images.length,
            allDecoded: images.every((img) => img.complete && img.naturalWidth > 0),
          };
        }),
      { message: "result thumbnails must decode" },
    )
    .toEqual({ count: 2, allDecoded: true });
  void decoded;

  // The artifact bytes are a valid PNG served with the right type.
  const firstRun = fakeApi!.state().runs.at(-1)!;
  const artifactCheck = await page.evaluate(async (url) => {
    const response = await fetch(url);
    const buffer = new Uint8Array(await response.arrayBuffer());
    return {
      status: response.status,
      type: response.headers.get("content-type"),
      signature: Array.from(buffer.slice(0, 8)),
    };
  }, `/api/generations/${firstRun.run_id}/artifacts/${firstRun.images[0].artifact_id}?download=1`);
  expect(artifactCheck.status).toBe(200);
  expect(artifactCheck.type).toBe("image/png");
  expect(artifactCheck.signature).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);

  // The banner keeps resident model AND GPU after current_run_id clears.
  await expect(page.getByText("Tongyi-MAI/Z-Image").first()).toBeVisible();
  await expect(page.getByText("Fake GPU 0").first()).toBeVisible();
  await expect(page.getByText("Idle").first()).toBeVisible();
  expect(fakeApi!.state().currentRunId).toBeNull();
});

test("a second submission in the same session replaces the completed run", async ({
  page,
  fakeApi,
}) => {
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();
  const turboValue = await page
    .locator("#generation-model option", { hasText: "Z-Image-Turbo" })
    .getAttribute("value");
  await page.getByLabel("Model").selectOption(turboValue!);
  fakeApi!.setStepsPerTick(0);

  const results = page.getByLabel("Run results");
  const runSeedLabel = (seed: number) =>
    results.getByText(`seed ${seed}`).first();

  await page.getByLabel("Prompt", { exact: true }).fill("first run");
  await page.getByLabel("Seed", { exact: true }).fill("100");
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(results.getByText("Completed").first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(runSeedLabel(100)).toBeVisible();

  // Terminal run A is followed by a fresh submission without leaving the
  // page; the panel must switch to run B and show its progress.
  await page.getByLabel("Prompt", { exact: true }).fill("second run");
  await page.getByLabel("Seed", { exact: true }).fill("200");
  await page.getByLabel("Images", { exact: true }).selectOption("2");
  await page.getByRole("button", { name: "Generate" }).click();

  await expect(runSeedLabel(200)).toBeVisible({ timeout: 10_000 });
  await expect(results.getByText("Completed").first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(results.getByText("seed 201").first()).toBeVisible();
  await expect(runSeedLabel(100)).toBeHidden();
});

test("capability metadata drives the form per profile", async ({ page }) => {
  await page.getByLabel("Model").selectOption({ index: 1 });
  await expect(page.getByLabel("Negative prompt")).toBeVisible();
  await expect(page.getByLabel("Guidance")).toBeVisible();

  const turbo = page.locator("#generation-model option", {
    hasText: "Z-Image-Turbo",
  });
  await page
    .getByLabel("Model")
    .selectOption(await turbo.getAttribute("value"));
  await expect(page.getByLabel("Negative prompt")).toBeHidden();
  await expect(page.getByLabel("Guidance")).toBeHidden();
  await expect(page.getByLabel("Steps")).toHaveValue("9");
});

test("blocks submission until required fields are filled", async ({ page }) => {
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(page.getByText("Enter a prompt.")).toBeVisible();
});

test("rejects a gated model download with a clear error", async ({ page }) => {
  await page.getByRole("link", { name: "Models" }).click();
  await page.getByRole("tab", { name: "Discover" }).click();
  await page.getByLabel("Search Hugging Face").fill("gated");
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByRole("button", { name: "Details" }).click();
  await page.getByRole("button", { name: "Download…" }).click();
  await expect(page.getByText(/gated; Hub access is required/i)).toBeVisible();
});
