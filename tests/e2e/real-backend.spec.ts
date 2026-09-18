import { expect, test } from "@playwright/test";

/**
 * Integration-mode checks against a REAL backend serving the built
 * frontend with fake infrastructure — no route interception anywhere.
 *
 * The full-flow test onboards a model from an EMPTY fake cache and
 * registers it, so it needs a FRESH server with isolated state per run:
 * point every XDG/HF path at a throwaway directory and use an unused
 * loopback port, e.g.
 *
 *   npm --prefix frontend run build
 *   RUN=$(mktemp -d) && env XDG_CONFIG_HOME=$RUN/config XDG_DATA_HOME=$RUN/data \
 *     XDG_CACHE_HOME=$RUN/cache XDG_STATE_HOME=$RUN/state HF_HOME=$RUN/hf \
 *     uv run --locked image-studio serve --fake-runtime --fake-hub \
 *       --host 127.0.0.1 --port 7860 &
 *   E2E_BASE_URL=http://127.0.0.1:7860 npm --prefix tests/e2e test real-backend
 *
 * HUB_QUERY matches the fake Hub catalog's repo ids ("z" finds both
 * Z-Image repositories). Stop the server afterwards; it owns the port
 * and the temporary state.
 */

const HUB_QUERY = "z";

test.beforeEach(() => {
  test.skip(
    process.env.E2E_BASE_URL === undefined,
    "integration mode only: set E2E_BASE_URL",
  );
});

test("runtime status answers with the fake implementation marker", async ({
  request,
}) => {
  const response = await request.get("/api/runtime");
  expect(response.ok()).toBeTruthy();
  const body = (await response.json()) as { implementation: string };
  expect(body.implementation).toBe("fake");
});

test("profiles capability metadata is served", async ({ request }) => {
  const response = await request.get("/api/profiles");
  expect(response.ok()).toBeTruthy();
  const body = (await response.json()) as {
    profiles: Array<{ profile_id: string }>;
  };
  const ids = body.profiles.map((profile) => profile.profile_id);
  expect(ids).toContain("z-image");
  expect(ids).toContain("z-image-turbo");
});

test("full flow: download+register, generate twice on one page, history, downloads, eject", async ({
  page,
  request,
}) => {
  test.setTimeout(240_000);

  // --- Deterministic model onboarding for a fresh, empty fake cache:
  // download through the real dialog, then register from the cache. ---
  await page.goto("/#/models");
  await page.getByRole("tab", { name: "Discover" }).click();
  await page.getByLabel("Search Hugging Face").fill(HUB_QUERY);
  await page.getByRole("button", { name: "Search" }).click();

  const detailButton = page.getByRole("button", { name: "Details" }).first();
  await expect(detailButton).toBeVisible({ timeout: 15_000 });
  await detailButton.click();

  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByText(/Structurally compatible|Not compatible/i),
  ).toBeVisible({ timeout: 15_000 });
  await dialog.getByRole("button", { name: "Download…" }).click();

  await page.getByRole("tab", { name: "Downloads" }).click();
  await expect(page.getByText("Completed").first()).toBeVisible({
    timeout: 180_000,
  });

  await page.getByRole("tab", { name: "Local Cache" }).click();
  await page.getByRole("button", { name: "Register" }).first().click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Register" })
    .click();
  await expect(page.getByText(/registered/i).first()).toBeVisible({
    timeout: 15_000,
  });

  // --- Generate page loaded ONCE; both runs submit on the same page. ---
  await page.goto("/#/generate");
  await expect(page.getByLabel("Prompt", { exact: true })).toBeVisible();

  const results = page.getByLabel("Run results");
  const gpuSelect = page.getByLabel("GPU", { exact: true });
  const modelSelect = page.getByLabel("Model");
  await expect(modelSelect).toBeVisible();
  // Explicit GPU selection (first device), never relying on a default.
  const gpuOption = await gpuSelect.locator("option").nth(1).textContent();
  expect(gpuOption).toBeTruthy();
  await gpuSelect.selectOption({ index: 1 });

  const expectDecodedImages = async (expectedCount: number) => {
    await expect
      .poll(
        async () => {
          const images = await page.evaluate((selector) =>
            Array.from(
              document.querySelectorAll<HTMLImageElement>(selector),
            ).map((img) => ({
              complete: img.complete,
              naturalWidth: img.naturalWidth,
            })),
          ".image-tile__button img");
          return (
            images.length === expectedCount &&
            images.every((img) => img.complete && img.naturalWidth > 0)
          );
        },
        {
          message: `${expectedCount} result images must load and decode`,
          timeout: 30_000,
        },
      )
      .toBe(true);
  };

  // --- First run: 2 images, explicit seed, Unicode prompt. ---
  await modelSelect.selectOption({ index: 1 });
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("integration alpha, a quiet harbor at dawn 你好");
  await page.getByLabel("Seed", { exact: true }).fill("400");
  await page.getByLabel("Images", { exact: true }).selectOption("2");
  await page.getByRole("button", { name: "Generate" }).click();

  await expect(results.getByText("Completed").first()).toBeVisible({
    timeout: 180_000,
  });
  await expect(results.getByText("2/2 images")).toBeVisible();
  await expectDecodedImages(2);

  // Browser download event for an image via the lightbox.
  await results
    .getByRole("button", { name: /Open image 1, seed 400/i })
    .click();
  const imageDownload = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download PNG" }).click();
  const imageDownloadInfo = await imageDownload;
  expect(imageDownloadInfo.suggestedFilename()).toMatch(/\.png$/);
  // Close the lightbox so the overlay cannot intercept the second run.
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close dialog" })
    .click();
  await expect(page.getByRole("dialog")).toBeHidden();

  // --- Second run on the SAME page (no reload): 1 image. ---
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("integration beta, second same-page run");
  await page.getByLabel("Seed", { exact: true }).fill("500");
  await page.getByLabel("Images", { exact: true }).selectOption("1");
  await page.getByRole("button", { name: "Generate" }).click();

  await expect(results.getByText("seed 500").first()).toBeVisible({
    timeout: 180_000,
  });
  await expect(results.getByText("Completed").first()).toBeVisible({
    timeout: 180_000,
  });
  await expectDecodedImages(1);

  // --- Metadata through the real API for the FIRST (batch) run. ---
  const alphaResponse = await request.get(
    "/api/generations?q=integration%20alpha&limit=10",
  );
  expect(alphaResponse.ok()).toBeTruthy();
  const alphaBody = (await alphaResponse.json()) as {
    runs: Array<{ run_id: string; status: string }>;
  };
  expect(alphaBody.runs.length).toBeGreaterThanOrEqual(1);
  const alphaRun = alphaBody.runs[0];
  expect(alphaRun.status).toBe("completed");

  const alphaDetail = (await (
    await request.get(`/api/generations/${alphaRun.run_id}`)
  ).json()) as {
    status: string;
    prompt: string;
    initial_seed: number;
    image_count: number;
    gpu: { name: string } | null;
    images: Array<{ artifact_id: string; status: string; seed: number }>;
  };
  // The first run's batch metadata: completed, Unicode prompt, seeds
  // 400/401, and the explicitly selected GPU.
  expect(alphaDetail.status).toBe("completed");
  expect(alphaDetail.prompt).toContain("你好");
  expect(alphaDetail.image_count).toBe(2);
  expect(alphaDetail.gpu?.name).toBe(gpuOption!);
  expect(alphaDetail.images.map((image) => image.seed)).toEqual([400, 401]);
  expect(
    alphaDetail.images.every((image) => image.status === "completed"),
  ).toBe(true);

  // --- Artifact and metadata bytes are valid over real HTTP. ---
  const artifactResponse = await request.get(
    `/api/generations/${alphaRun.run_id}/artifacts/${alphaDetail.images[0].artifact_id}?download=1`,
  );
  expect(artifactResponse.ok()).toBeTruthy();
  expect(artifactResponse.headers()["content-type"]).toContain("image/png");
  const artifactBytes = new Uint8Array(await artifactResponse.body());
  expect(Array.from(artifactBytes.slice(0, 4))).toEqual([137, 80, 78, 71]);

  // --- History: favorite, browser metadata download, trash, restore. ---
  await page.goto("/#/history");
  const card = page
    .locator(".run-card")
    .filter({ hasText: "integration beta, second same-page run" });
  await expect(card).toBeVisible({ timeout: 15_000 });
  await card.getByRole("button", { name: "Mark favorite" }).click();
  await expect(
    card.getByRole("button", { name: "Remove favorite" }),
  ).toBeVisible({ timeout: 10_000 });

  await card.getByRole("button", { name: /Open run from/i }).click();
  // Browser download event for metadata from the drawer footer.
  const metadataDownload = page.waitForEvent("download");
  await page.getByRole("link", { name: "Metadata" }).click();
  const metadataDownloadInfo = await metadataDownload;
  expect(metadataDownloadInfo.suggestedFilename()).toMatch(/metadata\.json$/);

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
  await page.getByRole("button", { name: "Close dialog" }).first().click();
  await expect(card).toBeHidden({ timeout: 10_000 });

  await page.getByRole("button", { name: "Trash" }).click();
  const trashedCard = page
    .locator(".run-card")
    .filter({ hasText: "integration beta, second same-page run" });
  await expect(trashedCard).toBeVisible({ timeout: 10_000 });
  await trashedCard.getByRole("button", { name: /Open run from/i }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Restore" })
    .click();
  await expect(trashedCard).toBeHidden({ timeout: 10_000 });

  // --- Manual Eject clears the resident model. ---
  await page.goto("/#/generate");
  const eject = page.getByRole("button", { name: "Eject model" });
  await expect(eject).toBeEnabled({ timeout: 30_000 });
  await eject.click();
  await page.getByRole("button", { name: /^Eject$/ }).click();
  await expect(page.getByText("No model loaded").first()).toBeVisible({
    timeout: 15_000,
  });
  const runtimeAfter = (await (await request.get("/api/runtime")).json()) as {
    resident: unknown;
  };
  expect(runtimeAfter.resident).toBeNull();
});
