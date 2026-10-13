import { expect, test } from "@playwright/test";

test("uploads, edits, reuses, and edits a result through the real API", async ({
  page,
  request,
}, testInfo) => {
  test.skip(
    !process.env.E2E_BASE_URL,
    "requires isolated fake-backend integration mode",
  );
  const download = await request.post("/api/downloads", {
    data: { repo_id: "Qwen/Qwen-Image-2.1", profile: "qwen-image-2.1" },
  });
  expect(download.ok()).toBeTruthy();
  const job = await download.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/downloads")).json()).jobs.find(
          (entry: { id: string }) => entry.id === job.id,
        )?.status,
    )
    .toBe("completed");
  const models = await (await request.get("/api/models")).json();
  const reg = models.registrations.find(
    (entry: { profile: string }) => entry.profile === "qwen-image-2.1",
  );
  await page.goto("/#/generate");
  await expect(page.getByLabel("Steps", { exact: true })).not.toHaveValue("");
  await page.getByLabel("Model", { exact: true }).selectOption(reg.id);
  await page.getByRole("radio", { name: "Image edit", exact: true }).check();
  await page
    .getByLabel("Edit instruction")
    .fill("Change the background to blue 山");
  await page.getByRole("button", { name: "Generate", exact: true }).click();
  await expect(page.getByText("Select one input image.")).toBeVisible();
  const png = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 384;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#509b82";
    ctx.fillRect(0, 0, 512, 384);
    ctx.fillStyle = "#eacf58";
    ctx.fillRect(100, 80, 180, 200);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  await page.getByLabel("Upload input image").setInputFiles({
    name: "reference.png",
    mimeType: "image/png",
    buffer: Buffer.from(png, "base64"),
  });
  await expect(
    page.getByRole("button", { name: "Replace image" }),
  ).toBeEnabled();
  await expect(page.getByLabel("Width", { exact: true })).toHaveValue("512");
  await expect(page.getByLabel("Height", { exact: true })).toHaveValue("384");
  for (const viewport of [
    { width: 1440, height: 1000 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => {
      window.scrollTo(0, 0);
      (document.activeElement as HTMLElement)?.blur();
    });
    if (viewport.width < 768)
      await expect(page.locator(".app-sidebar")).toBeHidden();
    await expect
      .poll(async () =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    await page.screenshot({
      path: testInfo.outputPath(`image-edit-${viewport.width}.png`),
      fullPage: true,
      animations: "disabled",
    });
  }
  const submitted = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/generations") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Generate", exact: true }).click();
  const run = await (await submitted).json();
  expect(run.mode).toBe("image-edit");
  expect(run.input_images).toHaveLength(1);
  await expect
    .poll(
      async () =>
        (await (await request.get(`/api/generations/${run.run_id}`)).json())
          .status,
    )
    .toBe("completed");
  const results = page.getByLabel("Run results");
  await expect(
    results.getByRole("button", { name: "Edit image", exact: true }).first(),
  ).toBeVisible();
  await results
    .getByRole("button", { name: "Edit image", exact: true })
    .first()
    .click();
  await expect(page.getByLabel("Edit instruction")).toHaveValue("");
  await expect(
    page.getByRole("radio", { name: "Image edit", exact: true }),
  ).toBeChecked();
  await expect(
    page.getByRole("button", { name: "Replace image" }),
  ).toBeEnabled();
  await request.post(`/api/generations/${run.run_id}/trash`);
  expect((await request.get(run.input_images[0].url)).ok()).toBeTruthy();
  await request.post(`/api/generations/${run.run_id}/restore`);
  await page.getByLabel("Edit instruction").fill("A second edit");
  const secondSubmitted = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/generations") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Generate", exact: true }).click();
  const second = await (await secondSubmitted).json();
  expect(second.input_images[0].input_id).not.toBe(
    run.input_images[0].input_id,
  );
  await expect
    .poll(
      async () =>
        (await (await request.get(`/api/generations/${second.run_id}`)).json())
          .status,
    )
    .toBe("completed");
  await results
    .getByRole("button", { name: "Edit image", exact: true })
    .first()
    .click();
  await expect(page.getByLabel("Edit instruction")).toHaveValue("");
  await page.goto("/#/history");
  const card = page.locator(".run-card").filter({
    has: page.locator(`img[src*="${second.run_id}"]`),
  });
  await card.getByRole("button", { name: /Open run from/ }).click();
  const drawer = page.getByRole("dialog");
  await expect(drawer.getByAltText("Input image")).toBeVisible();
  await drawer.getByRole("button", { name: "Reuse parameters" }).click();
  await expect(page.getByLabel("Edit instruction")).toHaveValue(
    "A second edit",
  );
  await expect(page.locator(".input-image-picker img")).toHaveAttribute(
    "src",
    second.input_images[0].url,
  );
});
