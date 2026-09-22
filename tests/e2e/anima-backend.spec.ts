import { expect, test } from "@playwright/test";

// Run against an isolated real API with --fake-runtime --fake-hub.
// No route interception, model weights or GPU execution.
test.beforeEach(() => {
  test.skip(
    !process.env.E2E_BASE_URL,
    "requires isolated fake-backend integration mode",
  );
});

test("original Anima checkpoints onboard, generate and retain their profile", async ({
  page,
  request,
}) => {
  test.setTimeout(90_000);
  const models = [
    {
      repo: "circlestone-labs/Anima",
      profile: "anima-turbo",
      steps: 10,
      guidance: 1,
    },
    {
      repo: "Gazingstars123/Anima-2.9B",
      profile: "anima-2.9b",
      steps: 40,
      guidance: 4,
    },
  ];
  for (const model of models) {
    await page.goto("/#/models");
    await page.getByRole("tab", { name: "Discover" }).click();
    await page.getByLabel("Search Hugging Face").fill(model.repo);
    await page.getByRole("button", { name: "Search", exact: true }).click();
    const card = page
      .locator(".model-card")
      .filter({ has: page.getByText(model.repo, { exact: true }) });
    await card.getByRole("button", { name: "Details" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByText("Structurally compatible")).toBeVisible();
    await expect(dialog.getByLabel("Profile")).toHaveValue(model.profile);
    const queued = page.waitForResponse(
      (r) =>
        r.url().endsWith("/api/downloads") && r.request().method() === "POST",
    );
    await dialog.getByRole("button", { name: "Download…" }).click();
    const job = (await (await queued).json()) as { id: string };
    await expect
      .poll(async () => {
        const response = await request.get("/api/downloads");
        const { jobs } = (await response.json()) as {
          jobs: { id: string; status: string }[];
        };
        return jobs.find((entry) => entry.id === job.id)?.status;
      })
      .toBe("completed");

    await page.getByRole("tab", { name: "My Models" }).click();
    await expect(page.getByText(model.repo, { exact: true })).toBeVisible();
    const library = await (await request.get("/api/models")).json();
    const registration = library.registrations.find(
      (entry: { repo_id: string; profile: string }) =>
        entry.repo_id === model.repo && entry.profile === model.profile,
    ) as { id: string; sources: unknown[] };
    expect(registration.sources).toHaveLength(2);

    await page.goto("/#/generate");
    // Let the initial model/profile queries populate defaults before switching models.
    await expect(page.getByLabel("Steps", { exact: true })).not.toHaveValue("");
    await page
      .getByLabel("Model", { exact: true })
      .selectOption(registration.id);
    await page.getByLabel("GPU", { exact: true }).selectOption("GPU-fake-0001");
    await expect(page.getByLabel("Steps", { exact: true })).toHaveValue(
      String(model.steps),
    );
    if (model.profile === "anima-turbo") {
      await expect(page.getByLabel("Guidance", { exact: true })).toHaveCount(0);
      await expect(
        page.getByLabel("Negative prompt", { exact: true }),
      ).toHaveCount(0);
    } else {
      await expect(page.getByLabel("Guidance", { exact: true })).toHaveValue(
        "4",
      );
      await page.getByLabel("Negative prompt", { exact: true }).fill("blurry");
    }
    await page
      .getByLabel("Prompt", { exact: true })
      .fill("A watercolor mountain landscape 山");
    const submitted = page.waitForResponse(
      (r) =>
        r.url().endsWith("/api/generations") && r.request().method() === "POST",
    );
    await page.getByRole("button", { name: "Generate", exact: true }).click();
    const run = (await (await submitted).json()) as {
      run_id: string;
      profile: string;
      guidance: number;
      sources: unknown[];
    };
    expect(run.profile).toBe(model.profile);
    expect(run.guidance).toBe(model.guidance);
    expect(run.sources).toEqual(registration.sources);
    await expect(page.locator(".image-tile__button img").first()).toBeVisible();
    await expect
      .poll(
        async () =>
          (await (await request.get(`/api/generations/${run.run_id}`)).json())
            .status,
      )
      .toBe("completed");
    const history = await (
      await request.get(`/api/generations/${run.run_id}`)
    ).json();
    expect(history.profile).toBe(model.profile);
    expect(history.sources).toEqual(registration.sources);
  }
  expect((await request.post("/api/runtime/eject")).ok()).toBeTruthy();
});
