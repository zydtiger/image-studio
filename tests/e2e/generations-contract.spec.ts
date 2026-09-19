import { expect, test } from "./fixtures/app";

/**
 * Transport-level contract checks for the run-listing boolean filters:
 * the fake must mirror the real API's FastAPI boolean query parsing and
 * the repository's explicit-status-wins composition for the default
 * cancellation exclusion. Requests go through in-page fetches so they use
 * the same context routing as the application itself.
 */

interface ListingBody {
  runs: { prompt: string }[];
  total: number;
}

test("run-listing boolean filters follow the API contract", async ({
  page,
  fakeApi,
}) => {
  test.skip(fakeApi === null, "transport fixtures only");
  // Establish a real origin so in-page fetches resolve relative URLs
  // through the same context routing as the application.
  await page.goto("/#/history");
  const registration = fakeApi!.addRegistration({
    repo_id: "Tongyi-MAI/Z-Image",
  });
  fakeApi!.seedRun({
    prompt: "saved images",
    registrationId: registration.id,
    completed: 2,
  });
  fakeApi!.seedRun({
    prompt: "cancelled empty",
    registrationId: registration.id,
    status: "cancelled",
  });

  const get = (query: string): Promise<{ status: number; body: ListingBody }> =>
    page.evaluate(
      async (url) => {
        const response = await fetch(url);
        return { status: response.status, body: await response.json() };
      },
      `/api/generations?${query}`,
    );
  const prompts = (body: ListingBody) => body.runs.map((run) => run.prompt);

  // An explicit cancelled status wins over the default exclusion: the
  // empty cancellation is returned, matching repository.py.
  const combined = await get("status=cancelled&exclude_empty_cancelled=true");
  expect(combined.status).toBe(200);
  expect(prompts(combined.body)).toEqual(["cancelled empty"]);

  // Without a status filter, the exclusion hides the empty cancellation.
  const library = await get("exclude_empty_cancelled=true");
  expect(library.status).toBe(200);
  expect(prompts(library.body)).toEqual(["saved images"]);

  // Omitted has_images applies no image filter; false forbids a
  // completed image.
  const omitted = await get("");
  expect(omitted.status).toBe(200);
  expect(prompts(omitted.body)).toEqual(["cancelled empty", "saved images"]);
  const withoutImages = await get("has_images=false");
  expect(withoutImages.status).toBe(200);
  expect(prompts(withoutImages.body)).toEqual(["cancelled empty"]);

  // Every FastAPI boolean spelling parses case-insensitively.
  for (const value of ["TRUE", "1", "on", "yes", "t", "y"]) {
    const spelled = await get(`has_images=${value}`);
    expect(spelled.status, `has_images=${value}`).toBe(200);
    expect(prompts(spelled.body)).toEqual(["saved images"]);
  }

  // Anything else is rejected with 422 for both boolean filters, including
  // inherited prototype keys the lookup could otherwise surface.
  const invalidHasImages = await get("has_images=maybe");
  expect(invalidHasImages.status).toBe(422);
  const inheritedKey = await get("has_images=constructor");
  expect(inheritedKey.status).toBe(422);
  const invalidExclusion = await get("exclude_empty_cancelled=2");
  expect(invalidExclusion.status).toBe(422);
});
