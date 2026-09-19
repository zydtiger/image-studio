import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiFetch, errorMessage, isApiError } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("apiFetch", () => {
  it("decodes a JSON response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ value: 7 })),
    );

    const result = await apiFetch<{ value: number }>("/ping");

    expect(result).toEqual({ value: 7 });
  });

  it("returns undefined for 204 responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(null, { status: 204 })),
    );

    await expect(apiFetch("/done")).resolves.toBeUndefined();
  });

  it("sends a JSON body with a JSON content type", async () => {
    const fetchMock = vi.fn(async (_url: string, _init: RequestInit) =>
      jsonResponse({ ok: true }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/tasks", { method: "POST", body: { prompt: "你好" } });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/tasks");
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ prompt: "你好" }));
    expect(new Headers(init.headers).get("content-type")).toBe(
      "application/json",
    );
  });

  it("parses the contract error envelope on 409 conflicts", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(
          {
            error: {
              code: "conflict",
              message: "Worker is busy",
              details: { state: "generating" },
            },
          },
          409,
        ),
      ),
    );

    const error: unknown = await apiFetch("/runtime/eject", {
      method: "POST",
    }).catch((cause: unknown) => cause);

    expect(isApiError(error)).toBe(true);
    if (isApiError(error)) {
      expect(error.kind).toBe("conflict");
      expect(error.status).toBe(409);
      expect(error.code).toBe("conflict");
      expect(error.message).toBe("Worker is busy");
      expect(error.details).toEqual({ state: "generating" });
    }
  });

  it("still reads legacy detail strings defensively", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ detail: "Legacy body" }, 400)),
    );

    await expect(apiFetch("/x")).rejects.toMatchObject({
      kind: "http",
      message: "Legacy body",
    });
  });

  it("falls back to a generic message for opaque error bodies", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ foo: 1 }, 500)),
    );

    await expect(apiFetch("/x")).rejects.toMatchObject({
      kind: "http",
      status: 500,
      message: "Request failed with HTTP status 500.",
    });
  });

  it("uses plain-text error bodies as the message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("Server Error", { status: 502 })),
    );

    await expect(apiFetch("/x")).rejects.toMatchObject({
      kind: "http",
      message: "Server Error",
    });
  });

  it("wraps network failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("fetch failed");
      }),
    );

    await expect(apiFetch("/x")).rejects.toMatchObject({
      kind: "network",
      message: "Could not reach the server.",
    });
  });

  it("rethrows abort errors untouched", async () => {
    const abortError = new DOMException(
      "The user aborted a request.",
      "AbortError",
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw abortError;
      }),
    );

    await expect(apiFetch("/x")).rejects.toBe(abortError);
  });

  it("rejects non-JSON success bodies", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response("<html></html>", {
            status: 200,
            headers: { "content-type": "text/html" },
          }),
      ),
    );

    await expect(apiFetch("/x")).rejects.toMatchObject({
      kind: "http",
      message: expect.stringContaining("invalid response"),
    });
  });
});

describe("errorMessage", () => {
  it("uses ApiError messages, abort names, and error messages", () => {
    expect(errorMessage(new ApiError("Nope", { kind: "http" }))).toBe("Nope");
    expect(errorMessage(new DOMException("x", "AbortError"))).toBe(
      "Request cancelled.",
    );
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage("strange")).toBe("An unexpected error occurred.");
  });
});

it("includes component validation details without exposing unrelated fields", () => {
  const error = new ApiError("Cached model is incomplete", {
    details: {
      problems: ["shared/vae/config.json is missing", null, "", 42],
      snapshot_path: "/cache/internal",
    },
    kind: "http",
    code: "cache_incomplete",
  });
  expect(errorMessage(error)).toBe(
    "Cached model is incomplete: shared/vae/config.json is missing",
  );
  expect(
    errorMessage(
      new ApiError("Failed", { kind: "http", details: { problems: {} } }),
    ),
  ).toBe("Failed");
});
