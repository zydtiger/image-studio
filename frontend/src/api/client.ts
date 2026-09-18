/**
 * Generic fetch helper for the Image Studio API.
 *
 * Every non-2xx response uses the contract envelope
 * `{ "error": { code, message, details? } }`; this module parses that
 * envelope without assuming anything about endpoint-specific payloads.
 */

export type ApiErrorKind = "network" | "conflict" | "http";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  /** Machine-readable code from the contract's ErrorInfo, when present. */
  readonly code?: string;
  readonly details?: Record<string, unknown> | null;
  /** Parsed response body of a failed request, when present. */
  readonly body?: unknown;

  constructor(
    message: string,
    options: {
      kind: ApiErrorKind;
      status?: number;
      code?: string;
      details?: Record<string, unknown> | null;
      body?: unknown;
    } = { kind: "http" },
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = options.kind;
    this.status = options.status;
    this.code = options.code;
    this.details = options.details;
    this.body = options.body;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

export interface ApiFetchOptions extends Omit<RequestInit, "body"> {
  /** JSON-serializable request body; `undefined` sends no body. */
  body?: unknown;
  /** API base path; defaults to "/api". */
  base?: string;
}

const JSON_CONTENT_TYPE = "application/json";

/**
 * Sends a JSON request and returns the decoded JSON response. Failed HTTP
 * responses throw an {@link ApiError}; aborted requests re-raise the
 * original `AbortError` untouched.
 */
export async function apiFetch<T>(
  path: string,
  options: ApiFetchOptions = {},
): Promise<T> {
  const { body, base = "/api", ...init } = options;
  const headers = new Headers(init.headers);
  headers.set("Accept", JSON_CONTENT_TYPE);
  let requestBody: BodyInit | undefined;
  if (body !== undefined) {
    headers.set("Content-Type", JSON_CONTENT_TYPE);
    requestBody = JSON.stringify(body);
  }

  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      ...init,
      headers,
      body: requestBody,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError("Could not reach the server.", { kind: "network" });
  }

  if (!response.ok) {
    throw await toApiError(response);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  const text = await response.text();
  if (text === "") {
    return undefined as T;
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(
      `Received an invalid response (HTTP ${response.status}).`,
      { kind: "http", status: response.status },
    );
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  const text = await response.text().catch(() => "");
  let body: unknown;
  if (text !== "") {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  const parsed = parseErrorEnvelope(body);
  return new ApiError(
    parsed.message ?? `Request failed with HTTP status ${response.status}.`,
    {
      kind: response.status === 409 ? "conflict" : "http",
      status: response.status,
      code: parsed.code,
      details: parsed.details,
      body,
    },
  );
}

interface EnvelopeParts {
  message?: string;
  code?: string;
  details?: Record<string, unknown> | null;
}

/** Reads the contract envelope, with defensive fallbacks for proxies. */
function parseErrorEnvelope(body: unknown): EnvelopeParts {
  if (typeof body === "string" && body.trim() !== "") {
    return { message: body };
  }
  if (typeof body !== "object" || body === null) {
    return {};
  }
  const record = body as Record<string, unknown>;
  const error = record.error;
  if (typeof error === "object" && error !== null) {
    const info = error as Record<string, unknown>;
    return {
      message: typeof info.message === "string" ? info.message : undefined,
      code: typeof info.code === "string" ? info.code : undefined,
      details:
        typeof info.details === "object" && info.details !== null
          ? (info.details as Record<string, unknown>)
          : null,
    };
  }
  for (const key of ["detail", "message"]) {
    const candidate = record[key];
    if (typeof candidate === "string" && candidate !== "") {
      return { message: candidate };
    }
  }
  return {};
}

/** Maps any thrown value to a short human-readable message for display. */
export function errorMessage(error: unknown): string {
  if (isApiError(error)) {
    return error.message;
  }
  if (error instanceof DOMException && error.name === "AbortError") {
    return "Request cancelled.";
  }
  if (error instanceof Error && error.message !== "") {
    return error.message;
  }
  return "An unexpected error occurred.";
}
