import { describe, expect, it } from "vitest";

import {
  MAX_IMAGE_COUNT,
  MIN_IMAGE_COUNT,
  parseSeed,
  validateDimension,
  validateImageCount,
  validateSeedWindow,
} from "./validation";

describe("validateDimension", () => {
  it("accepts multiples of 16 within 256-2048", () => {
    expect(validateDimension(256)).toEqual({ ok: true });
    expect(validateDimension(1024)).toEqual({ ok: true });
    expect(validateDimension(2048)).toEqual({ ok: true });
    expect(validateDimension(1232, "Width")).toEqual({ ok: true });
  });

  it("rejects values outside the range", () => {
    expect(validateDimension(240).ok).toBe(false);
    expect(validateDimension(2064).ok).toBe(false);
    expect(validateDimension(1024.5).ok).toBe(false);
  });

  it("rejects values that are not multiples of 16", () => {
    const result = validateDimension(1000, "Height");
    expect(result).toEqual({
      ok: false,
      message: "Height must be a multiple of 16.",
    });
  });
});

describe("validateImageCount", () => {
  it(`accepts whole numbers from ${MIN_IMAGE_COUNT} to ${MAX_IMAGE_COUNT}`, () => {
    expect(validateImageCount(1)).toEqual({ ok: true });
    expect(validateImageCount(4)).toEqual({ ok: true });
  });

  it("rejects out-of-range and fractional counts", () => {
    expect(validateImageCount(0).ok).toBe(false);
    expect(validateImageCount(5).ok).toBe(false);
    expect(validateImageCount(2.5).ok).toBe(false);
  });
});

describe("parseSeed", () => {
  it("treats blank input as empty so it resolves at submission", () => {
    expect(parseSeed("")).toEqual({ kind: "empty" });
    expect(parseSeed("   ")).toEqual({ kind: "empty" });
  });

  it("parses non-negative integers up to the contract MAX_SEED", () => {
    expect(parseSeed("0")).toEqual({ kind: "value", value: 0 });
    expect(parseSeed("42")).toEqual({ kind: "value", value: 42 });
    expect(parseSeed(" 4294967295 ")).toEqual({
      kind: "value",
      value: 4_294_967_295,
    });
  });

  it("rejects malformed and oversized seeds", () => {
    expect(parseSeed("abc")).toEqual({
      kind: "invalid",
      message: "Seed must be a non-negative integer.",
    });
    expect(parseSeed("-1").kind).toBe("invalid");
    expect(parseSeed("1e5").kind).toBe("invalid");
    expect(parseSeed("4294967296")).toEqual({
      kind: "invalid",
      message: "Seed must be at most 4294967295.",
    });
  });

  it("enforces the no-wrap seed window", () => {
    expect(validateSeedWindow(4_294_967_295, 1)).toEqual({ ok: true });
    expect(validateSeedWindow(4_294_967_295, 2).ok).toBe(false);
    expect(validateSeedWindow(100, 4)).toEqual({ ok: true });
  });
});
