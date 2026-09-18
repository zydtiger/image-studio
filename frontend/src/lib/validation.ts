import { MAX_SEED } from "../api/types";

/**
 * Client-side input guards for generation parameters.
 *
 * Bounds mirror the contract constants in `src/image_studio/schemas.py`
 * (dimensions 256-2048 in multiples of 16, 1-4 images, seed range
 * 0-MAX_SEED with no wrap within a run). The backend remains
 * authoritative; these helpers give immediate feedback in forms.
 */

export const MIN_DIMENSION = 256;
export const MAX_DIMENSION = 2048;
export const DIMENSION_STEP = 16;
export const MIN_IMAGE_COUNT = 1;
export const MAX_IMAGE_COUNT = 4;
export { MAX_SEED };

export type DimensionValidation = { ok: true } | { ok: false; message: string };

export function validateDimension(
  value: number,
  label = "Dimension",
): DimensionValidation {
  if (!Number.isInteger(value)) {
    return { ok: false, message: `${label} must be a whole number.` };
  }
  if (value < MIN_DIMENSION || value > MAX_DIMENSION) {
    return {
      ok: false,
      message: `${label} must be between ${MIN_DIMENSION} and ${MAX_DIMENSION} pixels.`,
    };
  }
  if (value % DIMENSION_STEP !== 0) {
    return {
      ok: false,
      message: `${label} must be a multiple of ${DIMENSION_STEP}.`,
    };
  }
  return { ok: true };
}

export type ImageCountValidation =
  { ok: true } | { ok: false; message: string };

export function validateImageCount(value: number): ImageCountValidation {
  if (
    !Number.isInteger(value) ||
    value < MIN_IMAGE_COUNT ||
    value > MAX_IMAGE_COUNT
  ) {
    return {
      ok: false,
      message: `Image count must be a whole number between ${MIN_IMAGE_COUNT} and ${MAX_IMAGE_COUNT}.`,
    };
  }
  return { ok: true };
}

export type SeedParse =
  | { kind: "empty" }
  | { kind: "value"; value: number }
  | { kind: "invalid"; message: string };

/**
 * Parses a seed input. An empty input resolves at submission time; a
 * filled input must be a non-negative integer within the contract range.
 */
export function parseSeed(input: string): SeedParse {
  const trimmed = input.trim();
  if (trimmed === "") {
    return { kind: "empty" };
  }
  if (!/^\d+$/.test(trimmed)) {
    return {
      kind: "invalid",
      message: "Seed must be a non-negative integer.",
    };
  }
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value) || value > MAX_SEED) {
    return {
      kind: "invalid",
      message: `Seed must be at most ${MAX_SEED}.`,
    };
  }
  return { kind: "value", value };
}

/**
 * Contract rule: seeds increase by one per image and never wrap, so
 * `seed + count - 1` must stay within MAX_SEED.
 */
export function validateSeedWindow(
  seed: number,
  count: number,
): { ok: true } | { ok: false; message: string } {
  if (seed + count - 1 > MAX_SEED) {
    return {
      ok: false,
      message: `Seed ${seed} with ${count} images would exceed the maximum seed ${MAX_SEED}; seeds never wrap.`,
    };
  }
  return { ok: true };
}
