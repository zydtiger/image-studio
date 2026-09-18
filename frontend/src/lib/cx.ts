/** Joins class name parts, skipping false, null, and undefined values. */
export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
