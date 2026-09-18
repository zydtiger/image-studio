/** Formatting helpers shared by generation, download, and history views. */

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

/** Formats a byte count with decimal SI units, e.g. "512 B", "1.5 MB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "—";
  }
  if (bytes < 1000) {
    return `${Math.round(bytes)} B`;
  }
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < BYTE_UNITS.length - 1) {
    value /= 1000;
    unit += 1;
  }
  const rounded = value >= 100 ? Math.round(value) : Number(value.toFixed(1));
  return `${rounded} ${BYTE_UNITS[unit]}`;
}

/** Formats a duration in milliseconds, e.g. "412 ms", "3.5 s", "1 m 05 s". */
export function formatDuration(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) {
    return "—";
  }
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)} ms`;
  }
  const seconds = milliseconds / 1000;
  if (seconds < 60) {
    return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  if (minutes < 60) {
    return `${minutes} m ${String(remainder).padStart(2, "0")} s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${String(minutes % 60).padStart(2, "0")} m`;
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;
const MONTH = 30.44 * DAY;
const YEAR = 365.25 * DAY;

const relativeFormatter = new Intl.RelativeTimeFormat("en", {
  numeric: "auto",
});

const RELATIVE_UNITS: ReadonlyArray<
  readonly [Intl.RelativeTimeFormatUnit, number]
> = [
  ["year", YEAR],
  ["month", MONTH],
  ["week", WEEK],
  ["day", DAY],
  ["hour", HOUR],
  ["minute", MINUTE],
];

/** Accepts epoch milliseconds, Date objects, or ISO 8601 strings. */
export type TimestampInput = number | Date | string;

function toMillis(timestamp: TimestampInput): number {
  if (typeof timestamp === "number") return timestamp;
  if (typeof timestamp === "string") return Date.parse(timestamp);
  return timestamp.getTime();
}

/**
 * Formats a timestamp relative to `now`, e.g. "just now", "5 minutes ago",
 * "3 weeks ago". Timestamps within ten seconds of `now` read "just now".
 */
export function formatRelativeTime(
  timestamp: TimestampInput,
  now: TimestampInput = Date.now(),
): string {
  const time = toMillis(timestamp);
  const nowMs = toMillis(now);
  if (!Number.isFinite(time) || !Number.isFinite(nowMs)) {
    return "—";
  }
  const delta = time - nowMs;
  if (Math.abs(delta) < 10_000) {
    return "just now";
  }
  for (const [unit, unitMs] of RELATIVE_UNITS) {
    if (Math.abs(delta) >= unitMs) {
      return relativeFormatter.format(Math.round(delta / unitMs), unit);
    }
  }
  return relativeFormatter.format(Math.round(delta / 1000), "second");
}

const dateTimeFormatter = new Intl.DateTimeFormat("en", {
  dateStyle: "medium",
  timeStyle: "short",
});

/** Formats a timestamp as an absolute local date and time. */
export function formatTimestamp(timestamp: TimestampInput): string {
  const time = toMillis(timestamp);
  if (!Number.isFinite(time)) {
    return "—";
  }
  return dateTimeFormatter.format(new Date(time));
}
