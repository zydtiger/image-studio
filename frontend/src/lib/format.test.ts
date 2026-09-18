import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatDuration,
  formatRelativeTime,
  formatTimestamp,
} from "./format";

describe("formatBytes", () => {
  it("formats byte, kilobyte, and megabyte ranges", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1_536_000)).toBe("1.5 MB");
    expect(formatBytes(153_600_000)).toBe("154 MB");
    expect(formatBytes(1_500_000_000)).toBe("1.5 GB");
  });

  it("rejects invalid input", () => {
    expect(formatBytes(Number.NaN)).toBe("—");
    expect(formatBytes(-5)).toBe("—");
  });
});

describe("formatDuration", () => {
  it("formats milliseconds, seconds, minutes, and hours", () => {
    expect(formatDuration(412)).toBe("412 ms");
    expect(formatDuration(3_400)).toBe("3.4 s");
    expect(formatDuration(30_000)).toBe("30 s");
    expect(formatDuration(65_000)).toBe("1 m 05 s");
    expect(formatDuration(3_600_000)).toBe("1 h 00 m");
    expect(formatDuration(7_325_000)).toBe("2 h 02 m");
  });

  it("rejects invalid input", () => {
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe("—");
    expect(formatDuration(-1)).toBe("—");
  });
});

describe("formatRelativeTime", () => {
  const now = Date.parse("2026-09-16T12:00:00Z");
  const SECOND = 1_000;
  const MINUTE = 60 * SECOND;
  const HOUR = 60 * MINUTE;
  const DAY = 24 * HOUR;

  it("reads 'just now' within ten seconds", () => {
    expect(formatRelativeTime(now + 5_000, now)).toBe("just now");
    expect(formatRelativeTime(now - 9_000, now)).toBe("just now");
  });

  it("formats past moments in natural units", () => {
    expect(formatRelativeTime(now - 30 * SECOND, now)).toBe("30 seconds ago");
    expect(formatRelativeTime(now - 5 * MINUTE, now)).toBe("5 minutes ago");
    expect(formatRelativeTime(now - 3 * HOUR, now)).toBe("3 hours ago");
    expect(formatRelativeTime(now - 2 * DAY, now)).toBe("2 days ago");
    expect(formatRelativeTime(now - 21 * DAY, now)).toBe("3 weeks ago");
    expect(formatRelativeTime(now - 60 * DAY, now)).toBe("2 months ago");
    // numeric: "auto" renders single units as "last ..." instead of "1 ...".
    expect(formatRelativeTime(now - 400 * DAY, now)).toBe("last year");
    expect(formatRelativeTime(now - 750 * DAY, now)).toBe("2 years ago");
  });

  it("formats future moments", () => {
    expect(formatRelativeTime(now + 10 * MINUTE, now)).toBe("in 10 minutes");
  });
});

describe("formatTimestamp", () => {
  it("formats valid timestamps as local date and time", () => {
    const stamp = new Date(2026, 8, 16, 14, 5).getTime();
    expect(formatTimestamp(stamp)).toContain("2026");
  });

  it("rejects invalid timestamps", () => {
    expect(formatTimestamp(Number.NaN)).toBe("—");
  });
});
