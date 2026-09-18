import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePolling } from "./usePolling";

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function setVisibility(visible: boolean) {
  vi.spyOn(document, "visibilityState", "get").mockReturnValue(
    visible ? "visible" : "hidden",
  );
  document.dispatchEvent(new Event("visibilitychange"));
}

describe("usePolling", () => {
  it("polls immediately, then on the active interval", async () => {
    const poll = vi.fn(async () => {});

    renderHook(() => usePolling(poll, { activeIntervalMs: 1000 }));

    expect(poll).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(poll).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(2000);
    expect(poll).toHaveBeenCalledTimes(4);
  });

  it("uses the background interval while the document is hidden", async () => {
    const poll = vi.fn(async () => {});

    renderHook(() =>
      usePolling(poll, {
        activeIntervalMs: 1000,
        backgroundIntervalMs: 5000,
      }),
    );

    setVisibility(false);
    await vi.advanceTimersByTimeAsync(1000);
    expect(poll).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(4000);
    expect(poll).toHaveBeenCalledTimes(2);
  });

  it("stops and restarts with the enabled flag", async () => {
    const poll = vi.fn(async () => {});
    const { rerender, unmount } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        usePolling(poll, { activeIntervalMs: 1000, enabled }),
      { initialProps: { enabled: true } },
    );

    rerender({ enabled: false });
    await vi.advanceTimersByTimeAsync(5000);
    expect(poll).toHaveBeenCalledTimes(1);

    rerender({ enabled: true });
    expect(poll).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1000);
    expect(poll).toHaveBeenCalledTimes(3);

    unmount();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(poll).toHaveBeenCalledTimes(3);
  });

  it("reports failures through onError and keeps polling", async () => {
    const failure = new Error("boom");
    const onError = vi.fn();
    const poll = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(failure)
      .mockResolvedValue(undefined);

    renderHook(() => usePolling(poll, { activeIntervalMs: 1000, onError }));

    await vi.advanceTimersByTimeAsync(0);
    expect(onError).toHaveBeenCalledWith(failure);
    await vi.advanceTimersByTimeAsync(1000);
    expect(poll).toHaveBeenCalledTimes(2);
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("skips the immediate call when immediate is false", async () => {
    const poll = vi.fn(async () => {});

    renderHook(() =>
      usePolling(poll, { activeIntervalMs: 1000, immediate: false }),
    );

    expect(poll).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1000);
    expect(poll).toHaveBeenCalledTimes(1);
  });

  it("aborts the in-flight signal on unmount", async () => {
    let seen: AbortSignal | undefined;
    const poll = (signal: AbortSignal) =>
      new Promise<void>((resolve) => {
        seen = signal;
        setTimeout(resolve, 50);
      });

    const { unmount } = renderHook(() =>
      usePolling(poll, { activeIntervalMs: 1000 }),
    );

    expect(seen).toBeDefined();
    unmount();
    expect(seen?.aborted).toBe(true);
  });
});
