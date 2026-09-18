import { useEffect, useRef } from "react";

export interface PollingOptions {
  /** Interval while the document is visible. */
  activeIntervalMs: number;
  /**
   * Interval while the document is hidden. Defaults to `activeIntervalMs`.
   */
  backgroundIntervalMs?: number;
  /** Enable or pause polling. Defaults to true. */
  enabled?: boolean;
  /** Run the first poll immediately when enabled. Defaults to true. */
  immediate?: boolean;
  /** Called when a poll fails; polling continues afterwards. */
  onError?: (error: unknown) => void;
  /**
   * Changing this value stops the current schedule and polls immediately;
   * use it to refresh after a user action (e.g. Eject).
   */
  refreshKey?: unknown;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/**
 * Repeatedly invokes `poll` with a fresh AbortSignal. The schedule adapts
 * to document visibility, aborts the in-flight request on unmount, and
 * never lets an individual failure break the loop: errors are surfaced
 * through `onError` and the next poll is still scheduled.
 */
export function usePolling(
  poll: (signal: AbortSignal) => Promise<void>,
  options: PollingOptions,
): void {
  const {
    activeIntervalMs,
    backgroundIntervalMs = activeIntervalMs,
    enabled = true,
    immediate = true,
    onError,
    refreshKey,
  } = options;

  const pollRef = useRef(poll);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    pollRef.current = poll;
    onErrorRef.current = onError;
  });

  useEffect(() => {
    if (!enabled) return;

    let disposed = false;
    let running = false;
    let controller: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const interval = () =>
      document.visibilityState === "visible"
        ? activeIntervalMs
        : backgroundIntervalMs;

    const scheduleNext = () => {
      if (disposed || running) return;
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        void run();
      }, interval());
    };

    const run = async () => {
      if (running) return;
      running = true;
      controller?.abort();
      const current = new AbortController();
      controller = current;
      try {
        await pollRef.current(current.signal);
      } catch (error) {
        // An aborted or disposed request belongs to a previous schedule;
        // reporting it through the latest onError closure would
        // misattribute an old failure to the new identity.
        if (!isAbortError(error) && !current.signal.aborted && !disposed) {
          onErrorRef.current?.(error);
        }
      } finally {
        running = false;
        if (!disposed) scheduleNext();
      }
    };

    const onVisibilityChange = () => {
      scheduleNext();
    };

    document.addEventListener("visibilitychange", onVisibilityChange);

    if (immediate) {
      void run();
    } else {
      scheduleNext();
    }

    return () => {
      disposed = true;
      if (timer !== null) clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [enabled, immediate, activeIntervalMs, backgroundIntervalMs, refreshKey]);
}
