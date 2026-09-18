import { useCallback, useRef, useState, type ReactNode } from "react";

import { ApiError } from "../../api/client";
import { ejectRuntime, getRuntime } from "../../api/runtime";
import type { RuntimeStatus } from "../../api/types";
import { usePolling } from "../../hooks/usePolling";
import { RuntimeContext, type RuntimeContextValue } from "./context";

/**
 * Global runtime state: the single resident model and its GPU, worker
 * state, queue depth, and Eject. Views register "attention" to raise the
 * polling rate to 1 Hz while they are active; otherwise the provider
 * polls slowly (and slower still while the document is hidden).
 */

const ATTENTION_INTERVAL_MS = 1_000;
const IDLE_INTERVAL_MS = 5_000;
const HIDDEN_INTERVAL_MS = 15_000;

export function RuntimeProvider({ children }: { children: ReactNode }) {
  const [runtime, setRuntime] = useState<RuntimeStatus>();
  const [error, setError] = useState<unknown>();
  const [attention, setAttentionActive] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const attentionCount = useRef(0);

  const setAttention = useCallback((active: boolean) => {
    attentionCount.current = Math.max(
      0,
      attentionCount.current + (active ? 1 : -1),
    );
    setAttentionActive(attentionCount.current > 0);
  }, []);

  const refresh = useCallback(() => {
    setRefreshKey((current) => current + 1);
  }, []);

  usePolling(
    async (signal) => {
      const status = await getRuntime(signal);
      setRuntime(status);
      setError(undefined);
    },
    {
      activeIntervalMs: attention ? ATTENTION_INTERVAL_MS : IDLE_INTERVAL_MS,
      backgroundIntervalMs: HIDDEN_INTERVAL_MS,
      onError: (cause) => setError(cause),
      refreshKey,
    },
  );

  const eject = useCallback(async () => {
    try {
      await ejectRuntime();
      setRefreshKey((current) => current + 1);
      return { ok: true } as const;
    } catch (cause) {
      if (cause instanceof ApiError) {
        setRefreshKey((current) => current + 1);
        return { ok: false, error: cause } as const;
      }
      throw cause;
    }
  }, []);

  const value: RuntimeContextValue = {
    runtime,
    error,
    connected: runtime !== undefined && error === undefined,
    refresh,
    eject,
    setAttention,
  };

  return (
    <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>
  );
}
