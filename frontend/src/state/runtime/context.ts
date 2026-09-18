import { createContext, useContext } from "react";

import type { RuntimeStatus } from "../../api/types";
import type { ApiError } from "../../api/client";

export interface EjectOutcome {
  ok: boolean;
  /** Present when the server refused the ejection (409 conflict). */
  error?: ApiError;
}

export interface RuntimeContextValue {
  runtime: RuntimeStatus | undefined;
  /** Last polling error; undefined while responses arrive. */
  error: unknown;
  connected: boolean;
  /** Force an immediate refresh (e.g. after actions elsewhere). */
  refresh: () => void;
  eject: () => Promise<EjectOutcome>;
  /** Raises the polling rate while any view is actively watching. */
  setAttention: (active: boolean) => void;
}

export const RuntimeContext = createContext<RuntimeContextValue | null>(null);

export function useRuntime(): RuntimeContextValue {
  const context = useContext(RuntimeContext);
  if (context === null) {
    throw new Error("useRuntime must be used within a RuntimeProvider.");
  }
  return context;
}
