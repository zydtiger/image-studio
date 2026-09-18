import { useCallback, useEffect, useRef, useState } from "react";

export interface ApiQueryResult<T> {
  data: T | undefined;
  loading: boolean;
  error: unknown;
  refetch: () => void;
}

interface RunRecord<T> {
  /** Token of the request run this record belongs to. */
  token: string;
  result?: T;
  error?: unknown;
}

/**
 * Minimal data-fetching hook with abort support. The query function
 * receives an AbortSignal and re-runs when `deps` change or `refetch` is
 * called. In-flight requests are aborted on re-run and on unmount.
 *
 * `deps` is serialized into a run token, so it must contain only
 * JSON-stable values (strings, numbers, booleans, null, plain arrays and
 * objects of those).
 */
export function useApiQuery<T>(
  query: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  options: { enabled?: boolean } = {},
): ApiQueryResult<T> {
  const { enabled = true } = options;
  const [version, setVersion] = useState(0);
  const [record, setRecord] = useState<RunRecord<T>>({ token: "" });
  const queryRef = useRef(query);

  useEffect(() => {
    queryRef.current = query;
  });

  const token = `${version}\u0000${JSON.stringify(deps)}`;
  const stale = record.token !== token;

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let active = true;
    queryRef.current(controller.signal).then(
      (result) => {
        if (active) setRecord({ token, result });
      },
      (cause: unknown) => {
        if (active && !controller.signal.aborted) {
          setRecord({ token, error: cause });
        }
      },
    );
    return () => {
      active = false;
      controller.abort();
    };
  }, [enabled, token]);

  const refetch = useCallback(() => {
    setVersion((current) => current + 1);
  }, []);

  return {
    data: stale ? undefined : record.result,
    // While a fresh run is in flight the last record is stale; show the
    // loading state instead of stale data (no stale-while-revalidate).
    loading: enabled && stale,
    error: stale ? undefined : record.error,
    refetch,
  };
}
