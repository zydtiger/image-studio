import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useApiQuery } from "./useApiQuery";

describe("useApiQuery", () => {
  it("resolves data and clears loading", async () => {
    const query = vi.fn(async () => ({ value: 3 }));

    const { result } = renderHook(() => useApiQuery(query, []));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual({ value: 3 });
    expect(result.current.error).toBeUndefined();
  });

  it("captures errors without throwing", async () => {
    const failure = new Error("offline");
    const query = vi.fn(async () => {
      throw failure;
    });

    const { result } = renderHook(() => useApiQuery(query, []));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe(failure);
    expect(result.current.data).toBeUndefined();
  });

  it("skips fetching while disabled", async () => {
    const query = vi.fn(async () => "x");

    const { result } = renderHook(() =>
      useApiQuery(query, [], { enabled: false }),
    );

    expect(result.current.loading).toBe(false);
    expect(query).not.toHaveBeenCalled();
  });

  it("refetches on demand and when deps change", async () => {
    const query = vi.fn(async () => 1);
    const { result, rerender } = renderHook(
      ({ dep }: { dep: string }) => useApiQuery(query, [dep]),
      { initialProps: { dep: "a" } },
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    act(() => result.current.refetch());
    await waitFor(() => expect(query).toHaveBeenCalledTimes(2));

    rerender({ dep: "b" });
    await waitFor(() => expect(query).toHaveBeenCalledTimes(3));
  });

  it("aborts the in-flight request on unmount", async () => {
    let seen: AbortSignal | undefined;
    const query = vi.fn(
      (signal: AbortSignal) =>
        new Promise<number>(() => {
          seen = signal;
        }),
    );

    const { unmount } = renderHook(() => useApiQuery(query, []));

    expect(seen).toBeDefined();
    unmount();
    expect(seen?.aborted).toBe(true);
  });

  it("keeps data during an opted-in refetch but clears it when dependencies change", async () => {
    let resolveRefresh!: (value: string) => void;
    const query = vi
      .fn()
      .mockResolvedValueOnce("original")
      .mockImplementationOnce(
        () =>
          new Promise<string>((resolve) => {
            resolveRefresh = resolve;
          }),
      )
      .mockResolvedValueOnce("filtered");
    const { result, rerender } = renderHook(
      ({ filter }: { filter: string }) =>
        useApiQuery(query, [filter], { keepDataOnRefetch: true }),
      { initialProps: { filter: "all" } },
    );

    await waitFor(() => expect(result.current.data).toBe("original"));
    act(() => result.current.refetch());
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBe("original");
    await act(async () => resolveRefresh("updated"));
    expect(result.current.data).toBe("updated");

    rerender({ filter: "favorites" });
    expect(result.current.data).toBeUndefined();
    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.data).toBe("filtered"));
  });
});
