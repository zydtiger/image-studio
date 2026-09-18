import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ERROR_TOAST_DURATION_MS, useToast, type ToastInput } from "./context";
import { ToastProvider } from "./ToastProvider";

function PushButton({ toast, label }: { toast: ToastInput; label: string }) {
  const { pushToast } = useToast();
  return (
    <button type="button" onClick={() => pushToast(toast)}>
      {label}
    </button>
  );
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("ToastProvider", () => {
  it("renders pushed toasts and auto-dismisses them", async () => {
    render(
      <ToastProvider>
        <PushButton
          label="Push error"
          toast={{ kind: "error", message: "Disk unreachable" }}
        />
      </ToastProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Push error" }));
    expect(screen.getByText("Disk unreachable")).toBeVisible();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ERROR_TOAST_DURATION_MS);
    });
    expect(screen.queryByText("Disk unreachable")).toBeNull();
  });

  it("keeps sticky toasts until dismissed", async () => {
    render(
      <ToastProvider>
        <PushButton
          label="Push sticky"
          toast={{ kind: "info", message: "Sticky note", durationMs: 0 }}
        />
      </ToastProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Push sticky" }));
    await vi.advanceTimersByTimeAsync(60_000);
    expect(screen.getByText("Sticky note")).toBeVisible();

    fireEvent.click(
      screen.getByRole("button", { name: "Dismiss notification" }),
    );
    expect(screen.queryByText("Sticky note")).toBeNull();
  });

  it("runs the toast action and dismisses the toast", () => {
    const onView = vi.fn();
    render(
      <ToastProvider>
        <PushButton
          label="Push action"
          toast={{
            kind: "success",
            message: "Saved",
            action: { label: "View", onClick: onView },
          }}
        />
      </ToastProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Push action" }));
    fireEvent.click(screen.getByRole("button", { name: "View" }));

    expect(onView).toHaveBeenCalledOnce();
    expect(screen.queryByText("Saved")).toBeNull();
  });

  it("announces errors assertively and info politely", () => {
    render(
      <ToastProvider>
        <PushButton
          label="Push error"
          toast={{ kind: "error", message: "Disk unreachable" }}
        />
        <PushButton
          label="Push info"
          toast={{ kind: "info", message: "Sticky note", durationMs: 0 }}
        />
      </ToastProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Push error" }));
    fireEvent.click(screen.getByRole("button", { name: "Push info" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Disk unreachable");
    expect(screen.getByRole("status")).toHaveTextContent("Sticky note");
  });
});
