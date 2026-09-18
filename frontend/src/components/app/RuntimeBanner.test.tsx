import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/client";
import type { RuntimeStatus } from "../../api/types";

const runtimeApi = vi.hoisted(() => ({
  getRuntime: vi.fn(),
  ejectRuntime: vi.fn(),
}));

vi.mock("../../api/runtime", () => runtimeApi);

import { RuntimeBanner } from "./RuntimeBanner";
import { RuntimeProvider } from "../../state/runtime/RuntimeProvider";
import { ToastProvider } from "../../state/toast/ToastProvider";

function status(overrides: Partial<RuntimeStatus>): RuntimeStatus {
  return {
    implementation: "real",
    state: "unloaded",
    resident: null,
    current_run_id: null,
    queue_depth: 0,
    last_error: null,
    ...overrides,
  };
}

function renderBanner(next: RuntimeStatus) {
  runtimeApi.getRuntime.mockResolvedValue(next);
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <RuntimeBanner />
      </RuntimeProvider>
    </ToastProvider>,
  );
}

beforeEach(() => {
  runtimeApi.getRuntime.mockReset();
  runtimeApi.ejectRuntime.mockReset();
});

describe("RuntimeBanner", () => {
  it("stays silent while the first response is pending, then reports failure", async () => {
    let rejectFirst!: (reason: unknown) => void;
    runtimeApi.getRuntime.mockImplementationOnce(
      () =>
        new Promise<RuntimeStatus>((_resolve, reject) => {
          rejectFirst = reject;
        }),
    );
    runtimeApi.getRuntime.mockResolvedValue(status({}));

    render(
      <ToastProvider>
        <RuntimeProvider>
          <RuntimeBanner />
        </RuntimeProvider>
      </ToastProvider>,
    );

    // Pending first response: connecting, never an unreachable alert.
    expect(await screen.findByText(/Connecting to the server/)).toBeVisible();
    expect(screen.queryByText(/Server unreachable/)).toBeNull();

    rejectFirst(new Error("connection refused"));

    expect(await screen.findByText(/Server unreachable/)).toBeInTheDocument();
    expect(screen.queryByText(/Connecting to the server/)).toBeNull();
  });

  it("shows the resident model and its GPU after the run clears", async () => {
    renderBanner(
      status({
        state: "idle",
        current_run_id: null,
        resident: {
          registration_id: "reg-1",
          repo_id: "Tongyi-MAI/Z-Image",
          commit_sha: "abcdef1234567890",
          profile: "z-image",
          dtype: "bfloat16",
          gpu: { uuid: "gpu-0", name: "RTX A" },
        },
      }),
    );

    await waitFor(() =>
      expect(screen.getByText("Tongyi-MAI/Z-Image")).toBeInTheDocument(),
    );
    expect(screen.getByText("RTX A")).toBeInTheDocument();
    expect(screen.getByText("Idle")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Eject model" })).toBeEnabled();
  });

  it("disables Eject while the worker is busy and explains why", async () => {
    renderBanner(
      status({
        state: "generating",
        current_run_id: "run-1234",
        resident: {
          registration_id: "reg-1",
          repo_id: "Tongyi-MAI/Z-Image",
          commit_sha: "abcdef1234567890",
          profile: "z-image",
          dtype: "bfloat16",
          gpu: { uuid: "gpu-0", name: "RTX A" },
        },
      }),
    );

    const eject = await screen.findByRole("button", { name: "Eject model" });
    expect(eject).toBeDisabled();
    expect(eject).toHaveAttribute(
      "title",
      "Eject is unavailable while the worker is generating.",
    );
    expect(screen.getByText("Generating")).toBeInTheDocument();
  });

  it("states when no model is loaded", async () => {
    renderBanner(status({ state: "unloaded" }));

    expect(await screen.findByText("No model loaded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Eject model" })).toBeDisabled();
  });

  it("marks a fake runtime visibly", async () => {
    renderBanner(status({ state: "unloaded", implementation: "fake" }));

    expect(
      await screen.findByText("Fake runtime (development)"),
    ).toBeInTheDocument();
  });

  it("ejects after confirmation and reports success", async () => {
    runtimeApi.ejectRuntime.mockResolvedValue(undefined);
    renderBanner(
      status({
        state: "idle",
        resident: {
          registration_id: "reg-1",
          repo_id: "Tongyi-MAI/Z-Image",
          commit_sha: "abcdef1234567890",
          profile: "z-image",
          dtype: "bfloat16",
          gpu: { uuid: "gpu-0", name: "RTX A" },
        },
      }),
    );

    fireEvent.click(await screen.findByRole("button", { name: "Eject model" }));
    fireEvent.click(await screen.findByRole("button", { name: /^Eject$/ }));

    await waitFor(() => expect(runtimeApi.ejectRuntime).toHaveBeenCalledOnce());
    expect(
      await screen.findByText(/Files and registrations are untouched/i),
    ).toBeInTheDocument();
  });

  it("surfaces a 409 conflict from the server", async () => {
    runtimeApi.ejectRuntime.mockRejectedValue(
      new ApiError("Worker is busy; eject while idle only.", {
        kind: "conflict",
        status: 409,
        code: "conflict",
      }),
    );
    renderBanner(
      status({
        state: "idle",
        resident: {
          registration_id: "reg-1",
          repo_id: "Tongyi-MAI/Z-Image",
          commit_sha: "abcdef1234567890",
          profile: "z-image",
          dtype: "bfloat16",
          gpu: { uuid: "gpu-0", name: "RTX A" },
        },
      }),
    );

    fireEvent.click(await screen.findByRole("button", { name: "Eject model" }));
    fireEvent.click(await screen.findByRole("button", { name: /^Eject$/ }));

    expect(
      await screen.findByText(/eject while idle only/i),
    ).toBeInTheDocument();
  });
});
