import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SystemInfo } from "../api/types";

const systemApi = vi.hoisted(() => ({ getSystem: vi.fn() }));
const runtimeApi = vi.hoisted(() => ({
  getRuntime: vi.fn(),
  ejectRuntime: vi.fn(),
}));

vi.mock("../api/system", () => systemApi);
vi.mock("../api/runtime", () => runtimeApi);

import SettingsPage from "./SettingsPage";
import { ToastProvider } from "../state/toast/ToastProvider";
import { RuntimeProvider } from "../state/runtime/RuntimeProvider";

const SYSTEM: SystemInfo = {
  host: "127.0.0.1",
  port: 7860,
  paths: {
    config_file: "/home/user/.config/image-studio/config.toml",
    data_dir: "/home/user/.local/share/image-studio",
    database_file: "/home/user/.local/share/image-studio/app.sqlite",
    outputs_dir: "/home/user/.local/share/image-studio/outputs",
    trash_dir: "/home/user/.local/share/image-studio/trash",
    thumbnails_dir: "/home/user/.cache/image-studio/thumbnails",
    log_file: "/home/user/.local/state/image-studio/logs/app.log",
    hub_cache_dir: "/home/user/.cache/huggingface/hub",
  },
  hf_logged_in: true,
  hf_username: "someone",
  gpus: [
    {
      uuid: "gpu-0",
      name: "NVIDIA RTX A",
      index: 0,
      memory_total_bytes: 26_000_000_000,
    },
  ],
  development: { fake_runtime: false, fake_hub: true },
};

function renderSettings() {
  systemApi.getSystem.mockResolvedValue(SYSTEM);
  runtimeApi.getRuntime.mockResolvedValue({
    implementation: "fake",
    state: "idle",
    resident: {
      registration_id: "reg-1",
      repo_id: "Tongyi-MAI/Z-Image",
      commit_sha: "abcdef123456",
      profile: "z-image",
      dtype: "bfloat16",
      gpu: { uuid: "gpu-0", name: "NVIDIA RTX A" },
    },
    current_run_id: null,
    queue_depth: 0,
    last_error: null,
  });
  return render(
    <ToastProvider>
      <RuntimeProvider>
        <SettingsPage />
      </RuntimeProvider>
    </ToastProvider>,
  );
}

beforeEach(() => {
  systemApi.getSystem.mockReset();
  runtimeApi.getRuntime.mockReset();
});

describe("SettingsPage", () => {
  it("shows effective paths, login, GPUs, and development flags", async () => {
    renderSettings();

    expect(await screen.findByText("Logged in as someone")).toBeInTheDocument();
    expect(
      screen.getByText("/home/user/.config/image-studio/config.toml"),
    ).toBeInTheDocument();
    // The GPU list and the resident-model section both name the device.
    expect(screen.getAllByText("NVIDIA RTX A").length).toBeGreaterThan(0);
    expect(screen.getByText("fake hub")).toBeInTheDocument();
    expect(screen.getByText(/require a restart/i)).toBeInTheDocument();
  });

  it("reports the resident model and its GPU", async () => {
    renderSettings();

    // The banner and resident sections render compound text; match parts.
    await waitFor(() =>
      expect(screen.getAllByText(/Tongyi-MAI\/Z-Image/).length).toBeGreaterThan(
        0,
      ),
    );
    expect(screen.getAllByText("NVIDIA RTX A").length).toBeGreaterThan(0);
    expect(screen.getByText("Idle")).toBeInTheDocument();
  });
});
