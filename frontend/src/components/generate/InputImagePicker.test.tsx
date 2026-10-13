import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import type { ProfileSpec } from "../../api/types";
import { uploadInput } from "../../api/inputs";
import { InputImagePicker } from "./InputImagePicker";

vi.mock("../../api/inputs", () => ({ uploadInput: vi.fn() }));
const profile: ProfileSpec = {
  profile_id: "qwen-image-2.1",
  label: "Qwen",
  generation_modes: ["text-to-image", "image-edit"],
  max_input_images: 1,
  max_input_bytes: 20971520,
  max_input_pixels: 16777216,
  default_steps: 40,
  min_steps: 1,
  max_steps: 100,
  guidance_default: 1,
  guidance_fixed: 1,
  negative_prompt_supported: false,
  default_width: 1024,
  default_height: 1024,
  dimension_multiple: 32,
  dtype: "bfloat16",
};
const image = {
  input_id: "input",
  url: "/api/inputs/input",
  width: 512,
  height: 384,
  size_bytes: 10,
  created_at: "2026-10-13T00:00:00Z",
};

beforeEach(() => vi.mocked(uploadInput).mockReset());

it("retains the current reference when a replacement fails and allows retry", async () => {
  const changed = vi.fn(),
    busy = vi.fn();
  vi.mocked(uploadInput).mockRejectedValueOnce(
    new Error("Invalid input image"),
  );
  vi.mocked(uploadInput).mockResolvedValueOnce(image);
  render(
    <InputImagePicker
      image={image}
      profile={profile}
      disabled={false}
      onChange={changed}
      onBusyChange={busy}
    />,
  );
  const file = new File(["data"], "ref.png", { type: "image/png" });
  fireEvent.change(screen.getByLabelText("Upload input image"), {
    target: { files: [file] },
  });
  await screen.findByRole("alert");
  expect(changed).not.toHaveBeenCalled();
  expect(screen.getByAltText("Input image")).toHaveAttribute("src", image.url);
  fireEvent.change(screen.getByLabelText("Upload input image"), {
    target: { files: [file] },
  });
  await waitFor(() => expect(changed).toHaveBeenCalledWith(image));
  expect(busy.mock.calls).toEqual([[true], [false], [true], [false]]);
});

it("uploads dropped images and rejects multiple references", async () => {
  const changed = vi.fn();
  vi.mocked(uploadInput).mockResolvedValue(image);
  const { container } = render(
    <InputImagePicker
      profile={profile}
      disabled={false}
      onChange={changed}
      onBusyChange={vi.fn()}
    />,
  );
  const file = new File(["data"], "ref.png", { type: "image/png" });
  fireEvent.drop(container.firstChild!, {
    dataTransfer: { files: [file, file] },
  });
  expect(screen.getByRole("alert")).toHaveTextContent("Select one image.");
  expect(uploadInput).not.toHaveBeenCalled();
  fireEvent.drop(container.firstChild!, { dataTransfer: { files: [file] } });
  await waitFor(() => expect(changed).toHaveBeenCalledWith(image));
});
