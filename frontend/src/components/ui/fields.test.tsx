import { fireEvent, render, screen } from "@testing-library/react";
import type { ChangeEvent } from "react";
import { describe, expect, it, vi } from "vitest";

import { NumberField, SelectField, TextField } from "./fields";

describe("TextField", () => {
  it("associates the label with the input", () => {
    render(<TextField id="prompt" label="Prompt" />);

    expect(screen.getByLabelText("Prompt")).toHaveAttribute("id", "prompt");
  });

  it("wires hints and errors through aria-describedby", () => {
    render(
      <TextField
        id="prompt"
        label="Prompt"
        hint="Any characters, including emoji"
        error="Prompt is required."
      />,
    );

    const input = screen.getByLabelText("Prompt");
    expect(input).toHaveAttribute("aria-invalid", "true");
    const describedBy = input.getAttribute("aria-describedby") ?? "";
    expect(describedBy).toContain("prompt-hint");
    expect(describedBy).toContain("prompt-error");
    expect(screen.getByRole("alert")).toHaveTextContent("Prompt is required.");
  });

  it("marks the control valid when there is no error", () => {
    render(<TextField id="prompt" label="Prompt" />);

    expect(screen.getByLabelText("Prompt")).not.toHaveAttribute("aria-invalid");
  });

  it("forwards change events with unicode content", () => {
    const onChange = vi.fn();
    render(<TextField id="prompt" label="Prompt" onChange={onChange} />);

    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "a red panda, 你好, 🎨" },
    });

    expect(onChange).toHaveBeenCalledOnce();
    const event = onChange.mock.calls[0][0] as ChangeEvent<HTMLInputElement>;
    expect(event.target.value).toBe("a red panda, 你好, 🎨");
  });
});

describe("NumberField", () => {
  it("renders a number input with range attributes", () => {
    render(
      <NumberField
        id="steps"
        label="Steps"
        min={1}
        max={100}
        step={1}
        defaultValue={9}
      />,
    );

    const input = screen.getByLabelText("Steps");
    expect(input).toHaveAttribute("type", "number");
    expect(input).toHaveAttribute("min", "1");
    expect(input).toHaveAttribute("max", "100");
    expect(input).toHaveAttribute("step", "1");
    expect(input).toHaveValue(9);
  });
});

describe("SelectField", () => {
  it("renders options and fires change events", () => {
    const onChange = vi.fn();
    render(
      <SelectField id="gpu" label="GPU" onChange={onChange}>
        <option value="gpu-a">GPU A</option>
        <option value="gpu-b">GPU B</option>
      </SelectField>,
    );

    const select = screen.getByLabelText("GPU");
    expect(select).toHaveValue("gpu-a");
    fireEvent.change(select, { target: { value: "gpu-b" } });
    const event = onChange.mock.calls[0][0] as ChangeEvent<HTMLSelectElement>;
    expect(event.target.value).toBe("gpu-b");
  });
});
