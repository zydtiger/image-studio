import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Button } from "./Button";

describe("Button", () => {
  it("fires clicks and applies variant and size classes", () => {
    const onClick = vi.fn();
    render(
      <Button variant="primary" size="sm" onClick={onClick}>
        Save
      </Button>,
    );

    const button = screen.getByRole("button", { name: "Save" });
    expect(button).toHaveClass("button", "button--primary", "button--sm");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("blocks interaction while disabled", () => {
    const onClick = vi.fn();
    render(
      <Button disabled onClick={onClick}>
        Save
      </Button>,
    );

    const button = screen.getByRole("button", { name: "Save" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("defaults to type=button so it never submits a form implicitly", () => {
    render(<Button>Plain</Button>);

    expect(screen.getByRole("button", { name: "Plain" })).toHaveAttribute(
      "type",
      "button",
    );
  });
});
