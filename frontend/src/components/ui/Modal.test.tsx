import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { Modal } from "./Modal";

function Harness() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open
      </button>
      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title="Confirm ejection"
        footer={
          <>
            <button type="button">Keep model</button>
            <button type="button">Eject</button>
          </>
        }
      >
        Body text
      </Modal>
    </>
  );
}

async function openDialog() {
  render(<Harness />);
  const trigger = screen.getByRole("button", { name: "Open" });
  trigger.focus();
  fireEvent.click(trigger);
  return { trigger, dialog: await screen.findByRole("dialog") };
}

describe("Modal", () => {
  it("renders nothing while closed", () => {
    render(
      <Modal open={false} onClose={vi.fn()} title="Title">
        Body
      </Modal>,
    );

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("renders as an accessible modal dialog with its title", async () => {
    const { dialog } = await openDialog();

    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText("Confirm ejection")).toBeInTheDocument();
    expect(screen.getByText("Body text")).toBeInTheDocument();
  });

  it("closes on Escape and restores focus to the trigger", async () => {
    const { trigger, dialog } = await openDialog();

    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("closes on an outside pointer press", async () => {
    await openDialog();

    const overlay = document.querySelector(".modal-overlay");
    expect(overlay).not.toBeNull();
    fireEvent.mouseDown(overlay as Element);

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("traps Tab focus inside the dialog", async () => {
    const { dialog } = await openDialog();

    const close = screen.getByRole("button", { name: "Close dialog" });
    const keep = screen.getByRole("button", { name: "Keep model" });
    const eject = screen.getByRole("button", { name: "Eject" });

    eject.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(document.activeElement).toBe(close);

    close.focus();
    fireEvent.keyDown(dialog, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(eject);

    expect(keep).toBeInTheDocument();
  });

  it("locks body scrolling while open and restores it after", async () => {
    await openDialog();

    expect(document.body.style.overflow).toBe("hidden");

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(document.body.style.overflow).toBe("");
  });

  it("gives nested dialogs distinct accessible names and ids", async () => {
    const outerClose = vi.fn();
    const innerClose = vi.fn();
    render(
      <Modal
        open
        onClose={outerClose}
        title="Run details drawer"
        variant="drawer"
      >
        <p>Drawer body</p>
        <Modal open onClose={innerClose} title="Move run to Trash?">
          <p>Confirm body</p>
        </Modal>
      </Modal>,
    );

    const dialogs = screen.getAllByRole("dialog");
    expect(dialogs).toHaveLength(2);
    const labelledBy = dialogs.map((dialog) =>
      dialog.getAttribute("aria-labelledby"),
    );
    expect(labelledBy[0]).not.toBe(labelledBy[1]);
    // Each labelledby id exists exactly once in the document.
    for (const id of labelledBy) {
      expect(document.querySelectorAll(`[id="${id}"]`)).toHaveLength(1);
    }
    expect(
      screen.getByRole("dialog", { name: "Run details drawer" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("dialog", { name: "Move run to Trash?" }),
    ).toBeInTheDocument();

    // Escape closes the topmost dialog only; locate it by name so the
    // assertion does not depend on portal mount order.
    fireEvent.keyDown(
      screen.getByRole("dialog", { name: "Move run to Trash?" }),
      { key: "Escape" },
    );
    expect(innerClose).toHaveBeenCalledOnce();
    expect(outerClose).not.toHaveBeenCalled();
  });
});
