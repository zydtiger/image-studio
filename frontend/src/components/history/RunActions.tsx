import { useState } from "react";

import type { RunSummary } from "../../api/types";
import { isRunActive } from "../../lib/statusTone";
import { ConfirmDialog } from "../ui/ConfirmDialog";
import { StarIcon, TrashIcon } from "../ui/icons";

export function RunActions({
  run,
  onToggleFavorite,
  onTrash,
}: {
  run: Pick<RunSummary, "favorite" | "trashed" | "status" | "image_count">;
  onToggleFavorite: (favorite: boolean) => void | Promise<void>;
  onTrash?: () => void | Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [confirmTrash, setConfirmTrash] = useState(false);
  const act = async (action: () => void | Promise<void>) => {
    if (busy) return;
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="run-actions" role="group" aria-label="Run actions">
      <button
        type="button"
        className={`icon-button icon-button--sm${run.favorite ? " run-actions__favorite--on" : ""}`}
        aria-pressed={run.favorite}
        aria-disabled={busy}
        aria-label={run.favorite ? "Remove favorite" : "Mark favorite"}
        title={run.favorite ? "Remove favorite" : "Mark favorite"}
        onClick={() => void act(() => onToggleFavorite(!run.favorite))}
      >
        <StarIcon filled={run.favorite} />
      </button>
      {onTrash !== undefined && !run.trashed ? (
        <button
          type="button"
          className="icon-button icon-button--sm run-actions__trash"
          aria-label="Move run to Trash"
          title={
            isRunActive(run.status)
              ? "Finish or cancel this run before moving it to Trash"
              : "Move run to Trash"
          }
          disabled={isRunActive(run.status)}
          aria-disabled={busy}
          onClick={() => {
            if (!busy) setConfirmTrash(true);
          }}
        >
          <TrashIcon />
        </button>
      ) : null}
      <ConfirmDialog
        open={confirmTrash}
        onClose={() => setConfirmTrash(false)}
        onConfirm={() => {
          if (onTrash !== undefined) void act(onTrash);
        }}
        title="Move run to Trash?"
        confirmLabel="Move to Trash"
        tone="danger"
      >
        This moves the entire run ({run.image_count}{" "}
        {run.image_count === 1 ? "image" : "images"}) to recoverable Trash.
        Restore it from History; nothing is permanently deleted.
      </ConfirmDialog>
    </div>
  );
}
