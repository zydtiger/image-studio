import { useState } from "react";

import type { RunSummary } from "../../api/types";
import { thumbnailUrl } from "../../api/generations";
import { Badge } from "../ui/Badge";
import { RunActions } from "./RunActions";
import { formatRelativeTime } from "../../lib/format";
import { runStatusTone } from "../../lib/statusTone";
import { runStatusLabel } from "../../lib/statusTone";

/**
 * One run in the history grid: preview thumbnail (when the run produced
 * images), status, model, prompt, and the favorite toggle.
 */
export function RunCard({
  run,
  onOpen,
  onToggleFavorite,
  onTrash,
}: {
  run: RunSummary;
  onOpen: () => void;
  onToggleFavorite: (favorite: boolean) => void | Promise<void>;
  onTrash?: () => void | Promise<void>;
}) {
  const [broken, setBroken] = useState(false);
  const hasPreview =
    run.preview_artifact_id !== null &&
    run.preview_artifact_id !== undefined &&
    run.completed_count > 0 &&
    !broken;

  return (
    <article className="run-card">
      <button
        type="button"
        className="run-card__preview"
        onClick={onOpen}
        aria-label={`Open run from ${formatRelativeTime(run.created_at)}: ${run.prompt.slice(0, 80)}`}
      >
        {hasPreview ? (
          <img
            src={thumbnailUrl(run.run_id, run.preview_artifact_id as string)}
            alt=""
            loading="lazy"
            onError={() => setBroken(true)}
          />
        ) : (
          <span className="run-card__no-preview">
            {run.completed_count}/{run.image_count} images
          </span>
        )}
      </button>
      <div className="run-card__body">
        <div className="run-card__head">
          <Badge tone={runStatusTone(run.status)}>
            {runStatusLabel(run.status)}
          </Badge>
          <RunActions
            run={run}
            onToggleFavorite={onToggleFavorite}
            onTrash={onTrash}
          />
        </div>
        <p className="run-card__prompt" title={run.prompt}>
          {run.prompt}
        </p>
        <p className="run-card__meta">
          <span className="run-card__repo" title={run.repo_id}>
            {run.repo_id}
          </span>
          <span>{run.profile}</span>
          <span>{formatRelativeTime(run.created_at)}</span>
        </p>
      </div>
    </article>
  );
}
