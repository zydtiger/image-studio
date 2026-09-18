import { useState } from "react";

import type { ArtifactView } from "../../api/types";
import { thumbnailUrl } from "../../api/generations";
import { Badge } from "../ui/Badge";

/**
 * One image tile from a run: completed images show their lazily generated
 * thumbnail (with a graceful fallback when bytes are missing), other
 * statuses explain themselves.
 */
export function ImageTile({
  runId,
  artifact,
  onOpen,
}: {
  runId: string;
  artifact: ArtifactView;
  onOpen?: (artifact: ArtifactView) => void;
}) {
  const [broken, setBroken] = useState(false);
  const openable =
    artifact.status === "completed" &&
    artifact.url !== null &&
    artifact.url !== undefined;

  return (
    <figure className="image-tile" data-status={artifact.status}>
      {artifact.status === "completed" && !broken ? (
        <button
          type="button"
          className="image-tile__button"
          onClick={() => onOpen?.(artifact)}
          disabled={!openable || onOpen === undefined}
          aria-label={`Open image ${artifact.index}, seed ${artifact.seed}`}
        >
          <img
            src={thumbnailUrl(runId, artifact.artifact_id)}
            alt={`Image ${artifact.index} of the run (seed ${artifact.seed})`}
            loading="lazy"
            onError={() => setBroken(true)}
          />
        </button>
      ) : (
        <div
          className="image-tile__placeholder"
          role="img"
          aria-label={`Image ${artifact.index}: ${artifact.status}`}
        >
          {artifact.status === "completed" ? (
            <>Image unavailable</>
          ) : artifact.status === "failed" ? (
            (artifact.error ?? "Failed")
          ) : artifact.status === "cancelled" ? (
            "Cancelled"
          ) : (
            "Pending"
          )}
        </div>
      )}
      <figcaption className="image-tile__meta">
        <span className="image-tile__index">#{artifact.index}</span>
        <span className="image-tile__seed" title={`Seed ${artifact.seed}`}>
          seed {artifact.seed}
        </span>
        {artifact.status !== "completed" ? (
          <Badge tone={artifact.status === "failed" ? "danger" : "neutral"}>
            {artifact.status}
          </Badge>
        ) : null}
      </figcaption>
    </figure>
  );
}
