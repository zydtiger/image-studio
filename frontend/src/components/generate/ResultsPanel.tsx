import { useState } from "react";
import { Link } from "react-router";

import { getRun, metadataUrl } from "../../api/generations";
import type { RunDetail } from "../../api/types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { ProgressBar } from "../ui/ProgressBar";
import { Spinner } from "../ui/Spinner";
import { Lightbox } from "../media/Lightbox";
import { usePolling } from "../../hooks/usePolling";
import { formatDuration, formatRelativeTime } from "../../lib/format";
import {
  isRunActive,
  runStatusLabel,
  runStatusTone,
} from "../../lib/statusTone";
import { ImageTile } from "./ImageTile";

function progressPercent(run: RunDetail): number | null {
  const { progress } = run;
  if (!progress) return null;
  const perImage =
    ((progress.image_index - 1) * progress.total_steps + progress.step) /
    (run.image_count * progress.total_steps);
  return Math.round(perImage * 100);
}

/**
 * Follows one run: polls detail at 1 Hz while the run can still change,
 * shows progress, per-image results with seeds, and clear failure and
 * partial states. All data comes from the API.
 *
 * The fetched record is keyed by run id: switching `runId` derives an
 * empty view immediately and re-keys the polling schedule (aborting the
 * in-flight request for the old run and fetching the new one at once),
 * and late responses or errors for a previous run are ignored.
 */
export function ResultsPanel({
  runId,
  onClear,
}: {
  runId: string | null;
  onClear: () => void;
}) {
  const [record, setRecord] = useState<{
    runId: string;
    run?: RunDetail;
    loadError?: unknown;
  }>();
  const [openArtifactId, setOpenArtifactId] = useState<{
    runId: string;
    artifactId: string;
  } | null>(null);

  const live = record !== undefined && record.runId === runId;
  const run = live ? record.run : undefined;
  const loadError = live ? record.loadError : undefined;
  const openArtifact =
    openArtifactId !== null && openArtifactId.runId === runId
      ? run?.images.find(
          (image) => image.artifact_id === openArtifactId.artifactId,
        )
      : undefined;

  // Poll only while the run can still change; terminal states keep the
  // last fetched detail. A runId switch makes `run` undefined again and
  // re-enables polling for the new identity.
  const pollEnabled =
    runId !== null && (run === undefined || isRunActive(run.status));

  usePolling(
    async (signal) => {
      const detail = await getRun(runId as string, signal);
      setRecord({ runId: runId as string, run: detail });
    },
    {
      activeIntervalMs: 1_000,
      backgroundIntervalMs: 5_000,
      enabled: pollEnabled,
      // Keying the schedule by run identity restarts polling on switch
      // even when the previous request is still stalled in flight.
      refreshKey: runId,
      onError: (error) =>
        setRecord({ runId: runId as string, loadError: error }),
    },
  );

  if (runId === null) {
    return (
      <section className="panel results-panel" aria-label="Run results">
        <header className="panel__header">
          <h2>Results</h2>
        </header>
        <EmptyState
          title="No run selected"
          description="Submit a generation or select a queued run to follow its progress here."
        />
      </section>
    );
  }

  if (run === undefined) {
    return (
      <section className="panel results-panel" aria-label="Run results">
        <header className="panel__header">
          <h2>Results</h2>
        </header>
        {loadError !== undefined ? (
          <EmptyState
            title="Run unavailable"
            description="The run could not be loaded."
          />
        ) : (
          <p className="panel__body">
            <Spinner label="Loading run" />
          </p>
        )}
      </section>
    );
  }

  const percent = progressPercent(run);
  const completed = run.images.filter(
    (image) => image.status === "completed",
  ).length;
  const elapsed =
    run.started_at != null
      ? formatDuration(
          Date.parse(run.finished_at ?? new Date().toISOString()) -
            Date.parse(run.started_at),
        )
      : null;

  return (
    <section className="panel results-panel" aria-label="Run results">
      <header className="panel__header">
        <h2>
          Results
          <span className="panel__subtitle">
            run {run.run_id.slice(0, 8)} ·{" "}
            {formatRelativeTime(Date.parse(run.created_at))}
          </span>
        </h2>
        <div className="panel__header-actions">
          <a
            className="button button--secondary button--sm"
            href={metadataUrl(run.run_id)}
            download
          >
            Metadata
          </a>
          <Button size="sm" variant="ghost" onClick={onClear}>
            Clear
          </Button>
        </div>
      </header>

      <div className="results-summary">
        <Badge tone={runStatusTone(run.status)}>
          {runStatusLabel(run.status)}
        </Badge>
        <span className="results-summary__model" title={run.repo_id}>
          {run.repo_id} · {run.profile}
        </span>
        <span>{run.gpu?.name ?? "GPU unknown"}</span>
        <span>
          {run.width}×{run.height} · {run.steps} steps · guidance {run.guidance}
        </span>
        <span title={`Initial seed ${run.initial_seed}`}>
          seed {run.initial_seed}
        </span>
        <span>
          {completed}/{run.image_count} images
          {elapsed !== null ? ` · ${elapsed}` : ""}
        </span>
        {run.queue_position !== null && run.queue_position !== undefined ? (
          <span>queue position {run.queue_position}</span>
        ) : null}
      </div>

      {run.status === "running" && percent !== null && run.progress ? (
        <div className="results-progress">
          <ProgressBar
            value={percent}
            label={`Run progress: image ${run.progress.image_index} of ${run.image_count}, step ${run.progress.step} of ${run.progress.total_steps}`}
          />
          <span className="results-progress__text" aria-hidden="true">
            Image {run.progress.image_index}/{run.image_count} · step{" "}
            {run.progress.step}/{run.progress.total_steps}
          </span>
        </div>
      ) : null}

      {run.error ? (
        <div className="notice notice--error" role="alert">
          <strong>{runStatusLabel(run.status)}:</strong> {run.error.message}
          {run.error.code === "worker_error" ? (
            <span className="notice__extra">
              The worker failed and was unloaded; nothing was changed silently.
            </span>
          ) : null}
        </div>
      ) : null}

      {run.images.length > 0 ? (
        <div className="results-grid">
          {run.images.map((artifact) => (
            <ImageTile
              key={artifact.artifact_id}
              runId={run.run_id}
              artifact={artifact}
              onOpen={(target) =>
                setOpenArtifactId({
                  runId: run.run_id,
                  artifactId: target.artifact_id,
                })
              }
            />
          ))}
        </div>
      ) : (
        <EmptyState
          title="No images yet"
          description="Images appear here as they complete."
        />
      )}

      <p className="results-footer">
        <Link to="/history">Open History</Link> for all runs, favorites, and
        image downloads.
      </p>

      {openArtifact ? (
        <Lightbox
          runId={run.run_id}
          artifact={openArtifact}
          imageCount={run.image_count}
          onClose={() => setOpenArtifactId(null)}
        />
      ) : null}
    </section>
  );
}
