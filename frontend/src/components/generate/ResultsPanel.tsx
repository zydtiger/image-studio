import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";

import { getRun, metadataUrl, thumbnailUrl } from "../../api/generations";
import type { ArtifactView, RunDetail, RunSummary } from "../../api/types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { ProgressBar } from "../ui/ProgressBar";
import { Spinner } from "../ui/Spinner";
import { Lightbox } from "../media/Lightbox";
import { usePolling } from "../../hooks/usePolling";
import { formatDuration, formatRelativeTime } from "../../lib/format";
import { cx } from "../../lib/cx";
import {
  isRunActive,
  runStatusLabel,
  runStatusTone,
} from "../../lib/statusTone";
import { ImageTile } from "./ImageTile";

/** Recent runs of the selected model, as fetched by the Generate page. */
export interface ModelRunsView {
  repoId: string;
  runs: RunSummary[];
  total: number;
}

function progressPercent(run: RunDetail): number | null {
  const { progress } = run;
  if (!progress) return null;
  const perImage =
    ((progress.image_index - 1) * progress.total_steps + progress.step) /
    (run.image_count * progress.total_steps);
  return Math.round(perImage * 100);
}

/**
 * One recent run of the selected model: preview thumbnail (when the run
 * produced images), status, and age. A real button inside the list item,
 * so keyboard and assistive-tech button semantics stay intact.
 */
function ModelRunChip({
  run,
  selected,
  onSelect,
}: {
  run: RunSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  const [broken, setBroken] = useState(false);
  const hasPreview =
    run.preview_artifact_id != null && run.completed_count > 0 && !broken;
  return (
    <button
      type="button"
      className={cx("model-run", selected && "model-run--selected")}
      aria-pressed={selected}
      aria-label={`View run from ${formatRelativeTime(run.created_at)}: ${run.prompt.slice(0, 80)}`}
      title={run.prompt}
      onClick={onSelect}
    >
      <span className="model-run__preview">
        {hasPreview ? (
          <img
            src={thumbnailUrl(run.run_id, run.preview_artifact_id as string)}
            alt=""
            loading="lazy"
            onError={() => setBroken(true)}
          />
        ) : (
          <span className="model-run__no-preview">
            {run.completed_count}/{run.image_count}
          </span>
        )}
      </span>
      <span className="model-run__meta">
        <Badge tone={runStatusTone(run.status)}>
          {runStatusLabel(run.status)}
        </Badge>
        <span>{formatRelativeTime(run.created_at)}</span>
      </span>
    </button>
  );
}

/**
 * Results of the selected model: a compact list of its recent runs (newest
 * first, thumbnails and status, Load more for older ones) above the followed
 * run's detail — progress, per-image results with seeds, and clear failure
 * and partial states. All data comes from the API.
 *
 * The fetched record is keyed by run id: switching `runId` derives an
 * empty view immediately and re-keys the polling schedule (aborting the
 * in-flight request for the old run and fetching the new one at once),
 * and late responses or errors for a previous run are ignored. An
 * authoritative record pushed through `runUpdate` (queue cancellation)
 * applies to the followed run promptly, and a stale in-flight read that
 * left the server before a terminal observation can no longer
 * resurrect the run as still active.
 *
 * The model-scoped run list is optional; without it the panel renders the
 * followed run only. The Generate page passes list data that is already
 * repo-guarded, so a straggler response for another model never reaches
 * this component.
 */
export function ResultsPanel({
  runId,
  onClear,
  runUpdate,
  modelRuns,
  modelRunsLoading,
  modelRunsError,
  onSelectRun,
  onRetryModelRuns,
  onLoadMoreModelRuns,
  onRunSettled,
}: {
  runId: string | null;
  onClear: () => void;
  /** Authoritative run record pushed from the queue's cancel response. */
  runUpdate?: { seq: number; run: RunDetail } | null;
  /** Recent runs of the selected model; omit for the run-only panel. */
  modelRuns?: ModelRunsView;
  modelRunsLoading?: boolean;
  modelRunsError?: unknown;
  onSelectRun?: (runId: string) => void;
  onRetryModelRuns?: () => void;
  onLoadMoreModelRuns?: () => void;
  /**
   * Fired once per follow when the followed run is observed terminal —
   * whether by transitioning from an active state or by a first read that
   * already reports a terminal status — so the page can refresh the
   * model's run list precisely when generation finishes.
   */
  onRunSettled?: (run: RunDetail) => void;
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
  const [appliedUpdateSeq, setAppliedUpdateSeq] = useState(0);
  // The latest selected run id, so a straggler read for a previous run
  // (resolved just after its abort raced with completion) can never
  // clobber the current selection's record.
  const selectedRunIdRef = useRef<string | null>(runId);
  useEffect(() => {
    selectedRunIdRef.current = runId;
  }, [runId]);

  // Apply each pushed record once, and only for the followed run: cancelling
  // another row must not change the selection's view. A delayed response
  // that still reports the run active cannot revert a terminal observation
  // — the same guard the polling path applies.
  useEffect(() => {
    if (
      runUpdate === undefined ||
      runUpdate === null ||
      runUpdate.seq <= appliedUpdateSeq ||
      runUpdate.run.run_id !== runId
    ) {
      return;
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect -- applying an externally pushed authoritative record
    setAppliedUpdateSeq(runUpdate.seq);
    setRecord((current) => {
      if (
        current?.runId === runUpdate.run.run_id &&
        current.run !== undefined &&
        !isRunActive(current.run.status) &&
        isRunActive(runUpdate.run.status)
      ) {
        return current;
      }
      return { runId: runUpdate.run.run_id, run: runUpdate.run };
    });
  }, [runUpdate, appliedUpdateSeq, runId]);

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
      if (selectedRunIdRef.current !== runId) {
        return; // a straggler read for a previously selected run
      }
      setRecord((current) => {
        // A read that left the server before a terminal observation (for
        // example the cancel response) must not resurrect the run as
        // still active; terminal states never revert.
        if (
          current?.runId === runId &&
          current.run !== undefined &&
          !isRunActive(current.run.status) &&
          isRunActive(detail.status)
        ) {
          return current;
        }
        return { runId: runId as string, run: detail };
      });
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

  // Report the followed run's terminal state exactly once per follow: a
  // run first observed already terminal (it finished before the first read
  // landed) reports just like one observed transitioning from active. The
  // watch is keyed by run id and resets when nothing is followed, so
  // re-selecting a run reports again but nothing fires repeatedly.
  const settledWatch = useRef<{ runId: string; reported: boolean } | null>(
    null,
  );
  useEffect(() => {
    if (run === undefined) {
      settledWatch.current = null;
      return;
    }
    if (onRunSettled === undefined) return;
    const watch = settledWatch.current;
    if (!isRunActive(run.status)) {
      if (watch?.runId !== run.run_id || !watch.reported) {
        onRunSettled(run);
      }
      settledWatch.current = { runId: run.run_id, reported: true };
    } else {
      settledWatch.current = { runId: run.run_id, reported: false };
    }
  }, [run, onRunSettled]);

  const modelScoped =
    modelRuns !== undefined ||
    modelRunsLoading === true ||
    modelRunsError !== undefined;
  const hasSelectableRuns = (modelRuns?.runs.length ?? 0) > 0;

  return (
    <section className="panel results-panel" aria-label="Run results">
      <header className="panel__header">
        <h2>
          Results
          {modelRuns !== undefined ? (
            <span className="panel__subtitle" title={modelRuns.repoId}>
              {modelRuns.repoId}
            </span>
          ) : null}
        </h2>
        {runId !== null ? (
          <div className="panel__header-actions">
            <a
              className="button button--secondary button--sm"
              href={metadataUrl(runId)}
              download
            >
              Metadata
            </a>
            <Button size="sm" variant="ghost" onClick={onClear}>
              Clear
            </Button>
          </div>
        ) : null}
      </header>

      {modelScoped ? (
        <div className="model-runs">
          {modelRunsError !== undefined ? (
            <div className="model-runs__error" role="alert">
              <p className="notice notice--error">
                Runs of this model could not be loaded.
              </p>
              {onRetryModelRuns !== undefined ? (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={onRetryModelRuns}
                >
                  Retry
                </Button>
              ) : null}
            </div>
          ) : modelRuns === undefined ? (
            <p className="model-runs__status">
              <Spinner label="Loading runs" />
            </p>
          ) : modelRuns.runs.length === 0 ? (
            <EmptyState
              title="No runs yet"
              description={`Runs generated with ${modelRuns.repoId} appear here.`}
            />
          ) : (
            <>
              <ul
                className="model-runs__grid"
                aria-label="Recent runs of this model"
              >
                {modelRuns.runs.map((summary) => (
                  <li key={summary.run_id} className="model-runs__item">
                    <ModelRunChip
                      run={summary}
                      selected={summary.run_id === runId}
                      onSelect={() => onSelectRun?.(summary.run_id)}
                    />
                  </li>
                ))}
              </ul>
              {modelRuns.runs.length < modelRuns.total &&
              onLoadMoreModelRuns !== undefined ? (
                <Button
                  size="sm"
                  variant="secondary"
                  className="model-runs__more"
                  onClick={onLoadMoreModelRuns}
                >
                  Load more ({modelRuns.runs.length} of {modelRuns.total})
                </Button>
              ) : null}
            </>
          )}
        </div>
      ) : null}

      {runId === null ? (
        hasSelectableRuns ? (
          <EmptyState
            title="No run selected"
            description="Pick a run from the list above, or submit a generation to follow it here."
          />
        ) : modelScoped ? null : (
          <EmptyState
            title="No run selected"
            description="Submit a generation or select a queued run to follow its progress here."
          />
        )
      ) : run === undefined ? (
        loadError !== undefined ? (
          <EmptyState
            title="Run unavailable"
            description="The run could not be loaded."
          />
        ) : (
          <p className="panel__body">
            <Spinner label="Loading run" />
          </p>
        )
      ) : (
        <RunDetailBody
          run={run}
          openArtifact={openArtifact}
          onOpenArtifact={(artifactId) =>
            setOpenArtifactId({ runId: run.run_id, artifactId })
          }
          onCloseArtifact={() => setOpenArtifactId(null)}
        />
      )}
    </section>
  );
}

/** The followed run's detail: summary, progress, images, and lightbox. */
function RunDetailBody({
  run,
  openArtifact,
  onOpenArtifact,
  onCloseArtifact,
}: {
  run: RunDetail;
  openArtifact: ArtifactView | undefined;
  onOpenArtifact: (artifactId: string) => void;
  onCloseArtifact: () => void;
}) {
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
    <>
      <div className="results-summary">
        <Badge tone={runStatusTone(run.status)}>
          {runStatusLabel(run.status)}
        </Badge>
        <span className="results-summary__run">
          run {run.run_id.slice(0, 8)}
        </span>
        <span>{formatRelativeTime(Date.parse(run.created_at))}</span>
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
              onOpen={(target) => onOpenArtifact(target.artifact_id)}
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

      {openArtifact !== undefined ? (
        <Lightbox
          runId={run.run_id}
          artifact={openArtifact}
          imageCount={run.image_count}
          onClose={onCloseArtifact}
        />
      ) : null}
    </>
  );
}
