import { useRef, useState } from "react";

import {
  cancelRun,
  getQueue,
  getRun,
  resumeQueue,
} from "../../api/generations";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { Spinner } from "../ui/Spinner";
import { useApiQuery } from "../../hooks/useApiQuery";
import { usePolling } from "../../hooks/usePolling";
import { useRuntime } from "../../state/runtime/context";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import type { QueueState, RunDetail } from "../../api/types";
import {
  isRunActive,
  runStatusLabel,
  runStatusTone,
} from "../../lib/statusTone";

/**
 * Global FIFO generation queue: paused-state banner with resume, the
 * active run with cooperative cancellation, and pending runs that cancel
 * immediately. Task cancellation is separate from Eject.
 */
export function QueuePanel({
  onFocusRun,
  onRunCancelled,
}: {
  onFocusRun: (runId: string) => void;
  /** Receives every authoritative cancel response, terminal or not. */
  onRunCancelled?: (run: RunDetail) => void;
}) {
  const { runtime } = useRuntime();
  const toast = useToast();
  const [queue, setQueue] = useState<QueueState>();
  const [queueError, setQueueError] = useState<unknown>();
  const [refreshKey, setRefreshKey] = useState(0);
  // Runs the server confirmed terminal through a cancel response. Reads
  // that left the server before that confirmation must not resurrect them
  // (terminal runs can never re-enter the queue or be current again), so a
  // stale runtime current_run_id keeps the active row hidden too. The ref
  // is read when a response resolves (stale-read filtering); the mirrored
  // state drives render-time visibility.
  const terminalRunIds = useRef<Set<string>>(new Set());
  const [activeHiddenRunIds, setActiveHiddenRunIds] = useState<Set<string>>(
    () => new Set(),
  );
  const currentRunId = runtime?.current_run_id ?? null;
  const activeVisible =
    currentRunId !== null && !activeHiddenRunIds.has(currentRunId);

  usePolling(
    async (signal) => {
      const state = await getQueue(signal);
      const hidden = terminalRunIds.current;
      setQueue(
        hidden.size === 0
          ? state
          : {
              ...state,
              pending: state.pending.filter((run) => !hidden.has(run.run_id)),
            },
      );
      setQueueError(undefined);
    },
    {
      activeIntervalMs: 1_000,
      backgroundIntervalMs: 5_000,
      onError: (error) => setQueueError(error),
      refreshKey,
    },
  );

  const activeRunQuery = useApiQuery(
    (signal) => getRun(currentRunId as string, signal),
    [currentRunId],
    { enabled: currentRunId !== null },
  );

  const onRunAction = async (
    action: () => Promise<unknown>,
    message: string,
  ) => {
    try {
      await action();
      setRefreshKey((current) => current + 1);
      toast.pushToast({ kind: "info", message });
    } catch (error) {
      toast.pushToast({ kind: "error", message: errorMessage(error) });
    }
  };

  const onCancelRun = async (
    runId: string,
    context: { active: boolean; paused: boolean },
  ) => {
    let detail: RunDetail;
    try {
      detail = await cancelRun(runId);
    } catch (error) {
      toast.pushToast({ kind: "error", message: errorMessage(error) });
      return;
    }
    // The response is authoritative: apply it directly instead of waiting
    // for the next queue read, and hand it to the followed-run Results.
    if (!isRunActive(detail.status)) {
      terminalRunIds.current.add(detail.run_id);
      setActiveHiddenRunIds((current) => new Set(current).add(detail.run_id));
      setQueue((current) =>
        current === undefined
          ? current
          : {
              ...current,
              pending: current.pending.filter(
                (run) => run.run_id !== detail.run_id,
              ),
            },
      );
    }
    setRefreshKey((current) => current + 1);
    onRunCancelled?.(detail);
    toast.pushToast({
      kind: "info",
      // Only a "cancelled" response may claim cancellation; a cooperative
      // cancel that raced with natural completion reports the run already
      // finished in its actual terminal state.
      message:
        detail.status === "cancelled"
          ? context.paused
            ? "Paused run cancelled."
            : context.active
              ? "Run cancelled."
              : "Run removed from the queue."
          : isRunActive(detail.status)
            ? "Cancellation requested at the next safe boundary."
            : `Run already finished (${runStatusLabel(detail.status).toLowerCase()}).`,
    });
  };

  const pending = queue?.pending ?? [];
  const active = activeRunQuery.data;

  return (
    <section className="panel queue-panel" aria-label="Generation queue">
      <header className="panel__header">
        <h2>Queue</h2>
        {queue?.paused ? (
          <Button
            size="sm"
            variant="primary"
            onClick={() =>
              void onRunAction(
                () => resumeQueue(),
                "Queue resumed in FIFO order.",
              )
            }
          >
            Resume queue
          </Button>
        ) : null}
      </header>

      {queue?.paused ? (
        <p className="notice notice--warning" role="status">
          The queue is paused after a server restart. Resume to continue the
          paused runs in their original order.
        </p>
      ) : null}

      {queueError !== undefined ? (
        queue === undefined ? (
          <p className="notice notice--error" role="alert">
            Queue status is unavailable. Retrying in the background.
          </p>
        ) : (
          <p className="notice notice--warning" role="status">
            Showing the last known queue ({errorMessage(queueError)}). Retrying
            in the background.
          </p>
        )
      ) : null}

      {!activeVisible && pending.length === 0 ? (
        queue === undefined && queueError === undefined ? (
          <p className="panel__body">
            <Spinner label="Loading queue" />
          </p>
        ) : queue !== undefined ? (
          <EmptyState
            title="Queue is empty"
            description="Submitted runs appear here with their progress."
          />
        ) : null
      ) : (
        <ul className="queue-list">
          {activeVisible ? (
            <li className="queue-item queue-item--active">
              <div className="queue-item__main">
                <span className="queue-item__title">
                  {active ? active.prompt : `Run ${currentRunId.slice(0, 8)}`}
                </span>
                <span className="queue-item__meta">
                  {active
                    ? `${active.repo_id} · ${active.gpu?.name ?? ""}`
                    : ""}
                </span>
              </div>
              <Badge tone="info">Running</Badge>
              <div className="queue-item__actions">
                <Button size="sm" onClick={() => onFocusRun(currentRunId)}>
                  View
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() =>
                    void onCancelRun(currentRunId, {
                      active: true,
                      paused: false,
                    })
                  }
                >
                  Cancel
                </Button>
              </div>
            </li>
          ) : null}
          {pending.map((run, index) => (
            <li key={run.run_id} className="queue-item">
              <div className="queue-item__main">
                <span className="queue-item__title" title={run.prompt}>
                  {run.prompt}
                </span>
                <span className="queue-item__meta">
                  #{index + 1} · {run.repo_id}
                </span>
              </div>
              <Badge tone={runStatusTone(run.status)}>
                {runStatusLabel(run.status)}
              </Badge>
              <div className="queue-item__actions">
                <Button size="sm" onClick={() => onFocusRun(run.run_id)}>
                  View
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() =>
                    void onCancelRun(run.run_id, {
                      active: false,
                      paused: run.status === "paused",
                    })
                  }
                >
                  Cancel
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
