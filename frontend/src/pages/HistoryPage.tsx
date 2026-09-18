import { useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { listRegistrations } from "../api/models";
import { listRuns, setFavorite, type RunListParams } from "../api/generations";
import type { RunStatus } from "../api/types";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { SelectField, TextField } from "../components/ui/fields";
import { RunCard } from "../components/history/RunCard";
import { RunDetailDrawer } from "../components/history/RunDetailDrawer";
import { useApiQuery } from "../hooks/useApiQuery";
import { useToast } from "../state/toast/context";
import { errorMessage } from "../api/client";

const PAGE_SIZE = 24;

const STATUS_OPTIONS: RunStatus[] = [
  "queued",
  "paused",
  "running",
  "completed",
  "partial",
  "failed",
  "cancelled",
  "interrupted",
];

type ViewMode = "library" | "trash";

/**
 * Run history with prompt search, filters, favorites, parameter reuse,
 * and the recoverable Trash view. Failed, cancelled, partial, and
 * interrupted runs are all retained.
 */
export default function HistoryPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const [view, setView] = useState<ViewMode>("library");
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<"" | RunStatus>("");
  const [model, setModel] = useState("");
  const [favoriteOnly, setFavoriteOnly] = useState(false);
  const [page, setPage] = useState(0);
  const [openRunId, setOpenRunId] = useState<string | null>(null);

  const registrationsQuery = useApiQuery(listRegistrations, []);
  const modelOptions = useMemo(() => {
    const repoIds = new Set(
      (registrationsQuery.data ?? []).map((entry) => entry.repo_id),
    );
    return Array.from(repoIds).sort();
  }, [registrationsQuery.data]);

  const params: RunListParams = {
    q: query === "" ? undefined : query,
    status: status === "" ? undefined : status,
    model: model === "" ? undefined : model,
    favorite: favoriteOnly ? true : undefined,
    trashed: view === "trash" ? "only" : undefined,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  const runsQuery = useApiQuery(
    (signal) => listRuns(params, signal),
    [view, query, status, model, favoriteOnly, page],
  );
  const runs = runsQuery.data?.runs ?? [];
  const total = runsQuery.data?.total ?? 0;
  const hasPrev = page > 0;
  const hasNext = (page + 1) * PAGE_SIZE < total;

  const toggleFavorite = async (runId: string, favorite: boolean) => {
    try {
      await setFavorite(runId, favorite);
      runsQuery.refetch();
    } catch (error) {
      toast.pushToast({ kind: "error", message: errorMessage(error) });
    }
  };

  return (
    <section className="page page--history">
      <header className="page-header">
        <h1>History</h1>
        <p>
          Every run with its parameters, results, favorites, downloads, and
          recoverable Trash.
        </p>
      </header>

      <div className="history-toolbar">
        <div className="segmented" role="group" aria-label="History view">
          <button
            type="button"
            className={
              view === "library"
                ? "segmented__item segmented__item--on"
                : "segmented__item"
            }
            aria-pressed={view === "library"}
            onClick={() => {
              setView("library");
              setPage(0);
            }}
          >
            Library
          </button>
          <button
            type="button"
            className={
              view === "trash"
                ? "segmented__item segmented__item--on"
                : "segmented__item"
            }
            aria-pressed={view === "trash"}
            onClick={() => {
              setView("trash");
              setPage(0);
            }}
          >
            Trash
          </button>
        </div>

        <form
          className="history-filters"
          onSubmit={(event) => {
            event.preventDefault();
            setQuery(queryInput.trim());
            setPage(0);
          }}
        >
          <TextField
            id="history-search"
            label="Search prompts"
            value={queryInput}
            placeholder="Search…"
            onChange={(event) => setQueryInput(event.target.value)}
            wrapperClassName="history-filters__search"
          />
          <SelectField
            id="history-status"
            label="Status"
            value={status}
            onChange={(event) => {
              setStatus(event.target.value as "" | RunStatus);
              setPage(0);
            }}
          >
            <option value="">Any status</option>
            {STATUS_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </SelectField>
          <SelectField
            id="history-model"
            label="Model"
            value={model}
            onChange={(event) => {
              setModel(event.target.value);
              setPage(0);
            }}
          >
            <option value="">Any model</option>
            {modelOptions.map((repoId) => (
              <option key={repoId} value={repoId}>
                {repoId}
              </option>
            ))}
          </SelectField>
          <label className="history-filters__favorite">
            <input
              type="checkbox"
              checked={favoriteOnly}
              onChange={(event) => {
                setFavoriteOnly(event.target.checked);
                setPage(0);
              }}
            />
            Favorites only
          </label>
          <Button type="submit" variant="secondary">
            Search
          </Button>
        </form>
      </div>

      {runsQuery.error !== undefined ? (
        <ErrorState
          title="History unavailable"
          message="Runs could not be loaded."
          onRetry={() => runsQuery.refetch()}
        />
      ) : runsQuery.loading && runs.length === 0 ? (
        <p className="field-hint">Loading runs…</p>
      ) : runs.length === 0 ? (
        <EmptyState
          title={view === "trash" ? "Trash is empty" : "No runs"}
          description={
            view === "trash"
              ? "Runs moved to Trash appear here and can be restored."
              : "Generation runs appear here after submission."
          }
        />
      ) : (
        <>
          <div className="history-grid">
            {runs.map((run) => (
              <RunCard
                key={run.run_id}
                run={run}
                onOpen={() => setOpenRunId(run.run_id)}
                onToggleFavorite={(favorite) =>
                  void toggleFavorite(run.run_id, favorite)
                }
              />
            ))}
          </div>
          <div className="history-pagination">
            <Button
              size="sm"
              disabled={!hasPrev}
              onClick={() => setPage((current) => Math.max(0, current - 1))}
            >
              Previous
            </Button>
            <span className="field-hint">
              {total === 0
                ? "No runs"
                : `${page * PAGE_SIZE + 1}–${Math.min(
                    (page + 1) * PAGE_SIZE,
                    total,
                  )} of ${total}`}
            </span>
            <Button
              size="sm"
              disabled={!hasNext}
              onClick={() => setPage((current) => current + 1)}
            >
              Next
            </Button>
          </div>
        </>
      )}

      {openRunId !== null ? (
        <RunDetailDrawer
          runId={openRunId}
          onClose={() => setOpenRunId(null)}
          onChanged={() => runsQuery.refetch()}
          onReuse={(payload) => {
            setOpenRunId(null);
            navigate("/generate", { state: { reuse: payload } });
          }}
        />
      ) : null}
    </section>
  );
}
