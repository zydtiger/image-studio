import { useState } from "react";

import { searchHubModels } from "../../api/hub";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { ErrorState } from "../ui/ErrorState";
import { TextField } from "../ui/fields";
import { useApiQuery } from "../../hooks/useApiQuery";
import { formatRelativeTime } from "../../lib/format";
import { ModelDetailDialog } from "./ModelDetailDialog";

/** Hugging Face search with model cards and access markers. */
export function DiscoverPanel({
  onDownloaded,
  onRegistered,
}: {
  onDownloaded: () => void;
  onRegistered: () => void;
}) {
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [openRepo, setOpenRepo] = useState<string | null>(null);

  const resultsQuery = useApiQuery(
    (signal) => searchHubModels(query, 20, signal),
    [query],
    { enabled: query.trim() !== "" },
  );
  const results = resultsQuery.data ?? [];

  return (
    <div className="discover">
      <form
        className="discover__search"
        onSubmit={(event) => {
          event.preventDefault();
          setQuery(queryInput.trim());
        }}
      >
        <TextField
          id="hub-search"
          label="Search Hugging Face"
          placeholder="e.g. Z-Image"
          value={queryInput}
          onChange={(event) => setQueryInput(event.target.value)}
          wrapperClassName="discover__input"
        />
        <Button type="submit" variant="primary">
          Search
        </Button>
      </form>

      {query === "" ? (
        <EmptyState
          title="Search the Hub"
          description="Find model repositories, check compatibility, then download or register a cached snapshot."
        />
      ) : resultsQuery.loading && results.length === 0 ? (
        <p className="field-hint">Searching…</p>
      ) : resultsQuery.error !== undefined ? (
        <ErrorState
          title="Search failed"
          message="The Hub could not be reached or rejected the search."
          onRetry={() => resultsQuery.refetch()}
        />
      ) : results.length === 0 ? (
        <EmptyState
          title="No results"
          description={`Nothing matched “${query}”.`}
        />
      ) : (
        <ul className="model-list">
          {results.map((model) => (
            <li key={model.repo_id} className="model-card">
              <div className="model-card__main">
                <button
                  type="button"
                  className="model-card__repo"
                  onClick={() => setOpenRepo(model.repo_id)}
                  title={model.repo_id}
                >
                  {model.repo_id}
                </button>
                <div className="model-card__meta">
                  {model.author ? <span>{model.author}</span> : null}
                  {model.gated ? <Badge tone="warning">Gated</Badge> : null}
                  {model.private ? <Badge tone="neutral">Private</Badge> : null}
                  {model.license ? <span>{model.license}</span> : null}
                  {model.downloads !== null && model.downloads !== undefined ? (
                    <span>
                      {model.downloads.toLocaleString("en")} downloads
                    </span>
                  ) : null}
                  {model.likes !== null && model.likes !== undefined ? (
                    <span>{model.likes.toLocaleString("en")} likes</span>
                  ) : null}
                  {model.last_modified ? (
                    <span>{formatRelativeTime(model.last_modified)}</span>
                  ) : null}
                </div>
              </div>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => setOpenRepo(model.repo_id)}
              >
                Details
              </Button>
            </li>
          ))}
        </ul>
      )}

      {openRepo !== null ? (
        <ModelDetailDialog
          repoId={openRepo}
          onClose={() => setOpenRepo(null)}
          onDownloaded={() => {
            setOpenRepo(null);
            onDownloaded();
          }}
          onRegistered={() => {
            setOpenRepo(null);
            onRegistered();
          }}
        />
      ) : null}
    </div>
  );
}
