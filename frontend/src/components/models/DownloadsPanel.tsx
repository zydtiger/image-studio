import { useState } from "react";

import {
  cancelDownload,
  listDownloads,
  retryDownload,
} from "../../api/downloads";
import type { DownloadJob } from "../../api/types";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { EmptyState } from "../ui/EmptyState";
import { ProgressBar } from "../ui/ProgressBar";
import { usePolling } from "../../hooks/usePolling";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import { downloadStatusLabel, downloadStatusTone } from "../../lib/statusTone";
import { formatBytes, formatRelativeTime } from "../../lib/format";

function jobActive(job: DownloadJob): boolean {
  return job.status === "queued" || job.status === "running";
}

function DownloadRow({
  job,
  onAction,
}: {
  job: DownloadJob;
  onAction: (action: () => Promise<unknown>, message: string) => void;
}) {
  const percent =
    job.progress.bytes_total !== null &&
    job.progress.bytes_total !== undefined &&
    job.progress.bytes_total > 0
      ? Math.round((job.progress.bytes_done / job.progress.bytes_total) * 100)
      : null;

  return (
    <li className="download-row">
      <div className="download-row__head">
        <span className="download-row__repo" title={job.repo_id}>
          {job.repo_id}
        </span>
        <Badge tone="info">{job.profile}</Badge>
        <Badge tone={downloadStatusTone(job.status)}>
          {downloadStatusLabel(job.status)}
        </Badge>
        <span className="field-hint">{formatRelativeTime(job.created_at)}</span>
      </div>
      {jobActive(job) ? (
        <div className="download-row__progress">
          <ProgressBar value={percent} label={`Downloading ${job.repo_id}`} />
          <span className="field-hint">
            {formatBytes(job.progress.bytes_done)}
            {job.progress.bytes_total !== null &&
            job.progress.bytes_total !== undefined
              ? ` of ${formatBytes(job.progress.bytes_total)}`
              : ""}
            {job.progress.files_total !== null &&
            job.progress.files_total !== undefined
              ? ` · ${job.progress.files_done}/${job.progress.files_total} files`
              : ""}
          </span>
        </div>
      ) : null}
      {job.error ? (
        <p className="notice notice--error" role="alert">
          {job.error.message}
        </p>
      ) : null}
      {job.status === "completed" ? (
        <p className="field-hint">
          Completed. Register the snapshot from Local Cache to use it.
        </p>
      ) : null}
      <div className="download-row__actions">
        {job.status === "failed" || job.status === "cancelled" ? (
          <Button
            size="sm"
            onClick={() =>
              onAction(
                () => retryDownload(job.id),
                "Download retry queued; cached files are reused.",
              )
            }
          >
            Retry
          </Button>
        ) : null}
        {job.status === "queued" ? (
          <Button
            size="sm"
            variant="secondary"
            onClick={() =>
              onAction(() => cancelDownload(job.id), "Download cancelled.")
            }
          >
            Cancel
          </Button>
        ) : null}
      </div>
    </li>
  );
}

/**
 * The separate single-task download queue with measured progress,
 * errors, retry, and queued-job cancellation.
 */
export function DownloadsPanel() {
  const toast = useToast();
  const [jobs, setJobs] = useState<DownloadJob[]>();
  const [refreshKey, setRefreshKey] = useState(0);

  usePolling(
    async (signal) => {
      const result = await listDownloads(signal);
      setJobs(result);
    },
    {
      activeIntervalMs: 1_000,
      backgroundIntervalMs: 5_000,
      onError: () => undefined,
      refreshKey,
    },
  );

  const onAction = async (action: () => Promise<unknown>, message: string) => {
    try {
      await action();
      setRefreshKey((current) => current + 1);
      toast.pushToast({ kind: "info", message });
    } catch (error) {
      toast.pushToast({ kind: "error", message: errorMessage(error) });
    }
  };

  if (jobs === undefined) {
    return <p className="field-hint">Loading downloads…</p>;
  }

  if (jobs.length === 0) {
    return (
      <EmptyState
        title="No downloads"
        description="Downloads queued from Discover appear here with their progress."
      />
    );
  }

  return (
    <ul className="download-list">
      {jobs.map((job) => (
        <DownloadRow key={job.id} job={job} onAction={onAction} />
      ))}
    </ul>
  );
}
