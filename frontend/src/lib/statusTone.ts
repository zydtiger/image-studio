import type { DownloadStatus, RunStatus, WorkerState } from "../api/types";
import type { BadgeTone } from "../components/ui/Badge";

/** Display labels and badge tones for contract enums. */

const RUN_STATUS_LABELS: Record<RunStatus, string> = {
  queued: "Queued",
  paused: "Paused",
  running: "Running",
  completed: "Completed",
  partial: "Partial",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

const RUN_STATUS_TONES: Record<RunStatus, BadgeTone> = {
  queued: "neutral",
  paused: "warning",
  running: "info",
  completed: "success",
  partial: "warning",
  failed: "danger",
  cancelled: "neutral",
  interrupted: "warning",
};

export function runStatusLabel(status: RunStatus): string {
  return RUN_STATUS_LABELS[status];
}

export function runStatusTone(status: RunStatus): BadgeTone {
  return RUN_STATUS_TONES[status];
}

const WORKER_STATE_LABELS: Record<WorkerState, string> = {
  unloaded: "No model loaded",
  loading: "Loading model",
  idle: "Idle",
  generating: "Generating",
  switching: "Switching model",
  ejecting: "Ejecting",
};

const WORKER_STATE_TONES: Record<WorkerState, BadgeTone> = {
  unloaded: "neutral",
  loading: "info",
  idle: "success",
  generating: "info",
  switching: "warning",
  ejecting: "warning",
};

export function workerStateLabel(state: WorkerState): string {
  return WORKER_STATE_LABELS[state];
}

export function workerStateTone(state: WorkerState): BadgeTone {
  return WORKER_STATE_TONES[state];
}

const DOWNLOAD_STATUS_LABELS: Record<DownloadStatus, string> = {
  queued: "Queued",
  running: "Downloading",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

const DOWNLOAD_STATUS_TONES: Record<DownloadStatus, BadgeTone> = {
  queued: "neutral",
  running: "info",
  completed: "success",
  failed: "danger",
  cancelled: "neutral",
};

export function downloadStatusLabel(status: DownloadStatus): string {
  return DOWNLOAD_STATUS_LABELS[status];
}

export function downloadStatusTone(status: DownloadStatus): BadgeTone {
  return DOWNLOAD_STATUS_TONES[status];
}

/** True while a run may still change on its own; drives polling. */
export function isRunActive(status: RunStatus): boolean {
  return status === "queued" || status === "running" || status === "paused";
}

/** Short 7-character commit prefix for display. */
export function shortCommit(commitSha: string): string {
  return commitSha.slice(0, 7);
}
