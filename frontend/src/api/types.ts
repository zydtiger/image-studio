/**
 * HTTP contract types, hand-mirrored from `src/image_studio/schemas.py`
 * (the authoritative contract). Shapes here must never diverge from the
 * Python schemas; timestamps are timezone-aware ISO 8601 strings.
 */

export const MAX_SEED = 4294967295;

export type ProfileId = "z-image" | "z-image-turbo";

export type WorkerState =
  "unloaded" | "loading" | "idle" | "generating" | "switching" | "ejecting";

export type RunStatus =
  | "queued"
  | "paused"
  | "running"
  | "completed"
  | "partial"
  | "failed"
  | "cancelled"
  | "interrupted";

export type ImageStatus = "pending" | "completed" | "failed" | "cancelled";

export type DownloadStatus =
  "queued" | "running" | "completed" | "failed" | "cancelled";

export type RegistrationStatus = "ready" | "missing_files";

export type ErrorCode =
  | "validation"
  | "not_found"
  | "conflict"
  | "gated_model"
  | "hub_auth_required"
  | "hub_unreachable"
  | "revision_not_found"
  | "cache_incomplete"
  | "unsupported_model"
  | "worker_error"
  | "storage_error"
  | "internal";

export interface ErrorInfo {
  code: ErrorCode;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface ProfileSpec {
  profile_id: ProfileId;
  label: string;
  default_steps: number;
  min_steps: number;
  max_steps: number;
  guidance_default: number;
  /** null means the user may adjust guidance; a value locks and hides it. */
  guidance_fixed: number | null;
  negative_prompt_supported: boolean;
  default_width: number;
  default_height: number;
  dtype: string;
}

export interface GenerationRequest {
  registration_id: string;
  gpu_uuid: string;
  prompt: string;
  negative_prompt?: string | null;
  width: number;
  height: number;
  steps?: number | null;
  guidance?: number | null;
  seed?: number | null;
  count: number;
}

export interface FrozenGpu {
  uuid: string;
  name: string;
}

export interface ResidentModel {
  registration_id: string;
  repo_id: string;
  commit_sha: string;
  profile: ProfileId;
  dtype: string;
  /** Required: keeps the resident GPU knowable after current_run_id clears. */
  gpu: FrozenGpu;
  /** Optional backend additions; absent on older builds. */
  pipeline_class?: string | null;
  dependency_versions?: Record<string, string>;
}

export interface RuntimeStatus {
  implementation: "real" | "fake";
  state: WorkerState;
  resident: ResidentModel | null;
  current_run_id: string | null;
  queue_depth: number;
  last_error: ErrorInfo | null;
}

export interface GpuInfo {
  uuid: string;
  name: string;
  index: number;
  memory_total_bytes?: number | null;
}

export interface HubModelSummary {
  repo_id: string;
  author?: string | null;
  private?: boolean;
  gated?: boolean;
  downloads?: number | null;
  likes?: number | null;
  last_modified?: string | null;
  pipeline_tag?: string | null;
  license?: string | null;
}

export interface HubFileEntry {
  path: string;
  size?: number | null;
}

export interface HubRevision {
  revision: string;
  commit_sha: string;
}

export interface HubModelDetail extends HubModelSummary {
  default_revision?: string | null;
  revisions: HubRevision[];
  files: HubFileEntry[];
}

export interface CompatibilityReport {
  repo_id: string;
  revision: string;
  commit_sha?: string | null;
  structurally_compatible: boolean;
  selectable_profiles: ProfileId[];
  findings: string[];
  notes: string[];
}

export interface CachedSnapshot {
  commit_sha?: string | null;
  path: string;
  size_bytes?: number | null;
  incomplete: boolean;
}

export interface CachedRepo {
  repo_id: string;
  size_on_disk_bytes?: number | null;
  refs: string[];
  snapshots: CachedSnapshot[];
}

export interface ModelRegistration {
  id: string;
  repo_id: string;
  commit_sha: string;
  profile: ProfileId;
  display_name?: string | null;
  status: RegistrationStatus;
  missing_files: string[];
  snapshot_path?: string | null;
  created_at: string;
  last_used_at?: string | null;
}

export interface RegistrationCreate {
  repo_id: string;
  revision?: string | null;
  profile: ProfileId;
  display_name?: string | null;
}

export interface RegistrationUpdate {
  display_name?: string | null;
  profile?: ProfileId | null;
}

export interface DownloadCreate {
  repo_id: string;
  revision?: string | null;
  profile: ProfileId;
}

export interface DownloadProgress {
  bytes_done: number;
  bytes_total?: number | null;
  files_done: number;
  files_total?: number | null;
}

export interface DownloadJob {
  id: string;
  repo_id: string;
  requested_revision?: string | null;
  resolved_commit?: string | null;
  profile: ProfileId;
  status: DownloadStatus;
  error?: ErrorInfo | null;
  progress: DownloadProgress;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface ArtifactView {
  artifact_id: string;
  index: number;
  seed: number;
  status: ImageStatus;
  width?: number | null;
  height?: number | null;
  size_bytes?: number | null;
  error?: string | null;
  url?: string | null;
  thumbnail_url?: string | null;
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  status: RunStatus;
  favorite: boolean;
  trashed: boolean;
  prompt: string;
  negative_prompt?: string | null;
  repo_id: string;
  profile: ProfileId;
  image_count: number;
  completed_count: number;
  preview_artifact_id?: string | null;
}

export interface RunProgressSnapshot {
  image_index: number;
  step: number;
  total_steps: number;
}

export interface RunDetail {
  run_id: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  status: RunStatus;
  favorite: boolean;
  trashed: boolean;
  registration_id: string;
  repo_id: string;
  commit_sha: string;
  profile: ProfileId;
  dtype: string;
  gpu?: FrozenGpu | null;
  prompt: string;
  negative_prompt?: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  initial_seed: number;
  image_count: number;
  pipeline_class?: string | null;
  dependency_versions: Record<string, string>;
  runtime_meta: Record<string, unknown>;
  queue_position?: number | null;
  progress?: RunProgressSnapshot | null;
  error?: ErrorInfo | null;
  images: ArtifactView[];
}

export interface FavoriteUpdate {
  favorite: boolean;
}

export interface QueueState {
  paused: boolean;
  pending: RunSummary[];
}

export interface SystemPaths {
  config_file: string;
  data_dir: string;
  database_file: string;
  outputs_dir: string;
  trash_dir: string;
  thumbnails_dir: string;
  log_file: string;
  hub_cache_dir: string;
}

export interface DevelopmentFlags {
  fake_runtime: boolean;
  fake_hub: boolean;
}

export interface SystemInfo {
  host: string;
  port: number;
  paths: SystemPaths;
  hf_logged_in: boolean;
  hf_username?: string | null;
  gpus: GpuInfo[];
  development: DevelopmentFlags;
}
