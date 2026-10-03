import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router";

import { getProfiles } from "../api/profiles";
import { listRegistrations } from "../api/models";
import { getSystem } from "../api/system";
import { listRuns, submitGeneration } from "../api/generations";
import type {
  GenerationRequest,
  GpuInfo,
  RunDetail,
  RunSummary,
} from "../api/types";
import { isApiError } from "../api/client";
import { ErrorState } from "../components/ui/ErrorState";
import { GenerationForm } from "../components/generate/GenerationForm";
import { QueuePanel } from "../components/generate/QueuePanel";
import { ResultsPanel } from "../components/generate/ResultsPanel";
import { useApiQuery } from "../hooks/useApiQuery";
import type { ReusePayload } from "../lib/reuse";
import { useRuntime } from "../state/runtime/context";
import {
  applyProfileDefaults,
  applyReuse,
  buildRequest,
  validateForm,
  type FormErrors,
  type GenerationFormState,
} from "../components/generate/formState";
import { EMPTY_FORM } from "../components/generate/formState";

const EMPTY_GPU_LIST: GpuInfo[] = [];

/** Runs shown per page in the model results list; Load more appends pages. */
const MODEL_RUNS_PAGE_SIZE = 8;

/** API ceiling for limit; a refresh re-reads loaded depth in these chunks. */
const MODEL_RUNS_REFRESH_MAX = 200;

const MODEL_PREFERENCE_KEY = "image-studio.generate.registration-id";

/**
 * The selected model is the only persisted Generate-page preference, so a
 * reload restores the browsed results. Blocked or unavailable storage is
 * non-fatal: reads fall back to the default model and writes are skipped.
 * Run records, images, and prompts always come from the API.
 */
function readStoredRegistrationId(): string | null {
  try {
    const raw = window.localStorage.getItem(MODEL_PREFERENCE_KEY);
    return typeof raw === "string" && raw !== "" ? raw : null;
  } catch {
    return null;
  }
}

function storeRegistrationId(id: string): void {
  try {
    window.localStorage.setItem(MODEL_PREFERENCE_KEY, id);
  } catch {
    // Preferences are best-effort; generation itself never needs them.
  }
}

/**
 * Core flow: choose model and GPU, write a prompt, generate, follow the
 * result. Form edits never load a model; submission freezes the settings.
 * The Results panel also browses the selected model's recent runs from the
 * API, so past generations survive a reload; only the model choice itself
 * is persisted locally.
 */
export default function GeneratePage() {
  const { runtime, setAttention } = useRuntime();
  const location = useLocation();
  const reuse = (location.state as { reuse?: ReusePayload } | null)?.reuse;

  const [form, setForm] = useState<GenerationFormState>(EMPTY_FORM);
  const [errors, setErrors] = useState<FormErrors>({});
  const [attempted, setAttempted] = useState(false);
  const [submitError, setSubmitError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const [followedRunId, setFollowedRunId] = useState<string | null>(null);
  // Authoritative cancel responses relayed from the queue to the followed
  // run's Results view; `seq` makes each response apply exactly once.
  const [runUpdate, setRunUpdate] = useState<{
    seq: number;
    run: RunDetail;
  } | null>(null);
  const runUpdateSeq = useRef(0);
  const [notices, setNotices] = useState<string[]>([]);
  const [gpuNeedsChoice, setGpuNeedsChoice] = useState(false);
  const reuseApplied = useRef(false);
  // Set once the user (or a resolved reuse payload) fixes the GPU; later
  // runtime arrivals must not override that choice.
  const userChoseGpu = useRef(false);
  // Read once per mount: the registration selected in the previous visit.
  const [storedRegistrationId] = useState(() => readStoredRegistrationId());

  // Results browsing is scoped to the selected model: recent runs with at
  // least one completed image come from the API (newest first,
  // non-trashed), and the newest run is followed by default. `modelRuns`
  // carries the repo it belongs to so a late response for a previous
  // model can never leak into the current view. Pages are requested by
  // offset in API-bounded slices; every request stays at the page size,
  // so no amount of Load more can exceed the API limit ceiling.
  const [modelRuns, setModelRuns] = useState<{
    repoId: string;
    runs: RunSummary[];
    total: number;
  }>();
  const [modelRunsError, setModelRunsError] = useState<{
    repoId: string;
    error: unknown;
  }>();
  // One pending "Load more" request: the repo it pages and the offset it
  // starts from. A click replaces it, aborting any still-older request.
  const [modelRunsMore, setModelRunsMore] = useState<{
    repoId: string;
    offset: number;
  } | null>(null);
  const [modelRunsRefresh, setModelRunsRefresh] = useState(0);
  // The repo whose newest run should be followed once its list arrives.
  // Explicit selections (submission, queue View, a clicked run, Clear)
  // consume it, so an arriving list never overrides an explicit choice.
  const pendingDefaultRepo = useRef<string | null>(null);
  // Bumped by every explicit selection, so a submission response that
  // resolves late can tell whether the user chose something in the meantime.
  const selectionVersion = useRef(0);
  // The currently selected repo as seen by async callbacks (submit responses).
  const selectedRepoIdRef = useRef<string | null>(null);

  const profilesQuery = useApiQuery(getProfiles, []);
  const modelsQuery = useApiQuery(listRegistrations, []);
  const systemQuery = useApiQuery(getSystem, []);

  const profiles = profilesQuery.data ?? [];
  const registrations = modelsQuery.data ?? [];
  // Stable empty fallback so effect dependencies do not churn per render.
  const gpus = useMemo(
    () => systemQuery.data?.gpus ?? EMPTY_GPU_LIST,
    [systemQuery.data],
  );

  useEffect(() => {
    setAttention(true);
    return () => setAttention(false);
  }, [setAttention]);

  // Default model: the stored selection when it still resolves to a ready
  // registration, else the first ready one, with the capability defaults of
  // its profile applied from the API. These three effects fill form state
  // from asynchronously arriving external data (queries), which cannot be
  // derived synchronously.
  useEffect(() => {
    const preferred =
      (storedRegistrationId !== null
        ? registrations.find(
            (entry) =>
              entry.id === storedRegistrationId && entry.status === "ready",
          )
        : undefined) ?? registrations.find((entry) => entry.status === "ready");
    const targetId =
      form.registrationId === "" && !reuse
        ? (preferred?.id ?? null)
        : form.registrationId;
    if (targetId === null) return;
    const spec = profiles.find(
      (entry) =>
        entry.profile_id ===
        registrations.find((reg) => reg.id === targetId)?.profile,
    );
    if (form.registrationId === "" && !reuse && targetId !== "") {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- filling a default from arriving async data
      setForm((current) =>
        spec === undefined
          ? { ...current, registrationId: targetId }
          : applyProfileDefaults(
              { ...current, registrationId: targetId },
              spec,
            ),
      );
    } else if (form.steps === "" && spec !== undefined) {
      // Profile metadata arrived after the selection; fill blanks once.
      setForm((current) =>
        current.steps === "" ? applyProfileDefaults(current, spec) : current,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run when list identities change
  }, [registrations, profiles]);

  // Default GPU: the resident worker's GPU once the runtime status is
  // known, else the first listed. Explicit user choices and a reused
  // historical GPU are never overwritten; a provisional first-GPU default
  // is corrected to the resident GPU when the runtime answers late.
  useEffect(() => {
    if (userChoseGpu.current || gpuNeedsChoice || runtime === undefined) {
      return;
    }
    const residentGpu = runtime.resident?.gpu.uuid;
    const preferred =
      residentGpu !== undefined && gpus.some((gpu) => gpu.uuid === residentGpu)
        ? residentGpu
        : (gpus[0]?.uuid ?? "");
    if (preferred !== "" && preferred !== form.gpuUuid) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- filling a default from arriving async data
      setForm((current) => ({ ...current, gpuUuid: preferred }));
    }
  }, [runtime, gpus, gpuNeedsChoice, form.gpuUuid]);

  // Apply a reuse payload once profiles, registrations, and GPUs are all
  // loaded successfully; failed queries leave the one-shot prefill
  // unconsumed so a later retry can still apply it.
  useEffect(() => {
    if (reuseApplied.current || reuse === undefined) {
      return;
    }
    if (
      profilesQuery.loading ||
      modelsQuery.loading ||
      systemQuery.loading ||
      profilesQuery.error !== undefined ||
      modelsQuery.error !== undefined ||
      systemQuery.error !== undefined
    ) {
      return;
    }
    reuseApplied.current = true;
    const result = applyReuse(form, reuse, registrations, profiles, gpus);
    // A resolvable historical GPU counts as the effective choice.
    if (result.state.gpuUuid !== "") {
      userChoseGpu.current = true;
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-shot prefill from arriving async data
    setForm(result.state);
    setNotices(result.notices);
    setGpuNeedsChoice(result.gpuNeedsChoice);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot prefill
  }, [
    reuse,
    profilesQuery.loading,
    modelsQuery.loading,
    systemQuery.loading,
    profilesQuery.error,
    modelsQuery.error,
    systemQuery.error,
  ]);

  const selectedRegistration = registrations.find(
    (entry) => entry.id === form.registrationId,
  );
  const profileSpec = profiles.find(
    (entry) => entry.profile_id === selectedRegistration?.profile,
  );
  const selectedRepoId = selectedRegistration?.repo_id ?? null;

  // Persist the effective model selection (default, reuse, or user pick) so
  // a reload restores the browsed results. Empty selections keep the last
  // stored preference.
  useEffect(() => {
    if (form.registrationId !== "") {
      storeRegistrationId(form.registrationId);
    }
  }, [form.registrationId]);

  // The followed run selection is scoped to one model: switching models
  // resets it and defers selection to the new model's newest run.
  useEffect(() => {
    pendingDefaultRepo.current = selectedRepoId;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- selection reset when the model identity changes
    setFollowedRunId(null);
  }, [selectedRepoId]);

  useEffect(() => {
    selectedRepoIdRef.current = selectedRepoId;
  }, [selectedRepoId]);

  // Mirror of how many records are currently held for the selected repo,
  // so refresh fetches (triggered by a counter) know their depth at run
  // time without re-keying on the list state itself.
  const loadedRunsRef = useRef<{ repoId: string; count: number }>({
    repoId: "",
    count: 0,
  });
  const modelRunsForSelection =
    modelRuns !== undefined && modelRuns.repoId === selectedRepoId
      ? modelRuns
      : undefined;
  useEffect(() => {
    loadedRunsRef.current = {
      repoId: selectedRepoId ?? "",
      count: modelRunsForSelection?.runs.length ?? 0,
    };
  });

  // First page (and every refresh) of the selected model's runs, newest
  // first, non-trashed, and only runs that already saved a completed
  // image — queued or failed runs with no images stay in History and the
  // Queue. A refresh re-reads everything already loaded in API-bounded
  // chunks (at most MODEL_RUNS_REFRESH_MAX per request), so records on
  // deeper loaded pages update too; a Load more that lands mid-refresh
  // keeps its deeper records. The schedule is keyed by repo (and
  // refresh) and aborted on any change, so a straggler response for a
  // previous model is discarded before it can reach state. Errors are
  // keyed by repo and stay visible until a later attempt succeeds.
  useEffect(() => {
    if (selectedRepoId === null) return;
    const controller = new AbortController();
    const { signal } = controller;
    const loaded =
      loadedRunsRef.current.repoId === selectedRepoId
        ? loadedRunsRef.current.count
        : 0;
    const depth = Math.max(loaded, MODEL_RUNS_PAGE_SIZE);
    (async () => {
      try {
        const collected: RunSummary[] = [];
        let total = 0;
        for (let offset = 0; offset < depth; offset += MODEL_RUNS_REFRESH_MAX) {
          const limit = Math.min(MODEL_RUNS_REFRESH_MAX, depth - offset);
          const result = await listRuns(
            { model: selectedRepoId, has_images: true, limit, offset },
            signal,
          );
          if (signal.aborted) return;
          collected.push(...result.runs);
          total = result.total;
          if (result.runs.length < limit) break;
        }
        const seen = new Set<string>();
        const runs = collected.filter((entry) => {
          if (seen.has(entry.run_id)) return false;
          seen.add(entry.run_id);
          return true;
        });
        setModelRuns((current) => {
          if (
            current?.repoId !== selectedRepoId ||
            current.runs.length <= runs.length
          ) {
            return { repoId: selectedRepoId, runs, total };
          }
          // Load more landed while the refresh was in flight; keep the
          // deeper records this refresh did not cover.
          const refreshedIds = new Set(runs.map((entry) => entry.run_id));
          const tail = current.runs.filter(
            (entry) => !refreshedIds.has(entry.run_id),
          );
          return {
            repoId: selectedRepoId,
            runs: [...runs, ...tail],
            total: Math.max(total, current.total),
          };
        });
        setModelRunsError(undefined);
      } catch (cause) {
        if (!signal.aborted) {
          setModelRunsError({ repoId: selectedRepoId, error: cause });
        }
      }
    })();
    return () => controller.abort();
  }, [selectedRepoId, modelRunsRefresh]);

  // Load more: one offset-keyed page beyond the records already held, for
  // the repo it was clicked under. A model switch leaves the request
  // unissued (or aborted) because the repo no longer matches, and appended
  // records are deduplicated against what is already held.
  useEffect(() => {
    if (modelRunsMore === null || modelRunsMore.repoId !== selectedRepoId) {
      return;
    }
    const controller = new AbortController();
    listRuns(
      {
        model: modelRunsMore.repoId,
        has_images: true,
        limit: MODEL_RUNS_PAGE_SIZE,
        offset: modelRunsMore.offset,
      },
      controller.signal,
    ).then(
      (result) => {
        if (controller.signal.aborted) return;
        setModelRuns((current) => {
          if (current?.repoId !== modelRunsMore.repoId) return current;
          const known = new Set(current.runs.map((entry) => entry.run_id));
          const fresh = result.runs.filter((entry) => !known.has(entry.run_id));
          return {
            repoId: current.repoId,
            runs: [...current.runs, ...fresh],
            total: result.total,
          };
        });
      },
      (cause: unknown) => {
        if (controller.signal.aborted) return;
        setModelRunsError({ repoId: modelRunsMore.repoId, error: cause });
      },
    );
    return () => controller.abort();
  }, [modelRunsMore, selectedRepoId]);

  // Default follow: once the selected model's runs arrive, follow the newest
  // (or nothing when the model has no runs yet). Only the list for the
  // pending repo may consume the default, and explicit selections have
  // already cleared it, so this can never override them.
  useEffect(() => {
    if (modelRuns === undefined) return;
    if (pendingDefaultRepo.current !== modelRuns.repoId) return;
    pendingDefaultRepo.current = null;
    setFollowedRunId(modelRuns.runs[0]?.run_id ?? null);
  }, [modelRuns]);

  // Refresh the run list when the worker finishes a run without ever
  // polling the list itself. The followed run's own polling reports its
  // completion (ResultsPanel's onRunSettled), including a first read that
  // is already terminal; this runtime trigger catches runs that finish
  // while unfollowed or between runtime polls — the worker's current run
  // being replaced by another run counts too, since the previous run left
  // the worker without a null observation in between. Submission and
  // cancel responses bump the same counter.
  const currentRunId = runtime?.current_run_id ?? null;
  const previousCurrentRunId = useRef<string | null | undefined>(undefined);
  useEffect(() => {
    const previous = previousCurrentRunId.current;
    previousCurrentRunId.current = currentRunId;
    if (typeof previous === "string" && previous !== currentRunId) {
      setModelRunsRefresh((current) => current + 1);
    }
  }, [currentRunId]);

  const { errors: liveErrors } = useMemo(
    () => validateForm(form, profileSpec),
    [form, profileSpec],
  );
  const displayedErrors = attempted ? liveErrors : errors;

  const onChange = (patch: Partial<GenerationFormState>) => {
    // A user-made GPU choice satisfies the reuse needs-choice hint and
    // pins the selection against later default corrections.
    if (patch.gpuUuid !== undefined) {
      setGpuNeedsChoice(false);
      userChoseGpu.current = true;
    }
    setForm((current) => ({ ...current, ...patch }));
    setSubmitError(undefined);
  };

  const onSelectRegistration = (id: string) => {
    // A model choice must not satisfy the missing-GPU requirement: only an
    // explicit GPU selection may. Leaving gpuNeedsChoice set also keeps the
    // default-GPU effect from silently filling the first device.
    const registration = registrations.find((entry) => entry.id === id);
    const spec = profiles.find(
      (entry) => entry.profile_id === registration?.profile,
    );
    setForm((current) =>
      spec === undefined
        ? { ...current, registrationId: id }
        : applyProfileDefaults({ ...current, registrationId: id }, spec),
    );
    setSubmitError(undefined);
  };

  /** Explicit selection: overrides and cancels any pending default follow. */
  const focusRun = (runId: string | null) => {
    selectionVersion.current += 1;
    pendingDefaultRepo.current = null;
    setFollowedRunId(runId);
  };

  const onSubmit = async () => {
    setAttempted(true);
    const { values: checked, errors: checkErrors } = validateForm(
      form,
      profileSpec,
    );
    setErrors(checkErrors);
    if (Object.keys(checkErrors).length > 0 || profileSpec === undefined) {
      return;
    }
    const request: GenerationRequest = buildRequest(checked, profileSpec);
    // What was selected when the request left; the response must respect
    // whatever the user chose while it was in flight.
    const submission = {
      repoId: selectedRepoId,
      selectionVersion: selectionVersion.current,
    };
    setSubmitting(true);
    setSubmitError(undefined);
    try {
      const detail = await submitGeneration(request);
      const stillCurrent =
        selectedRepoIdRef.current === submission.repoId &&
        selectionVersion.current === submission.selectionVersion;
      if (stillCurrent) {
        focusRun(detail.run_id);
      }
      // Show the queued run in the model's run list right away (only when
      // the submitted model is still the one being browsed).
      if (selectedRepoIdRef.current === submission.repoId) {
        setModelRunsRefresh((current) => current + 1);
      }
    } catch (error) {
      setSubmitError(
        isApiError(error)
          ? error.message
          : "Submission failed. Check the connection and try again.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const modelsError = modelsQuery.error;
  const systemError = systemQuery.error;
  const profilesError = profilesQuery.error;

  return (
    <section className="page page--generate">
      <header className="page-header">
        <h1>Generate</h1>
        <p>
          Text-to-image with the global queue. Pick a registered model and a GPU
          for this request; settings freeze at submission.
        </p>
      </header>

      {notices.map((notice) => (
        <p key={notice} className="notice notice--info" role="status">
          {notice}
        </p>
      ))}

      {modelsError !== undefined ? (
        <ErrorState
          title="Models could not be loaded"
          message="The registration list is unavailable right now."
          onRetry={() => modelsQuery.refetch()}
        />
      ) : null}
      {systemError !== undefined ? (
        <ErrorState
          title="System information unavailable"
          message="GPU selection needs the system endpoint."
          onRetry={() => systemQuery.refetch()}
        />
      ) : null}
      {profilesError !== undefined ? (
        <ErrorState
          title="Profiles could not be loaded"
          message="Capability metadata is unavailable right now."
          onRetry={() => profilesQuery.refetch()}
        />
      ) : null}

      <div className="generate-layout">
        <div className="generate-layout__form">
          <GenerationForm
            form={form}
            errors={displayedErrors}
            registrations={registrations}
            gpus={gpus}
            profileSpec={profileSpec}
            profileLoading={profilesQuery.loading}
            resident={runtime?.resident ?? null}
            gpuNeedsChoice={gpuNeedsChoice}
            submitting={submitting}
            onChange={onChange}
            onSelectRegistration={onSelectRegistration}
            onSubmit={() => void onSubmit()}
          />
          {submitError !== undefined ? (
            <div className="notice notice--error" role="alert">
              {submitError}
            </div>
          ) : null}
        </div>
        <div className="generate-layout__side">
          <QueuePanel
            onFocusRun={focusRun}
            onRunCancelled={(run) => {
              runUpdateSeq.current += 1;
              setRunUpdate({ seq: runUpdateSeq.current, run });
              // Cancellations change the run list (status or removal from
              // the queue); pick the new state up without a reload.
              setModelRunsRefresh((current) => current + 1);
            }}
          />
          <ResultsPanel
            runId={followedRunId}
            onClear={() => focusRun(null)}
            runUpdate={runUpdate}
            modelRuns={modelRunsForSelection}
            modelRunsLoading={
              modelsQuery.loading ||
              (selectedRepoId !== null &&
                modelRunsForSelection === undefined &&
                modelRunsError?.repoId !== selectedRepoId)
            }
            modelRunsError={
              modelRunsError !== undefined &&
              modelRunsError.repoId === selectedRepoId
                ? modelRunsError.error
                : undefined
            }
            onSelectRun={focusRun}
            onRetryModelRuns={() =>
              setModelRunsRefresh((current) => current + 1)
            }
            onLoadMoreModelRuns={() =>
              setModelRunsMore({
                repoId: selectedRepoId as string,
                offset: modelRunsForSelection?.runs.length ?? 0,
              })
            }
            onRunSettled={() => setModelRunsRefresh((current) => current + 1)}
            onRunChanged={(updated) => {
              if (selectedRepoIdRef.current !== updated.repo_id) return;
              setModelRuns((current) => {
                if (current?.repoId !== updated.repo_id) return current;
                const present = current.runs.some(
                  (entry) => entry.run_id === updated.run_id,
                );
                return {
                  ...current,
                  runs: updated.trashed
                    ? current.runs.filter(
                        (entry) => entry.run_id !== updated.run_id,
                      )
                    : current.runs.map((entry) =>
                        entry.run_id === updated.run_id ? updated : entry,
                      ),
                  total: current.total - (updated.trashed && present ? 1 : 0),
                };
              });
              setModelRunsMore(null);
              setModelRunsRefresh((current) => current + 1);
            }}
          />
        </div>
      </div>
    </section>
  );
}
