import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router";

import { getProfiles } from "../api/profiles";
import { listRegistrations } from "../api/models";
import { getSystem } from "../api/system";
import { submitGeneration } from "../api/generations";
import type { GenerationRequest, GpuInfo } from "../api/types";
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

/**
 * Core flow: choose model and GPU, write a prompt, generate, follow the
 * result. Form edits never load a model; submission freezes the settings.
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
  const [notices, setNotices] = useState<string[]>([]);
  const [gpuNeedsChoice, setGpuNeedsChoice] = useState(false);
  const reuseApplied = useRef(false);
  // Set once the user (or a resolved reuse payload) fixes the GPU; later
  // runtime arrivals must not override that choice.
  const userChoseGpu = useRef(false);

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

  // Default model: first ready registration until the user chooses one,
  // with the capability defaults of its profile applied from the API.
  // These three effects fill form state from asynchronously arriving
  // external data (queries), which cannot be derived synchronously.
  useEffect(() => {
    const targetId =
      form.registrationId === "" && !reuse
        ? (registrations.find((entry) => entry.status === "ready")?.id ?? null)
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
    setSubmitting(true);
    setSubmitError(undefined);
    try {
      const detail = await submitGeneration(request);
      setFollowedRunId(detail.run_id);
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
          <QueuePanel onFocusRun={setFollowedRunId} />
          <ResultsPanel
            runId={followedRunId}
            onClear={() => setFollowedRunId(null)}
          />
        </div>
      </div>
    </section>
  );
}
