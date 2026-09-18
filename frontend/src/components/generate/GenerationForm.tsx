import type {
  GpuInfo,
  ModelRegistration,
  ProfileSpec,
  ResidentModel,
} from "../../api/types";
import { Button } from "../ui/Button";
import { NumberField, SelectField, TextField } from "../ui/fields";
import { cx } from "../../lib/cx";
import type { FormErrors, GenerationFormState } from "./formState";

export interface GenerationFormProps {
  form: GenerationFormState;
  errors: FormErrors;
  registrations: ModelRegistration[];
  gpus: GpuInfo[];
  profileSpec: ProfileSpec | undefined;
  profileLoading: boolean;
  resident: ResidentModel | null;
  /** Set when reuse prefill could not resolve the historical GPU. */
  gpuNeedsChoice: boolean;
  submitting: boolean;
  onChange: (patch: Partial<GenerationFormState>) => void;
  onSelectRegistration: (id: string) => void;
  onSubmit: () => void;
}

function modelLabel(registration: ModelRegistration): string {
  const name = registration.display_name ?? registration.repo_id;
  return `${name} (${registration.profile})`;
}

function replacementNotice(
  resident: ResidentModel | null,
  registration: ModelRegistration | undefined,
  gpuUuid: string,
  gpuName: string | undefined,
): { tone: "info" | "warning"; text: string } | null {
  if (!resident || !registration) return null;
  const gpu = gpuUuid === "" ? null : gpuUuid;
  const sameModel =
    resident.registration_id === registration.id &&
    resident.profile === registration.profile;
  const sameGpu = resident.gpu.uuid === gpu;
  if (sameModel && sameGpu) {
    return {
      tone: "info",
      text: "Reuses the resident model on its current GPU.",
    };
  }
  const targetGpu = gpuName ?? "the selected GPU";
  const targetName = registration.display_name ?? registration.repo_id;
  return {
    tone: "warning",
    text: `After the current work finishes, ${resident.repo_id} on ${resident.gpu.name} is fully unloaded and ${targetName} loads on ${targetGpu}.`,
  };
}

/**
 * Capability-driven generation form. Control visibility and bounds come
 * from the profile spec served by the API; editing never loads a model.
 */
export function GenerationForm({
  form,
  errors,
  registrations,
  gpus,
  profileSpec,
  profileLoading,
  resident,
  gpuNeedsChoice,
  submitting,
  onChange,
  onSelectRegistration,
  onSubmit,
}: GenerationFormProps) {
  const readyRegistrations = registrations.filter(
    (entry) => entry.status === "ready",
  );
  const missingRegistrations = registrations.filter(
    (entry) => entry.status === "missing_files",
  );
  const selected = registrations.find(
    (entry) => entry.id === form.registrationId,
  );
  const selectedGpu = gpus.find((entry) => entry.uuid === form.gpuUuid);
  const notice = replacementNotice(
    resident,
    selected,
    form.gpuUuid,
    selectedGpu?.name,
  );
  const supportsNegative = profileSpec?.negative_prompt_supported === true;
  const guidanceAdjustable =
    profileSpec !== undefined && profileSpec.guidance_fixed === null;

  return (
    <form
      className="generation-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <SelectField
        id="generation-model"
        label="Model"
        error={errors.registrationId}
        value={form.registrationId}
        onChange={(event) => onSelectRegistration(event.target.value)}
        hint={
          readyRegistrations.length === 0
            ? "No ready models. Register or download one on the Models page."
            : undefined
        }
      >
        <option value="" disabled>
          Select a model…
        </option>
        {readyRegistrations.map((entry) => (
          <option key={entry.id} value={entry.id}>
            {modelLabel(entry)}
          </option>
        ))}
        {missingRegistrations.map((entry) => (
          <option key={entry.id} value={entry.id} disabled>
            {modelLabel(entry)} — missing files
          </option>
        ))}
      </SelectField>

      <SelectField
        id="generation-gpu"
        label="GPU"
        error={errors.gpuUuid}
        value={form.gpuUuid}
        onChange={(event) => onChange({ gpuUuid: event.target.value })}
        hint={gpuNeedsChoice ? "Choose a GPU to continue." : undefined}
      >
        <option value="" disabled>
          Select a GPU…
        </option>
        {gpus.map((gpu) => (
          <option key={gpu.uuid} value={gpu.uuid}>
            {gpu.name}
          </option>
        ))}
      </SelectField>

      <TextField
        id="generation-prompt"
        label="Prompt"
        error={errors.prompt}
        value={form.prompt}
        placeholder="Describe the image…"
        multiline
        rows={4}
        maxLength={8000}
        onChange={(event) => onChange({ prompt: event.target.value })}
      />

      {supportsNegative ? (
        <TextField
          id="generation-negative-prompt"
          label="Negative prompt"
          hint="What to avoid in the image; supported by this profile."
          value={form.negativePrompt}
          multiline
          rows={2}
          maxLength={8000}
          onChange={(event) => onChange({ negativePrompt: event.target.value })}
        />
      ) : null}

      <div className="form-grid">
        <NumberField
          id="generation-width"
          label="Width"
          error={errors.width}
          value={form.width}
          min={256}
          max={2048}
          step={16}
          onChange={(event) => onChange({ width: event.target.value })}
        />
        <NumberField
          id="generation-height"
          label="Height"
          error={errors.height}
          value={form.height}
          min={256}
          max={2048}
          step={16}
          onChange={(event) => onChange({ height: event.target.value })}
        />
        <NumberField
          id="generation-steps"
          label="Steps"
          error={errors.steps}
          value={form.steps}
          min={profileSpec?.min_steps}
          max={profileSpec?.max_steps}
          step={1}
          disabled={profileLoading}
          onChange={(event) => onChange({ steps: event.target.value })}
          hint={
            profileSpec
              ? `${profileSpec.min_steps}-${profileSpec.max_steps}`
              : undefined
          }
        />
        {guidanceAdjustable ? (
          <NumberField
            id="generation-guidance"
            label="Guidance"
            error={errors.guidance}
            value={form.guidance}
            min={0}
            step={0.1}
            onChange={(event) => onChange({ guidance: event.target.value })}
          />
        ) : null}
        <TextField
          id="generation-seed"
          label="Seed"
          hint="Leave empty for a random seed resolved at submission."
          error={errors.seed}
          value={form.seed}
          inputMode="numeric"
          onChange={(event) => onChange({ seed: event.target.value })}
        />
        <SelectField
          id="generation-count"
          label="Images"
          error={errors.count}
          value={form.count}
          onChange={(event) => onChange({ count: event.target.value })}
        >
          {[1, 2, 3, 4].map((value) => (
            <option key={value} value={String(value)}>
              {value}
            </option>
          ))}
        </SelectField>
      </div>

      {profileSpec !== undefined && !guidanceAdjustable ? (
        <p className="field-hint">
          Guidance is fixed at {profileSpec.guidance_fixed} for{" "}
          {profileSpec.label}; the field is not submitted.
        </p>
      ) : null}

      {notice ? (
        <p
          className={cx(
            "notice",
            notice.tone === "warning" ? "notice--warning" : "notice--info",
          )}
        >
          {notice.text}
        </p>
      ) : null}

      <div className="generation-form__actions">
        <Button type="submit" variant="primary" disabled={submitting}>
          {submitting ? "Submitting…" : "Generate"}
        </Button>
      </div>
    </form>
  );
}
