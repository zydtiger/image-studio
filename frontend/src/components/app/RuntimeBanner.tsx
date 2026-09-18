import { useState } from "react";

import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { useToast } from "../../state/toast/context";
import { errorMessage } from "../../api/client";
import { useRuntime } from "../../state/runtime/context";
import {
  shortCommit,
  workerStateLabel,
  workerStateTone,
} from "../../lib/statusTone";
import { ConfirmDialog } from "../ui/ConfirmDialog";

/**
 * Global status strip: resident model AND its GPU (kept visible after the
 * current run clears), worker state, queue depth, the explicit Eject
 * action (idle only), and the honest fake-development indicator.
 */
export function RuntimeBanner() {
  const { runtime, error, connected, refresh, eject } = useRuntime();
  const toast = useToast();
  const [confirming, setConfirming] = useState(false);

  const resident = runtime?.resident ?? null;
  const state = runtime?.state;
  const busy =
    state === "loading" ||
    state === "generating" ||
    state === "switching" ||
    state === "ejecting";
  const canEject = state === "idle" && resident !== null;
  const ejectDisabledReason = busy
    ? `Eject is unavailable while the worker is ${workerStateLabel(state ?? "loading").toLowerCase()}.`
    : resident === null || state === "unloaded"
      ? "No model is loaded."
      : undefined;

  const onEject = async () => {
    const outcome = await eject();
    if (outcome.ok) {
      toast.pushToast({
        kind: "success",
        message: "Model ejected. Files and registrations are untouched.",
      });
    } else if (outcome.error) {
      toast.pushToast({
        kind: "error",
        message: outcome.error.message,
      });
    }
    setConfirming(false);
  };

  return (
    <section
      className="runtime-banner"
      aria-label="Runtime status"
      data-offline={!connected}
    >
      {runtime === undefined && error === undefined ? (
        <p className="runtime-banner__stale" role="status">
          Connecting to the server…
        </p>
      ) : runtime === undefined && error !== undefined ? (
        <p className="runtime-banner__error" role="alert">
          Server unreachable. Retrying in the background.
        </p>
      ) : (
        <>
          {runtime?.implementation === "fake" ? (
            <Badge tone="warning">Fake runtime (development)</Badge>
          ) : null}
          <div className="runtime-banner__cell">
            <span className="runtime-banner__label">Resident model</span>
            {resident ? (
              <span className="runtime-banner__value runtime-banner__value--strong">
                {resident.repo_id}
                <span className="runtime-banner__sub">
                  {resident.profile} · {shortCommit(resident.commit_sha)} ·{" "}
                  {resident.dtype}
                </span>
              </span>
            ) : (
              <span className="runtime-banner__value">None</span>
            )}
          </div>
          <div className="runtime-banner__cell">
            <span className="runtime-banner__label">GPU</span>
            <span className="runtime-banner__value">
              {resident ? resident.gpu.name : "—"}
            </span>
          </div>
          <div className="runtime-banner__cell">
            <span className="runtime-banner__label">Worker</span>
            <Badge tone={workerStateTone(state ?? "unloaded")}>
              {workerStateLabel(state ?? "unloaded")}
            </Badge>
          </div>
          <div className="runtime-banner__cell">
            <span className="runtime-banner__label">Queue</span>
            <span className="runtime-banner__value">
              {runtime ? runtime.queue_depth : "…"}
            </span>
          </div>
          {runtime?.current_run_id ? (
            <div className="runtime-banner__cell">
              <span className="runtime-banner__label">Current run</span>
              <span className="runtime-banner__value runtime-banner__value--mono">
                {runtime.current_run_id.slice(0, 8)}
              </span>
            </div>
          ) : null}
        </>
      )}
      <div className="runtime-banner__actions">
        <Button
          size="sm"
          variant="secondary"
          disabled={!canEject}
          title={ejectDisabledReason}
          onClick={() => setConfirming(true)}
        >
          Eject model
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => refresh()}
          title="Refresh runtime status"
        >
          Refresh
        </Button>
      </div>
      {error !== undefined && runtime !== undefined ? (
        <p className="runtime-banner__stale" role="status">
          Showing the last known status ({errorMessage(error)}).
        </p>
      ) : null}
      <ConfirmDialog
        open={confirming}
        onClose={() => setConfirming(false)}
        onConfirm={() => void onEject()}
        title="Eject resident model?"
        confirmLabel="Eject"
        tone="danger"
      >
        The worker stops and releases the model. Files in the cache and all
        registrations stay untouched. Generation starts fresh on the next
        submission.
      </ConfirmDialog>
    </section>
  );
}
