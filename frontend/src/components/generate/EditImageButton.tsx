import { useState } from "react";
import { useNavigate } from "react-router";
import { Pencil } from "lucide-react";
import { inputFromArtifact } from "../../api/inputs";
import { getRun } from "../../api/generations";
import { errorMessage } from "../../api/client";
import { buildReusePayload } from "../../lib/reuse";
import { Spinner } from "../ui/Spinner";

export function EditImageButton({
  runId,
  artifactId,
}: {
  runId: string;
  artifactId: string;
}) {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const edit = async () => {
    setBusy(true);
    setError(undefined);
    try {
      const run = await getRun(runId);
      const image = await inputFromArtifact(runId, artifactId);
      const reuse = {
        ...buildReusePayload(run),
        mode: "image-edit" as const,
        inputImages: [image],
        prompt: "",
        count: 1,
      };
      navigate("/generate", { state: { reuse } });
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button
        type="button"
        className="icon-button icon-button--sm image-tile__edit"
        aria-label="Edit image"
        title={busy ? "Preparing image..." : "Edit image"}
        aria-busy={busy}
        disabled={busy}
        onClick={() => void edit()}
      >
        {busy ? (
          <Spinner size="sm" label="Preparing image" />
        ) : (
          <Pencil size={16} aria-hidden="true" />
        )}
      </button>
      {error && (
        <p className="image-tile__error" role="alert">
          {error}
        </p>
      )}
    </>
  );
}
