import { useEffect, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { uploadInput } from "../../api/inputs";
import { errorMessage } from "../../api/client";
import type { InputImage, ProfileSpec } from "../../api/types";
import { Button } from "../ui/Button";

export function InputImagePicker({
  image,
  profile,
  disabled,
  onChange,
  onBusyChange,
}: {
  image?: InputImage;
  profile: ProfileSpec;
  disabled: boolean;
  onChange: (image?: InputImage) => void;
  onBusyChange: (busy: boolean) => void;
}) {
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  const upload = async (files: FileList | null) => {
    if (!files?.length || busy || disabled) return;
    setError(undefined);
    if (files.length !== 1) {
      setError("Select one image.");
      return;
    }
    if (files[0].size > profile.max_input_bytes) {
      setError("Image exceeds 20 MiB.");
      return;
    }
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    onBusyChange(true);
    try {
      const uploaded = await uploadInput(files[0], abort.signal);
      if (!abort.signal.aborted) onChange(uploaded);
    } catch (error) {
      if (!abort.signal.aborted) setError(errorMessage(error));
    } finally {
      if (!abort.signal.aborted) {
        setBusy(false);
        onBusyChange(false);
      }
    }
  };

  return (
    <div
      className="input-image-picker"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        void upload(event.dataTransfer.files);
      }}
    >
      <span className="field-label">Input image</span>
      {image && (
        <img
          className="input-image-preview"
          src={image.url}
          alt="Input image"
        />
      )}
      <div className="input-image-actions">
        <input
          ref={fileInput}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          aria-label="Upload input image"
          hidden
          onChange={(event) => {
            void upload(event.target.files);
            event.target.value = "";
          }}
        />
        <Button
          type="button"
          variant="secondary"
          disabled={busy || disabled}
          onClick={() => fileInput.current?.click()}
        >
          <Upload size={16} aria-hidden="true" />
          {busy ? "Uploading..." : image ? "Replace image" : "Upload image"}
        </Button>
        {image && (
          <>
            <span>
              {image.width} × {image.height}
            </span>
            <button
              type="button"
              className="icon-button"
              title="Remove input image"
              aria-label="Remove input image"
              disabled={busy || disabled}
              onClick={() => {
                onChange();
                setError(undefined);
              }}
            >
              <X size={18} aria-hidden="true" />
            </button>
          </>
        )}
      </div>
      {error && (
        <p role="alert" className="notice notice--error">
          {error}
        </p>
      )}
    </div>
  );
}
