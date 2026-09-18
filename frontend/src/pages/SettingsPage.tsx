import { getSystem } from "../api/system";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { ErrorState } from "../components/ui/ErrorState";
import { Spinner } from "../components/ui/Spinner";
import { useApiQuery } from "../hooks/useApiQuery";
import { useRuntime } from "../state/runtime/context";
import { formatBytes } from "../lib/format";
import { shortCommit, workerStateLabel } from "../lib/statusTone";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="settings-section">
      <h2>{title}</h2>
      <div className="settings-section__body">{children}</div>
    </section>
  );
}

function PathRow({ label, path }: { label: string; path: string }) {
  return (
    <div className="detail-row">
      <dt>{label}</dt>
      <dd>
        <span className="mono settings-path">{path}</span>
      </dd>
    </div>
  );
}

/**
 * Effective configuration and environment. Values are read-only:
 * configuration changes require a server restart, which this page states.
 */
export default function SettingsPage() {
  const systemQuery = useApiQuery(getSystem, []);
  const { runtime } = useRuntime();
  const system = systemQuery.data;

  return (
    <section className="page page--settings">
      <header className="page-header">
        <h1>Settings</h1>
        <p>
          Effective configuration as served by the backend. Configuration
          changes are made in the config file and require a restart; nothing is
          edited live.
        </p>
      </header>

      {systemQuery.error !== undefined ? (
        <ErrorState
          title="System information unavailable"
          message="The system endpoint could not be reached."
          onRetry={() => systemQuery.refetch()}
        />
      ) : system === undefined ? (
        <p>
          <Spinner label="Loading system information" />
        </p>
      ) : (
        <>
          <Section title="Server">
            <dl className="detail-list">
              <div className="detail-row">
                <dt>Listener</dt>
                <dd>
                  {system.host}:{system.port}
                </dd>
              </div>
              {system.development.fake_runtime ||
              system.development.fake_hub ? (
                <div className="detail-row">
                  <dt>Development</dt>
                  <dd>
                    {system.development.fake_runtime ? (
                      <Badge tone="warning">fake runtime</Badge>
                    ) : null}{" "}
                    {system.development.fake_hub ? (
                      <Badge tone="warning">fake hub</Badge>
                    ) : null}
                  </dd>
                </div>
              ) : null}
            </dl>
          </Section>

          <Section title="Hugging Face">
            <dl className="detail-list">
              <div className="detail-row">
                <dt>Login</dt>
                <dd>
                  {system.hf_logged_in
                    ? `Logged in${system.hf_username ? ` as ${system.hf_username}` : ""}`
                    : "Not logged in. Gated models need an existing Hub login."}
                </dd>
              </div>
              <PathRow label="Hub cache" path={system.paths.hub_cache_dir} />
            </dl>
          </Section>

          <Section title="GPUs">
            {system.gpus.length === 0 ? (
              <p className="field-hint">No GPUs are visible to the server.</p>
            ) : (
              <ul className="settings-gpus">
                {system.gpus.map((gpu) => (
                  <li key={gpu.uuid}>
                    <span>{gpu.name}</span>
                    <span className="field-hint">
                      index {gpu.index}
                      {gpu.memory_total_bytes !== null &&
                      gpu.memory_total_bytes !== undefined
                        ? ` · ${formatBytes(gpu.memory_total_bytes)}`
                        : ""}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Section>

          <Section title="Paths">
            <dl className="detail-list">
              <PathRow label="Config file" path={system.paths.config_file} />
              <PathRow label="Data directory" path={system.paths.data_dir} />
              <PathRow label="Database" path={system.paths.database_file} />
              <PathRow label="Outputs" path={system.paths.outputs_dir} />
              <PathRow label="Trash" path={system.paths.trash_dir} />
              <PathRow label="Thumbnails" path={system.paths.thumbnails_dir} />
              <PathRow label="Log file" path={system.paths.log_file} />
            </dl>
          </Section>

          <Section title="Resident model">
            {runtime?.resident ? (
              <dl className="detail-list">
                <div className="detail-row">
                  <dt>Model</dt>
                  <dd>
                    {runtime.resident.repo_id} · {runtime.resident.profile} ·{" "}
                    <span className="mono">
                      {shortCommit(runtime.resident.commit_sha)}
                    </span>
                  </dd>
                </div>
                <div className="detail-row">
                  <dt>GPU</dt>
                  <dd>{runtime.resident.gpu.name}</dd>
                </div>
                <div className="detail-row">
                  <dt>Precision</dt>
                  <dd>{runtime.resident.dtype}</dd>
                </div>
                <div className="detail-row">
                  <dt>Worker</dt>
                  <dd>{workerStateLabel(runtime.state)}</dd>
                </div>
              </dl>
            ) : (
              <p className="field-hint">
                No model is resident. The model stays loaded after generation
                until you eject it; there is no idle timeout.
              </p>
            )}
          </Section>
        </>
      )}

      <p className="settings-refresh">
        <Button
          size="sm"
          variant="secondary"
          onClick={() => systemQuery.refetch()}
        >
          Refresh
        </Button>
      </p>
    </section>
  );
}
