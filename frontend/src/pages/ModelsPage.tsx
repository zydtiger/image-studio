import { useState } from "react";

import { listRegistrations } from "../api/models";
import type { ModelRegistration } from "../api/types";
import { Tabs } from "../components/ui/Tabs";
import { usePolling } from "../hooks/usePolling";
import { DiscoverPanel } from "../components/models/DiscoverPanel";
import { DownloadsPanel } from "../components/models/DownloadsPanel";
import { LocalCachePanel } from "../components/models/LocalCachePanel";
import { MyModelsPanel } from "../components/models/MyModelsPanel";

type ModelsTab = "discover" | "library" | "cache" | "downloads";

/**
 * Model management: Hub discovery, application registrations, the local
 * cache, and the separate download queue.
 */
export default function ModelsPage() {
  const [tab, setTab] = useState<ModelsTab>("discover");
  const [registrations, setRegistrations] = useState<ModelRegistration[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>();
  const [refreshKey, setRefreshKey] = useState(0);
  const refreshModels = () => setRefreshKey((value) => value + 1);
  usePolling(
    async (signal) => {
      const models = await listRegistrations(signal);
      if (signal.aborted) return;
      setRegistrations(models);
      setError(undefined);
      setLoading(false);
    },
    {
      enabled: tab === "library",
      activeIntervalMs: 1_000,
      backgroundIntervalMs: 5_000,
      refreshKey,
      onError: (cause) => {
        setError(cause);
        setLoading(false);
      },
    },
  );

  return (
    <section className="page page--models">
      <header className="page-header">
        <h1>Models</h1>
        <p>
          Search Hugging Face, manage registrations, inspect the local cache,
          and follow downloads.
        </p>
      </header>
      <Tabs
        aria-label="Model views"
        activeId={tab}
        onChange={(id) => setTab(id as ModelsTab)}
        tabs={[
          {
            id: "library",
            label: "My Models",
            content: (
              <MyModelsPanel
                registrations={registrations}
                loading={loading}
                error={error}
                onRetry={refreshModels}
                onChanged={refreshModels}
                onGoDiscover={() => setTab("discover")}
              />
            ),
          },
          {
            id: "cache",
            label: "Local Cache",
            content: <LocalCachePanel onRegistered={refreshModels} />,
          },
          {
            id: "discover",
            label: "Discover",
            content: (
              <DiscoverPanel
                onDownloaded={() => setTab("downloads")}
                onRegistered={refreshModels}
              />
            ),
          },
          {
            id: "downloads",
            label: "Downloads",
            content: <DownloadsPanel />,
          },
        ]}
      />
    </section>
  );
}
