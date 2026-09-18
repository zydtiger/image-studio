import type { ReactNode } from "react";
import { HashRouter, Navigate, Route, Routes, useLocation } from "react-router";

import { ErrorBoundary } from "./components/ErrorBoundary";
import { AppShell } from "./components/app/AppShell";
import GeneratePage from "./pages/GeneratePage";
import HistoryPage from "./pages/HistoryPage";
import ModelsPage from "./pages/ModelsPage";
import NotFoundPage from "./pages/NotFoundPage";
import SettingsPage from "./pages/SettingsPage";
import { ToastProvider } from "./state/toast/ToastProvider";
import { RuntimeProvider } from "./state/runtime/RuntimeProvider";

/** Resets the error boundary whenever the route changes. */
function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  return <ErrorBoundary resetKey={pathname}>{children}</ErrorBoundary>;
}

export default function App() {
  return (
    <ToastProvider>
      <RuntimeProvider>
        <HashRouter>
          <RoutedErrorBoundary>
            <Routes>
              <Route element={<AppShell />}>
                <Route index element={<Navigate to="/generate" replace />} />
                <Route path="/generate" element={<GeneratePage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<NotFoundPage />} />
              </Route>
            </Routes>
          </RoutedErrorBoundary>
        </HashRouter>
      </RuntimeProvider>
    </ToastProvider>
  );
}
