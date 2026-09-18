import { Component, type ErrorInfo, type ReactNode } from "react";

import { ErrorState } from "./ui/ErrorState";

interface ErrorBoundaryProps {
  children: ReactNode;
  /**
   * Changing this value resets a caught error, which lets route changes
   * recover the view without remounting the whole tree.
   */
  resetKey?: string;
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    if (
      this.state.error !== null &&
      prevProps.resetKey !== this.props.resetKey
    ) {
      this.reset();
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Unhandled interface error:", error, info.componentStack);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (error === null) {
      return this.props.children;
    }
    if (this.props.fallback) {
      return this.props.fallback(error, this.reset);
    }
    return (
      <ErrorState
        title="Unexpected interface error"
        message="Something broke while rendering this view."
        detail={error.message}
        onRetry={this.reset}
        retryLabel="Reset view"
      />
    );
  }
}
