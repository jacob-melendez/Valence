import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props { name: string; children: ReactNode }
interface State { error: Error | null }

const RETRY_MS = 5000;

/** Keeps one panel's render error (e.g. an unexpected backend payload) from blanking the whole console. */
export class PanelBoundary extends Component<Props, State> {
  override state: State = { error: null };
  private retry: ReturnType<typeof setTimeout> | null = null;

  static getDerivedStateFromError(error: unknown): State {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  override componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error(`[panel ${this.props.name}]`, error, info.componentStack);
    // Data keeps streaming, so try again shortly: the next message may render fine.
    if (this.retry) clearTimeout(this.retry);
    this.retry = setTimeout(() => this.setState({ error: null }), RETRY_MS);
  }

  override componentWillUnmount() {
    if (this.retry) clearTimeout(this.retry);
  }

  override render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div role="alert" className="panel border-destructive">
        <div className="panel-title text-destructive">Panel error — {this.props.name}</div>
        <div className="font-mono text-sm text-destructive">{error.message || String(error)}</div>
        <button onClick={() => this.setState({ error: null })} className="mt-3 rounded-md border px-3 py-1.5 font-mono text-sm hover:bg-accent">
          Retry
        </button>
      </div>
    );
  }
}
