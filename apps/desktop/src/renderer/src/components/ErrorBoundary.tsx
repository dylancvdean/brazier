import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = {
  /** What failed to render, in the fallback text ("Couldn't show this message."). */
  label: string
  /** Changing this clears a caught error, e.g. when the user switches modes. */
  resetKey?: unknown
  /** Custom fallback; receives a callback that retries rendering. */
  fallback?: (retry: () => void) => ReactNode
  children: ReactNode
}

type State = { error: Error | null; resetKey: unknown }

/**
 * Contains a render failure to one message or workspace instead of blanking
 * the whole window.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, resetKey: this.props.resetKey }

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) }
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (Object.is(props.resetKey, state.resetKey)) return null
    return { error: null, resetKey: props.resetKey }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`Render failed: ${this.props.label}`, error, info.componentStack)
  }

  private retry = (): void => this.setState({ error: null })

  render(): ReactNode {
    if (!this.state.error) return this.props.children
    if (this.props.fallback) return this.props.fallback(this.retry)
    return (
      <div className="render-error" role="alert">
        <span>Couldn&apos;t show this {this.props.label}.</span>
        <button type="button" onClick={this.retry}>
          Retry
        </button>
      </div>
    )
  }
}

/** Runs `render` as a child component so an {@link ErrorBoundary} can catch it. */
export function RenderSlot({ render }: { render: () => ReactNode }): ReactNode {
  return render()
}
