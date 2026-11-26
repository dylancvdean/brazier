/**
 * The Agent screen's view of one session, as a pure reduction of the events the
 * shared agent core emits. It mirrors the desktop's `AgentMode` reducer.
 */
import type {
  AgentApproval, AgentEvent, AgentMessage, AgentPermissionMode, AgentPermissionSettings, AgentRunSummary,
  ExecutionLocation, SandboxDescription, ToolExecutionRecord,
} from '../../../desktop/src/agent/core/types';

export type { AgentApproval, AgentEvent, AgentMessage, AgentPermissionMode, AgentRunSummary };

export type AgentSessionSummary = {
  id: string; title: string; workspace_path?: string | null; model: string; runtime_id: string;
  permission_mode: AgentPermissionMode; permission_settings: AgentPermissionSettings;
  enabled_tools?: string[] | null; last_run_status: string; created_at: string; updated_at: string;
};
export type AgentSessionDetail = {
  session: AgentSessionSummary;
  messages: { payload: AgentMessage }[];
  tool_executions: ToolExecutionRecord[];
  pending_approvals: AgentApproval[];
};
export type AgentTool = {
  name: string; label: string; description: string; risk: string; needs_workspace: boolean; power_tool?: boolean;
};
export type AgentCapabilities = {
  sandbox: { backend: string; isolated: boolean; sandboxed_execution: boolean; detail: string };
  permission_modes: AgentPermissionMode[];
};

export type TimelineStatus = 'running' | 'reviewing' | 'awaiting-approval' | 'completed' | 'failed' | 'denied';
export type TimelineEntry = {
  toolCallId: string; tool: string; args: Record<string, unknown>; environment: 'sandbox' | 'host';
  status: TimelineStatus; timestamp: string;
  /** Set while "Approve for me" is checking or has allowed this call. */
  reviewNote?: string;
  sandbox?: SandboxDescription; output?: string; error?: string; exitCode?: number | null;
  changedPaths?: string[]; truncated?: boolean; durationMs?: number;
};

export type AgentView = {
  messages: AgentMessage[];
  timeline: TimelineEntry[];
  approvals: AgentApproval[];
  /** Assistant text and reasoning of the turn being generated. */
  streaming: string;
  reasoning: string;
  running: boolean;
  /** What the host is doing before the first token: loading the model or reading the prompt. */
  progress: string;
  summary: AgentRunSummary | null;
  error: string;
};

export const emptyView: AgentView = {
  messages: [], timeline: [], approvals: [], streaming: '', reasoning: '', running: false, progress: '',
  summary: null, error: '',
};

/** Modes a paired device may choose. `skip-permissions` needs proof from the host's own desktop. */
export const REMOTE_PERMISSION_MODES: readonly AgentPermissionMode[] = ['ask', 'approve-for-me', 'sandbox-only'];

export const PERMISSION_LABELS: Record<AgentPermissionMode, { title: string; detail: string }> = {
  ask: { title: 'Ask first', detail: 'Approve writes, commands, network use, and anything outside the workspace.' },
  'approve-for-me': {
    title: 'Approve for me',
    detail: "The task's model approves sandboxed actions that stay within your instructions. Host and destructive actions still ask you.",
  },
  'sandbox-only': { title: 'Sandbox only', detail: 'Sandboxed work runs without prompts. Host access is refused outright.' },
  'skip-permissions': { title: 'Skip permissions', detail: 'Only available from the host itself.' },
};

/** The modes this device can pick on this host, in the desktop's order. */
export function permissionModesFor(capabilities: AgentCapabilities | null): AgentPermissionMode[] {
  return REMOTE_PERMISSION_MODES.filter((mode) => !capabilities || (capabilities.permission_modes.includes(mode) &&
    (mode !== 'sandbox-only' || capabilities.sandbox.sandboxed_execution)));
}

/** Tools a session can toggle: the standard set, plus any power tool the host already enabled for it. */
export function selectableTools(catalog: readonly AgentTool[], enabled: readonly string[] | null | undefined): AgentTool[] {
  return catalog.filter((tool) => !tool.power_tool || enabled?.includes(tool.name));
}

/** Daemon refusals a paired device cannot fix by itself, said in terms of what to do on the host. */
export function agentErrorText(message: string): string {
  if (message.includes('owner must authorize this workspace')) {
    return 'This folder has not been opened on the host yet. Open it once in Agent mode in Brazier on the host, then try again.';
  }
  if (message.includes('local desktop')) return 'That permission level can only be granted from the host itself.';
  return message;
}

export function argsPreview(tool: string, args: Record<string, unknown>): string {
  if (typeof args.prompt === 'string' && tool === 'spawn_subagent') return args.prompt.trim().slice(0, 80);
  if (typeof args.command === 'string') return args.command;
  if (typeof args.path === 'string') return args.path;
  if (typeof args.from === 'string' && typeof args.to === 'string') return `${args.from} → ${args.to}`;
  if (typeof args.query === 'string') return `“${args.query}”`;
  if (typeof args.process_id === 'string') return args.process_id;
  const json = JSON.stringify(args);
  return json === '{}' ? tool : json.slice(0, 120);
}

function timelineFromRecords(records: readonly ToolExecutionRecord[]): TimelineEntry[] {
  return records.map((record) => ({
    toolCallId: record.tool_call_id ?? record.id, tool: record.tool, args: record.arguments ?? {},
    environment: record.environment === 'host' ? 'host' : 'sandbox',
    status: record.status === 'completed' || record.status === 'denied' || record.status === 'awaiting-approval'
      ? record.status : 'failed',
    sandbox: record.sandbox, output: record.output_preview, error: record.error ?? undefined,
    exitCode: record.exit_code ?? null, changedPaths: record.changed_paths ?? [], truncated: record.truncated,
    durationMs: record.duration_ms, timestamp: record.created_at,
  }));
}

/** The view of a session as the host stored it. */
export function viewFromDetail(detail: AgentSessionDetail): AgentView {
  return {
    ...emptyView, messages: detail.messages.map((record) => record.payload),
    timeline: timelineFromRecords(detail.tool_executions), approvals: detail.pending_approvals,
  };
}

/** Assistant text worth a bubble: tool-only turns and a literal "null" from weak models are blank. */
function assistantText(text: unknown): string {
  if (typeof text !== 'string') return '';
  const trimmed = text.trim();
  return !trimmed || trimmed === 'null' || trimmed === 'undefined' ? '' : text;
}

function isVisible(message: AgentMessage): boolean {
  if (message.role === 'tool') return false;
  if (message.role !== 'assistant') return true;
  return Boolean(assistantText(message.text) || message.reasoning?.trim() || message.error?.trim());
}

export type TranscriptRow =
  | { kind: 'message'; key: string; message: AgentMessage }
  | { kind: 'tool'; key: string; entry: TimelineEntry };

/** Messages and tool calls in the order they happened; a message precedes a tool call of the same instant. */
export function transcriptRows(view: Pick<AgentView, 'messages' | 'timeline'>): TranscriptRow[] {
  const rows = [
    ...view.messages.flatMap((message, index) => isVisible(message)
      ? [{ row: { kind: 'message' as const, key: `m${index}`, message }, at: message.timestamp, order: 0, index }] : []),
    ...view.timeline.map((entry, index) => (
      { row: { kind: 'tool' as const, key: `t${entry.toolCallId}`, entry }, at: entry.timestamp, order: 1, index })),
  ];
  rows.sort((a, b) => a.at.localeCompare(b.at) || a.order - b.order || a.index - b.index);
  return rows.map(({ row }) => row);
}

const patch = (view: AgentView, toolCallId: string, change: Partial<TimelineEntry>): AgentView => ({
  ...view, timeline: view.timeline.map((entry) => entry.toolCallId === toolCallId ? { ...entry, ...change } : entry),
});

const stopped = (view: AgentView): AgentView => ({
  ...view, running: false, streaming: '', reasoning: '', approvals: [], progress: '',
  // Nothing is still running once the run ends, whatever the last event said.
  timeline: view.timeline.map((entry) => ['running', 'reviewing', 'awaiting-approval'].includes(entry.status)
    ? { ...entry, status: 'failed' as const, error: entry.error ?? 'Stopped before it finished.' } : entry),
});

/** Reduce one agent event into the view. */
export function applyEvent(view: AgentView, event: AgentEvent): AgentView {
  switch (event.type) {
    case 'run-started':
      return { ...view, running: true, summary: null, streaming: '', reasoning: '', progress: '', error: '' };
    case 'model-loading':
      return { ...view, progress: event.message };
    case 'prefill-progress':
      return { ...view, progress: event.total > 0
        ? `Reading the task… ${Math.min(100, Math.round(100 * (event.cached + event.processed) / event.total))}%` : '' };
    case 'text-delta':
      return event.channel === 'reasoning'
        ? { ...view, progress: '', reasoning: view.reasoning + event.delta }
        : { ...view, progress: '', streaming: view.streaming + event.delta };
    case 'tool-started':
      if (event.tool === 'spawn_subagent') return view;
      return {
        ...view, timeline: [...view.timeline.filter((entry) => entry.toolCallId !== event.toolCallId), {
          toolCallId: event.toolCallId, tool: event.tool, args: event.args, environment: event.environment,
          status: 'running', sandbox: event.sandbox, timestamp: event.timestamp,
        }],
      };
    case 'tool-output': {
      const entry = view.timeline.find((item) => item.toolCallId === event.toolCallId);
      return entry ? patch(view, event.toolCallId, { output: (entry.output ?? '') + event.chunk }) : view;
    }
    case 'approval-reviewed':
      return patch(view, event.toolCallId, event.phase === 'started'
        ? { status: 'reviewing', reviewNote: 'Checking this against your instructions…' }
        : event.phase === 'approved'
          ? { status: 'running', reviewNote: event.reason ? `Approved for you: ${event.reason}` : 'Approved for you.' }
          : { status: 'awaiting-approval', reviewNote: event.reason ? `Needs you: ${event.reason}` : 'Needs your decision.' });
    case 'approval-required': {
      const approvals = view.approvals.some((entry) => entry.id === event.approval.id)
        ? view.approvals : [...view.approvals, event.approval];
      return patch({ ...view, approvals }, event.toolCallId,
        { status: 'awaiting-approval', environment: event.approval.environment });
    }
    case 'approval-resolved':
      return { ...view, approvals: view.approvals.filter((entry) => entry.id !== event.approvalId) };
    case 'tool-completed':
      return patch(view, event.toolCallId, {
        status: 'completed', environment: event.environment, sandbox: event.sandbox, output: event.output,
        exitCode: event.exitCode, changedPaths: event.changedPaths, truncated: event.truncated, durationMs: event.durationMs,
      });
    case 'tool-failed':
      return patch(view, event.toolCallId, {
        status: event.denied ? 'denied' : 'failed', environment: event.environment, sandbox: event.sandbox,
        error: event.error, durationMs: event.durationMs,
      });
    case 'subagent-started':
      return {
        ...view, timeline: [...view.timeline.filter((entry) => entry.toolCallId !== event.childSessionId), {
          toolCallId: event.childSessionId, tool: 'subagent', args: { prompt: event.prompt }, environment: 'sandbox',
          status: 'running', timestamp: event.timestamp,
        }],
      };
    case 'subagent-progress':
      return event.activity.status === 'awaiting-approval'
        ? patch(view, event.childSessionId, { status: 'awaiting-approval' }) : view;
    case 'subagent-completed':
      return patch(view, event.childSessionId, event.status === 'completed'
        ? { status: 'completed', output: event.summary } : { status: 'failed', error: event.summary });
    case 'message-committed': {
      // The composer already showed the user's own turn.
      if (event.message.role === 'user') return view;
      const next = event.message.role === 'assistant' ? { ...view, streaming: '', reasoning: '' } : view;
      return isVisible(event.message) ? { ...next, messages: [...next.messages, event.message] } : next;
    }
    case 'run-completed':
      return { ...stopped(view), summary: event.summary };
    case 'run-cancelled':
      return stopped(view);
    case 'run-failed':
      return { ...stopped(view), error: agentErrorText(event.error) };
    case 'compacted':
      return {
        ...view, messages: [
          { role: 'system', text: event.state.summary, timestamp: event.state.compactedAt },
          ...(event.keptMessages ?? view.messages.slice(-6)),
        ],
      };
    default:
      return view;
  }
}

/** A short "what was done" line for the end of a run. */
export function summaryLine(summary: AgentRunSummary): string {
  const parts = [
    `${summary.toolCalls} tool call${summary.toolCalls === 1 ? '' : 's'}`,
    summary.filesChanged.length ? `${summary.filesChanged.length} file${summary.filesChanged.length === 1 ? '' : 's'} changed` : '',
    summary.failures.length ? `${summary.failures.length} failed` : '',
    summary.approvalsRequested ? `${summary.approvalsRequested} approval${summary.approvalsRequested === 1 ? '' : 's'}` : '',
  ];
  return parts.filter(Boolean).join(' · ');
}

export type ExecutionLocationLine = Pick<ExecutionLocation, 'daemon_display_name' | 'platform' | 'arch'>;
export const locationLine = (location: ExecutionLocationLine) =>
  `${location.daemon_display_name} · ${location.platform}/${location.arch}`;
