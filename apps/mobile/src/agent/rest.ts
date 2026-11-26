/** Agent routes of a paired host. Every one of them needs the key's `agent` scope. */
import { DaemonError, UnreachableError, type Credential } from '../api.ts';
import {
  agentErrorText, type AgentApproval, type AgentCapabilities, type AgentPermissionMode, type AgentSessionDetail,
  type AgentSessionSummary, type AgentTool,
} from './state.ts';

type Init = { method?: string; body?: string; headers?: Record<string, string>; credentials?: 'omit'; redirect?: 'error' };
export type AgentFetch = (url: string, init: Init) => Promise<Pick<Response, 'ok' | 'status' | 'json'>>;

export type SessionDraft = {
  title: string; workspace_path: string; model: string; permission_mode: AgentPermissionMode;
  enabled_tools?: string[];
};

export function agentApi(fetch: AgentFetch, credential: Credential) {
  async function call<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
    let response: Awaited<ReturnType<AgentFetch>>;
    try {
      response = await fetch(credential.origin + path, {
        method, credentials: 'omit', redirect: 'error',
        headers: { authorization: `Bearer ${credential.apiKey}`, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
    } catch {
      throw new UnreachableError('Could not reach the host. Check the address, network, and HTTPS certificate.');
    }
    if (response.ok) return (response.status === 204 ? undefined : await response.json()) as T;
    const payload = await response.json().catch(() => null) as { error?: { message?: string } } | null;
    if (response.status === 401) throw new DaemonError('This device is no longer authorized. Pair again.', 401);
    throw new DaemonError(
      agentErrorText(payload?.error?.message ?? `The host returned HTTP ${response.status}.`), response.status);
  }

  return {
    capabilities: () => call<AgentCapabilities>('/api/v1/agent/capabilities'),
    tools: async () => (await call<{ data: AgentTool[] }>('/api/v1/agent/tools')).data,
    /** Only the sessions this device started: the host hides everyone else's from a paired key. */
    sessions: async () => (await call<{ data: AgentSessionSummary[] }>('/api/v1/agent/sessions')).data,
    session: (id: string) => call<AgentSessionDetail>(`/api/v1/agent/sessions/${encodeURIComponent(id)}`),
    /**
     * Never sends `confirm_elevated_permissions` or asks for host auto-approval: those need proof
     * from the host's own desktop, which a paired device cannot give.
     */
    createSession: (draft: SessionDraft) => call<AgentSessionSummary>('/api/v1/agent/sessions', 'POST', {
      ...draft, permission_settings: { auto_approve_sandboxed_actions: true, auto_approve_host_actions: false },
    }),
    setWorkspace: (id: string, workspace_path: string) =>
      call<AgentSessionSummary>(`/api/v1/agent/sessions/${encodeURIComponent(id)}`, 'PATCH', { workspace_path }),
    deleteSession: (id: string) => call<void>(`/api/v1/agent/sessions/${encodeURIComponent(id)}`, 'DELETE'),
    /** Checks the folder exists on the host and that its owner has opened it to paired devices. */
    decide: (approval: AgentApproval, decision: 'approve' | 'deny', scope?: 'once' | 'session') =>
      call<AgentApproval>(`/api/v1/agent/approvals/${encodeURIComponent(approval.id)}`, 'POST', {
        decision, scope, expected_execution_location: approval.execution_location,
      }),
  };
}

export type AgentApi = ReturnType<typeof agentApi>;
