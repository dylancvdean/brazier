/**
 * Runs the desktop's agent core inside the app. The core is the same code the desktop runs in a
 * utility process; here it runs on the JS thread and talks to the paired host with the app's
 * streaming, certificate-pinned `fetch`.
 */
import type { WorkerCommandInput, WorkerMessage } from '../../../desktop/src/agent/core/protocol';
import type { AgentEvent, AgentPermissionMode } from '../../../desktop/src/agent/core/types';
import { AgentWorkerCore } from '../../../desktop/src/agent/workerCore';
import type { Credential } from '../api.ts';

type NativeFetch = (url: string, init?: {
  method?: string; body?: string; headers?: Record<string, string>; signal?: AbortSignal;
  credentials?: 'omit'; redirect?: 'error';
}) => Promise<unknown>;

export class AgentController {
  private readonly core: AgentWorkerCore;
  private readonly pending = new Map<string, { resolve: (data: unknown) => void; reject: (error: Error) => void }>();
  private ready: Promise<unknown> | null = null;
  private next = 0;

  constructor(nativeFetch: NativeFetch, private readonly credential: Credential,
    private readonly onEvent: (sessionId: string, event: AgentEvent) => void) {
    const http = ((input: RequestInfo | URL, init?: RequestInit) => nativeFetch(String(input), {
      method: init?.method, signal: init?.signal ?? undefined,
      body: typeof init?.body === 'string' ? init.body : undefined,
      headers: Object.fromEntries(new Headers(init?.headers).entries()),
      credentials: 'omit', redirect: 'error',
    })) as typeof fetch;
    this.core = new AgentWorkerCore((message) => this.receive(message), undefined, http);
  }

  private receive(message: WorkerMessage) {
    if (message.type === 'event') this.onEvent(message.sessionId, message.event);
    if (message.type !== 'result') return;
    const waiter = this.pending.get(message.requestId);
    this.pending.delete(message.requestId);
    if (message.ok) waiter?.resolve(message.data);
    else waiter?.reject(new Error(message.error));
  }

  private send(command: WorkerCommandInput): Promise<unknown> {
    const requestId = `m${this.next += 1}`;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      void this.core.handle({ ...command, requestId } as Parameters<AgentWorkerCore['handle']>[0]);
    });
  }

  private async command(command: WorkerCommandInput): Promise<unknown> {
    this.ready ??= this.send({
      type: 'init', connection: { address: this.credential.origin, apiKey: this.credential.apiKey },
    }).catch((error) => { this.ready = null; throw error; });
    await this.ready;
    return this.send(command);
  }

  /** Resolves when the run ends; its progress arrives through `onEvent`. */
  run = (sessionId: string, text: string) => this.command({ type: 'run', sessionId, input: { text } });
  /** Reload the session from the host, e.g. after its workspace changed. */
  open = (sessionId: string) => this.command({ type: 'open-session', sessionId });
  cancel = (sessionId: string) => this.command({ type: 'cancel', sessionId });
  setModel = (sessionId: string, id: string) => this.command({ type: 'set-model', sessionId, model: { id } });
  setTools = (sessionId: string, tools: string[]) => this.command({ type: 'set-tools', sessionId, tools });
  setPermissionMode = (sessionId: string, mode: AgentPermissionMode) =>
    this.command({ type: 'set-permission-mode', sessionId, mode });
  close = (sessionId: string) => this.command({ type: 'close-session', sessionId });
  dispose = () => { void this.send({ type: 'shutdown' }).catch(() => undefined); };
}
