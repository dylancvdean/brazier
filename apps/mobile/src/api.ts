import { isCertificateSha256, normalizeShareableOrigin, PAIRING_SCOPES, type PairingScope } from '../../../packages/pairing/src/invite.ts';

export type InviteInput = { origin: string; pairingId: string; code: string; certificateSha256?: string };
export type Credential = { origin: string; instanceId: string; apiKey: string; certificateSha256?: string };
export type Preview = {
  daemon: { instance_id: string; display_name: string };
  pairing: { id: string; client_name: string; scopes: PairingScope[]; expires_at: number };
};
export type Host = { displayName: string; scopes: PairingScope[] };
export type Model = { id: string; engine?: string; owned_by?: string; capabilities?: { output_modalities: string[] } };
export type Conversation = { id: string; title: string };
export type Message = {
  id: string; parent_id: string | null; role: string; content: unknown; status?: string;
  tool_calls?: unknown[] | null; tool_call_id?: string | null;
};
export type OpenAiMessage = { role: string; content: string; tool_calls?: unknown[]; tool_call_id?: string };
export type Tool = { name: string; title: string; description?: string; network: boolean; source?: string; server_name?: string };
export type ToolCall = { call_id: string; name: string; arguments: string; output: string; is_error: boolean };
/** A message the host produced while running tools; saved so later turns keep the tool results. */
export type TranscriptMessage = { role: string; content: unknown; tool_calls?: unknown[] | null; tool_call_id?: string | null };
export type ChatEvents = {
  onToken: (token: string) => void;
  onToolCall?: (call: ToolCall) => void;
  onTranscript?: (message: TranscriptMessage) => void;
};

export class DaemonError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

/** The request never got an answer from the host. */
export class UnreachableError extends DaemonError {}

export function pairingOrigin(input: string, allowAndroidEmulator = false): string {
  if (allowAndroidEmulator) {
    const url = new URL(input.trim());
    if (url.protocol === 'http:' && url.hostname === '10.0.2.2' && url.pathname === '/' &&
        !url.username && !url.password && !url.search && !url.hash) return url.origin;
  }
  return normalizeShareableOrigin(input);
}

function scopesAreValid(scopes: unknown): scopes is PairingScope[] {
  return Array.isArray(scopes) && scopes.every((scope) => PAIRING_SCOPES.includes(scope));
}

export function parseCredential(raw: string, allowAndroidEmulator = false): Credential {
  const value = JSON.parse(raw) as Credential;
  if (typeof value?.origin !== 'string' || typeof value.instanceId !== 'string' || !value.instanceId ||
      typeof value.apiKey !== 'string' || !value.apiKey ||
      (value.certificateSha256 !== undefined && !isCertificateSha256(value.certificateSha256))) {
    throw new Error('Invalid stored host credential.');
  }
  return {
    origin: pairingOrigin(value.origin, allowAndroidEmulator), instanceId: value.instanceId, apiKey: value.apiKey,
    ...(value.certificateSha256 ? { certificateSha256: value.certificateSha256 } : {}),
  };
}

type MobileRequestInit = {
  method?: string; body?: string; headers?: HeadersInit; signal?: AbortSignal;
  credentials?: 'omit'; redirect?: 'error';
};
type Request = (url: string, init: MobileRequestInit) => Promise<Pick<Response, 'ok' | 'status' | 'json' | 'body'>>;

async function request(fetch: Request, origin: string, path: string, init: MobileRequestInit, key?: string): ReturnType<Request> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set('content-type', 'application/json');
  if (key) headers.set('authorization', `Bearer ${key}`);
  let response: Awaited<ReturnType<Request>>;
  try {
    response = await fetch(origin + path, { ...init, headers, credentials: 'omit', redirect: 'error' });
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new UnreachableError('Could not reach the host. Check the address, network, and HTTPS certificate.');
  }
  if (response.ok) return response;
  const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
  if (response.status === 401 && key) throw new DaemonError('This device is no longer authorized. Pair again.', 401);
  if (response.status === 403) throw new DaemonError('This device does not have permission for this action.', 403);
  throw new DaemonError(body?.error?.message ?? `The host returned HTTP ${response.status}.`, response.status);
}

async function exchange(fetch: Request, input: InviteInput, action: 'preview' | 'claim', instanceId?: string): ReturnType<Request> {
  try {
    return await request(fetch, input.origin, `/api/v1/auth/pairings/${encodeURIComponent(input.pairingId)}/${action}`, {
      method: 'POST',
      body: JSON.stringify({ code: input.code, expected_daemon_instance_id: instanceId }),
    });
  } catch (error) {
    if (error instanceof DaemonError && error.status === 400) {
      throw new DaemonError('The invite expired, was already used or canceled, or has an incorrect code. Ask for a fresh invite.', 400);
    }
    if (error instanceof DaemonError && error.status === 409) {
      throw new DaemonError('The host changed after review. Ask for a fresh invite.', 409);
    }
    throw error;
  }
}

export async function previewInvite(fetch: Request, input: InviteInput): Promise<Preview> {
  const data = await (await exchange(fetch, input, 'preview')).json() as Preview;
  if (!data?.daemon?.instance_id || !data.daemon.display_name || data.pairing?.id !== input.pairingId ||
      !data.pairing.client_name || !scopesAreValid(data.pairing.scopes) || !Number.isSafeInteger(data.pairing.expires_at)) {
    throw new DaemonError('The host returned an invalid pairing preview.');
  }
  if (data.pairing.expires_at <= Date.now() / 1000) throw new DaemonError('This invite expired. Ask for a fresh invite.');
  if (!data.pairing.scopes.includes('inference')) throw new DaemonError('Chat needs an invitation with inference access.');
  return data;
}

async function verifyPublicIdentity(fetch: Request, origin: string, instanceId: string): Promise<void> {
  const updateHost = 'The host does not support public identity verification. Update Brazier on the host.';
  const probe = await request(fetch, origin,
    '/api/v1/auth/pairings/00000000-0000-0000-0000-000000000000/preview', {
      method: 'POST', body: JSON.stringify({ code: '' }),
    }).catch((error) => {
    if (error instanceof DaemonError && [400, 404, 405].includes(error.status ?? 0)) {
      throw new DaemonError(updateHost);
    }
    throw error;
  });
  const identity = await probe.json().catch(() => null) as { instance_id?: unknown } | null;
  if (typeof identity?.instance_id !== 'string' || !identity.instance_id) throw new DaemonError(updateHost);
  assertIdentity(identity.instance_id, instanceId);
}

function assertIdentity(actual: unknown, expected: string): void {
  if (actual !== expected) {
    throw new DaemonError('The host identity changed. Verify its address and pair again.', 409);
  }
}

export async function claimInvite(fetch: Request, input: InviteInput, preview: Preview): Promise<Credential> {
  if (preview.pairing.expires_at <= Date.now() / 1000) throw new DaemonError('This invite expired. Ask for a fresh invite.');
  await verifyPublicIdentity(fetch, input.origin, preview.daemon.instance_id);
  const data = await (await exchange(fetch, input, 'claim', preview.daemon.instance_id)).json() as { api_key?: string };
  if (typeof data?.api_key !== 'string' || !data.api_key) {
    throw new DaemonError('The host did not return a credential. It may have used the invitation; ask for a fresh invite.');
  }
  return {
    origin: input.origin, instanceId: preview.daemon.instance_id, apiKey: data.api_key,
    ...(input.certificateSha256 ? { certificateSha256: input.certificateSha256 } : {}),
  };
}

function isChatModel(model: Model): boolean {
  const engine = model.engine || model.owned_by?.replace(/^brazier:/, '') || '';
  return !['whisper.cpp', 'streaming-asr', 'stable-diffusion.cpp', 'personaplex'].includes(engine) &&
    !['sdcpp-image:', 'sdcpp-video:', 'personaplex:', 'whisper:', 'streaming-asr:']
      .some((prefix) => model.id.startsWith(prefix)) &&
    (!model.capabilities || model.capabilities.output_modalities.includes('text'));
}

export function daemonClient(fetch: Request, credential: Credential) {
  const json = async <T>(path: string, init: MobileRequestInit = {}): Promise<T> => {
    return (await request(fetch, credential.origin, path, init, credential.apiKey)).json() as Promise<T>;
  };
  const post = (body: object, signal?: AbortSignal) => ({ method: 'POST', body: JSON.stringify(body), signal });
  const messagesPath = (id: string) => `/api/v1/conversations/${encodeURIComponent(id)}/messages`;
  const createMessage = (id: string, message: {
    parent_id: string | null; role: string; content: unknown; model?: string;
    tool_calls?: unknown[] | null; tool_call_id?: string | null;
  }) => json<Message>(messagesPath(id), post(message));
  return {
    async verify(): Promise<Host> {
      await verifyPublicIdentity(fetch, credential.origin, credential.instanceId);
      const info = await json<{
        product: string; management_api: { major: number };
        daemon: { instance_id: string; display_name: string };
        client: { scopes: PairingScope[] };
      }>('/api/v1/daemon/info');
      if (info?.product !== 'brazier' || info.management_api?.major !== 1 || !info.daemon?.display_name ||
          !scopesAreValid(info.client?.scopes)) throw new DaemonError('The address does not serve a compatible Brazier host.');
      assertIdentity(info.daemon.instance_id, credential.instanceId);
      if (!info.client.scopes.includes('inference')) throw new DaemonError('This device does not have inference access.', 403);
      return { displayName: info.daemon.display_name, scopes: info.client.scopes };
    },
    async models(): Promise<Model[]> {
      return (await json<{ data: Model[] }>('/v1/models')).data.filter(isChatModel);
    },
    prepare: (modelId: string, signal?: AbortSignal) =>
      json('/api/v1/models/prepare', post({ model_id: modelId, mode: 'chat' }, signal)),
    async conversations(): Promise<Conversation[]> {
      return (await json<{ data: Conversation[] }>('/api/v1/conversations')).data;
    },
    createConversation: (title: string) => json<Conversation>(
      '/api/v1/conversations', post({ title: Array.from(title).slice(0, 72).join('') })),
    async messages(id: string): Promise<Message[]> {
      return (await json<{ data: Message[] }>(messagesPath(id))).data;
    },
    createMessage,
    async appendUserMessage(id: string, viewedBranch: readonly Message[], content: string) {
      const user = await createMessage(id, { parent_id: viewedBranch.at(-1)?.id ?? null, role: 'user', content });
      return { user, history: [...viewedBranch, user] };
    },
    async tools(): Promise<Tool[]> {
      return (await json<{ data: Tool[] }>('/api/v1/tools')).data;
    },
    /** `tools` are the names the host may run for the model; omit or pass none for a plain chat. */
    async chat(
      model: string, messages: OpenAiMessage[], signal: AbortSignal, events: ChatEvents | ChatEvents['onToken'],
      tools: readonly string[] = [],
    ): Promise<string> {
      const response = await request(fetch, credential.origin, '/v1/chat/completions', {
        ...post({
          model, messages, stream: true,
          ...(tools.length ? { builtin_tools: true, builtin_tool_names: tools } : {}),
        }, signal),
        headers: { accept: 'text/event-stream', 'x-brazier-mode': 'chat' },
      }, credential.apiKey);
      if (!response.body) throw new DaemonError('The host did not return a chat stream.');
      return consumeChatStream(response.body, signal, events);
    },
  };
}

export function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((part) => part?.type === 'text' && typeof part.text === 'string' ? [part.text] : []).join('\n');
}

export function chatBranch(messages: Message[]): Message[] {
  const visible = messages.filter((message) => !['queued', 'cancelled', 'superseded'].includes(message.status ?? ''));
  const byId = new Map(messages.map((message) => [message.id, message]));
  const branch: Message[] = [];
  const seen = new Set<string>();
  let next = visible.at(-1);
  while (next && !seen.has(next.id)) {
    seen.add(next.id);
    branch.push(next);
    next = next.parent_id ? byId.get(next.parent_id) : undefined;
  }
  return branch.reverse();
}

const wireContent = (content: unknown) => typeof content === 'string' ? content : JSON.stringify(content ?? '');

export function chatContext(messages: Message[]): OpenAiMessage[] {
  return messages.flatMap((message): OpenAiMessage[] => {
    if (message.role === 'tool') {
      return message.tool_call_id
        ? [{ role: 'tool', tool_call_id: message.tool_call_id, content: wireContent(message.content) }] : [];
    }
    if (message.role === 'assistant' && message.tool_calls?.length) {
      return [{ role: 'assistant', content: wireContent(message.content), tool_calls: message.tool_calls }];
    }
    const content = messageText(message.content);
    return content.trim() && ['system', 'user', 'assistant'].includes(message.role)
      ? [{ role: message.role, content }]
      : [];
  });
}

/** Tool calls the model made in the saved messages, with each result filled in from its tool message. */
export function toolCallsOf(messages: readonly Message[]): Map<string, ToolCall[]> {
  const byMessage = new Map<string, ToolCall[]>();
  const byCall = new Map<string, ToolCall>();
  for (const message of messages) {
    if (message.role === 'assistant' && message.tool_calls?.length) {
      const calls = message.tool_calls.flatMap((value, index): ToolCall[] => {
        const call = value as { id?: unknown; function?: { name?: unknown; arguments?: unknown } };
        if (typeof call?.function?.name !== 'string') return [];
        const args = call.function.arguments;
        return [{
          call_id: typeof call.id === 'string' ? call.id : `${message.id}-${index}`, name: call.function.name,
          arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}), output: '', is_error: false,
        }];
      });
      for (const call of calls) byCall.set(call.call_id, call);
      byMessage.set(message.id, calls);
    } else if (message.role === 'tool' && message.tool_call_id) {
      const call = byCall.get(message.tool_call_id);
      if (call) call.output = messageText(message.content) || wireContent(message.content);
    }
  }
  return byMessage;
}

export async function consumeChatStream(
  body: ReadableStream<Uint8Array>, signal: AbortSignal, events: ChatEvents | ChatEvents['onToken'],
): Promise<string> {
  const { onToken, onToolCall, onTranscript } = typeof events === 'function' ? { onToken: events } as ChatEvents : events;
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let result = '';
  let buffer = '';
  let completed = false;
  const cancel = () => { void reader.cancel().catch(() => undefined); };
  const checkAbort = () => {
    if (signal.aborted) throw new DaemonError('Chat stopped.');
  };
  signal.addEventListener('abort', cancel, { once: true });
  try {
    for (;;) {
      checkAbort();
      const chunk = await reader.read();
      checkAbort();
      if (chunk.done) throw new DaemonError('The chat stream closed before finishing.');
      buffer += decoder.decode(chunk.value, { stream: true });
      let separator: RegExpExecArray | null;
      while ((separator = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator[0].length);
        const data = frame.split(/\r?\n/).filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart()).join('\n');
        if (!data) continue;
        if (data === '[DONE]') {
          completed = true;
          return result;
        }
        const payload = JSON.parse(data) as {
          error?: { message?: string } | string;
          choices?: { delta?: { content?: string; tool_calls?: { function?: { name?: string } }[] }; finish_reason?: string | null }[];
          brazier?: { tool_call?: ToolCall; transcript_message?: TranscriptMessage };
        };
        if (payload.error) throw new DaemonError(typeof payload.error === 'string' ? payload.error :
          payload.error.message ?? 'Chat generation failed.');
        if (payload.brazier?.tool_call) onToolCall?.(payload.brazier.tool_call);
        if (payload.brazier?.transcript_message) onTranscript?.(payload.brazier.transcript_message);
        const choice = payload.choices?.[0];
        if (choice?.finish_reason === 'tool_calls' && choice.delta?.tool_calls?.length) {
          const names = choice.delta.tool_calls.map((call) => call.function?.name ?? 'unknown').join(', ');
          throw new DaemonError(`The model asked for tools this app cannot run: ${names}.`);
        }
        const token = choice?.delta?.content;
        if (token) {
          result += token;
          onToken(token);
          checkAbort();
        }
      }
    }
  } finally {
    signal.removeEventListener('abort', cancel);
    if (!completed && !signal.aborted) cancel();
    reader.releaseLock();
  }
}
