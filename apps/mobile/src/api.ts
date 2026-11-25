import { normalizeShareableOrigin, PAIRING_SCOPES, type PairingScope } from '../../../packages/pairing/src/invite.ts';

export type InviteInput = { origin: string; pairingId: string; code: string };
export type Credential = { origin: string; instanceId: string; apiKey: string };
export type Preview = {
  daemon: { instance_id: string; display_name: string };
  pairing: { id: string; client_name: string; scopes: PairingScope[]; expires_at: number };
};
export type Host = { displayName: string; scopes: PairingScope[] };
export type Model = { id: string; capabilities?: { output_modalities: string[] } };
export type Conversation = { id: string; title: string };
export type Message = { id: string; parent_id: string | null; role: string; content: unknown; status?: string };
export type OpenAiMessage = { role: string; content: string };

export class DaemonError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

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
      typeof value.apiKey !== 'string' || !value.apiKey) throw new Error('Invalid stored host credential.');
  return { origin: pairingOrigin(value.origin, allowAndroidEmulator), instanceId: value.instanceId, apiKey: value.apiKey };
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
    throw new DaemonError('Could not reach the host. Check the address, network, and HTTPS certificate.');
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

export async function claimInvite(fetch: Request, input: InviteInput, preview: Preview): Promise<Credential> {
  if (preview.pairing.expires_at <= Date.now() / 1000) throw new DaemonError('This invite expired. Ask for a fresh invite.');
  const data = await (await exchange(fetch, input, 'claim', preview.daemon.instance_id)).json() as { api_key?: string };
  if (typeof data?.api_key !== 'string' || !data.api_key) {
    throw new DaemonError('The host did not return a credential. It may have used the invitation; ask for a fresh invite.');
  }
  return { origin: input.origin, instanceId: preview.daemon.instance_id, apiKey: data.api_key };
}

export function daemonClient(fetch: Request, credential: Credential) {
  const json = async <T>(path: string, init: MobileRequestInit = {}): Promise<T> => {
    return (await request(fetch, credential.origin, path, init, credential.apiKey)).json() as Promise<T>;
  };
  const post = (body: object, signal?: AbortSignal) => ({ method: 'POST', body: JSON.stringify(body), signal });
  const messagesPath = (id: string) => `/api/v1/conversations/${encodeURIComponent(id)}/messages`;
  const createMessage = (id: string, message: {
    parent_id: string | null; role: 'user' | 'assistant'; content: string; model?: string;
  }) => json<Message>(messagesPath(id), post(message));
  const assertIdentity = (instanceId: unknown) => {
    if (instanceId !== credential.instanceId) {
      throw new DaemonError('The host identity changed. Verify its address and pair again.', 409);
    }
  };
  return {
    async verify(): Promise<Host> {
      const probe = await request(fetch, credential.origin,
        '/api/v1/auth/pairings/00000000-0000-0000-0000-000000000000/preview', post({ code: '' }));
      const identity = await probe.json() as { instance_id?: unknown } | null;
      if (typeof identity?.instance_id !== 'string' || !identity.instance_id) {
        throw new DaemonError('The host does not support public identity verification. Update Brazier on the host.');
      }
      assertIdentity(identity.instance_id);
      const info = await json<{
        product: string; management_api: { major: number };
        daemon: { instance_id: string; display_name: string };
        client: { scopes: PairingScope[] };
      }>('/api/v1/daemon/info');
      if (info?.product !== 'brazier' || info.management_api?.major !== 1 || !info.daemon?.display_name ||
          !scopesAreValid(info.client?.scopes)) throw new DaemonError('The address does not serve a compatible Brazier host.');
      assertIdentity(info.daemon.instance_id);
      if (!info.client.scopes.includes('inference')) throw new DaemonError('This device does not have inference access.', 403);
      return { displayName: info.daemon.display_name, scopes: info.client.scopes };
    },
    async models(): Promise<Model[]> {
      return (await json<{ data: Model[] }>('/v1/models')).data.filter((model) =>
        !model.capabilities || model.capabilities.output_modalities.includes('text'));
    },
    prepare: (modelId: string, signal?: AbortSignal) =>
      json('/api/v1/models/prepare', post({ model_id: modelId, mode: 'chat' }, signal)),
    async conversations(): Promise<Conversation[]> {
      return (await json<{ data: Conversation[] }>('/api/v1/conversations')).data;
    },
    createConversation: (title: string) => json<Conversation>('/api/v1/conversations', post({ title })),
    async messages(id: string): Promise<Message[]> {
      return (await json<{ data: Message[] }>(messagesPath(id))).data;
    },
    createMessage,
    async appendUserMessage(id: string, viewedBranch: readonly Message[], content: string) {
      const user = await createMessage(id, { parent_id: viewedBranch.at(-1)?.id ?? null, role: 'user', content });
      return { user, history: [...viewedBranch, user] };
    },
    async chat(model: string, messages: OpenAiMessage[], signal: AbortSignal, onToken: (token: string) => void): Promise<string> {
      const response = await request(fetch, credential.origin, '/v1/chat/completions', {
        ...post({ model, messages, stream: true }, signal),
        headers: { accept: 'text/event-stream', 'x-brazier-mode': 'chat' },
      }, credential.apiKey);
      if (!response.body) throw new DaemonError('The host did not return a chat stream.');
      return consumeChatStream(response.body, signal, onToken);
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

export function chatContext(messages: Message[]): OpenAiMessage[] {
  return messages.flatMap((message) => {
    const content = messageText(message.content);
    return content.trim() && ['system', 'user', 'assistant'].includes(message.role)
      ? [{ role: message.role, content }]
      : [];
  });
}

export async function consumeChatStream(
  body: ReadableStream<Uint8Array>, signal: AbortSignal, onToken: (token: string) => void,
): Promise<string> {
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
          choices?: { delta?: { content?: string } }[];
        };
        if (payload.error) throw new DaemonError(typeof payload.error === 'string' ? payload.error :
          payload.error.message ?? 'Chat generation failed.');
        const token = payload.choices?.[0]?.delta?.content;
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
