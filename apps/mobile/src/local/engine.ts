import { DaemonError, type ChatEvents, type Conversation, type Message, type Model, type OpenAiMessage, type Tool } from '../api.ts';
import type { NewMessage } from './conversations.ts';
import type { RuntimeId } from './huggingface.ts';
import type { InstalledModel } from './models.ts';

/** What Chat needs to produce a reply. A paired host and this device both provide it. */
export type ChatEngine = {
  models(): Promise<Model[]>;
  /** Load the model so the first reply does not wait for it. */
  prepare(modelId: string, signal?: AbortSignal): Promise<unknown>;
  /** Streams the reply through `events`; aborting `signal` stops it. */
  chat(
    model: string, messages: OpenAiMessage[], signal: AbortSignal, events: ChatEvents | ChatEvents['onToken'],
    tools?: readonly string[],
  ): Promise<string>;
};

/** The engine plus where its conversations are kept: the host, or this device. */
export type ChatBackend = ChatEngine & {
  conversations(): Promise<Conversation[]>;
  createConversation(title: string): Promise<Conversation>;
  messages(id: string): Promise<Message[]>;
  createMessage(id: string, message: NewMessage): Promise<Message>;
  appendUserMessage(id: string, viewedBranch: readonly Message[], content: string):
    Promise<{ user: Message; history: Message[] }>;
  tools(): Promise<Tool[]>;
  /** On this device: a reason to ask before loading the model, such as one too large for memory. */
  loadWarning?(modelId: string): Promise<string | null>;
};

export type ChatTurn = { role: 'system' | 'user' | 'assistant'; content: string };

/** One way of running a model on this device. */
export type Runtime = {
  id: RuntimeId;
  /** False when this phone, OS version or build cannot run it; its models are then not offered. */
  available(): Promise<boolean>;
  /** Models that need no download, such as the system model. */
  builtIn?(): Promise<Model[]>;
  /** `model` is null for a built-in model. */
  load(model: InstalledModel | null, signal?: AbortSignal): Promise<void>;
  generate(messages: ChatTurn[], signal: AbortSignal, onToken: (token: string) => void): Promise<string>;
  unload(): Promise<void>;
};

export async function availableRuntimes(runtimes: readonly Runtime[]): Promise<Runtime[]> {
  const checks = await Promise.all(runtimes.map((runtime) => runtime.available().catch(() => false)));
  return runtimes.filter((_, index) => checks[index]);
}

/**
 * On-device chat is text only: tool rounds and empty messages are left out. Neighbouring turns from the same
 * side become one, because a stopped reply leaves two user messages in a row and many chat templates
 * require the sides to alternate.
 */
export function chatTurns(messages: readonly OpenAiMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const message of messages) {
    const { role } = message;
    if ((role !== 'system' && role !== 'user' && role !== 'assistant') || message.tool_calls?.length ||
        !message.content.trim()) continue;
    const last = turns.at(-1);
    if (last?.role === role) last.content += `\n\n${message.content}`;
    else turns.push({ role, content: message.content });
  }
  return turns;
}

/**
 * Text to show for a streamed llama.cpp token. When the runtime parses the reply it reports the answer so far
 * without the model's reasoning, so only what was added to that answer is new.
 */
export function answerDelta(shown: string, data: { token: string; content?: string; reasoning_content?: string; accumulated_text?: string }): string {
  const parsed = data.content !== undefined || data.reasoning_content !== undefined || data.accumulated_text !== undefined;
  if (!parsed) return data.token;
  const content = data.content ?? '';
  return content.startsWith(shown) ? content.slice(shown.length) : '';
}

/** The local engine: picks the runtime for a model and keeps at most one model loaded. */
export type LocalEngine = ChatEngine & {
  /** Frees the loaded model; with an id, only if that model is the one loaded. */
  unload(modelId?: string): Promise<void>;
};

export function localEngine(runtimes: readonly Runtime[], installed: () => Promise<InstalledModel[]>): LocalEngine {
  let loaded: { id: string; runtime: Runtime } | null = null;
  let loading: Promise<unknown> = Promise.resolve();

  async function resolve(modelId: string): Promise<{ runtime: Runtime; model: InstalledModel | null }> {
    const usable = await availableRuntimes(runtimes);
    for (const runtime of usable) {
      if ((await runtime.builtIn?.())?.some(({ id }) => id === modelId)) return { runtime, model: null };
    }
    const model = (await installed()).find(({ id }) => id === modelId);
    if (!model) throw new DaemonError('This model is not on this device. Add it in Models.');
    const runtime = usable.find(({ id }) => id === model.runtime);
    if (!runtime) throw new DaemonError(`This device cannot run ${model.runtime} models.`);
    return { runtime, model };
  }

  function prepare(modelId: string, signal?: AbortSignal): Promise<void> {
    const next = loading.catch(() => undefined).then(async () => {
      if (loaded?.id === modelId) return;
      const { runtime, model } = await resolve(modelId);
      // Phones rarely have memory for two models, so the previous one goes first.
      const previous = loaded;
      loaded = null;
      await previous?.runtime.unload().catch(() => undefined);
      await runtime.load(model, signal);
      loaded = { id: modelId, runtime };
    });
    loading = next;
    return next;
  }

  return {
    async models() {
      const usable = await availableRuntimes(runtimes);
      const builtIn = (await Promise.all(usable.map((runtime) => runtime.builtIn?.() ?? []))).flat();
      const downloaded = (await installed()).filter((model) => usable.some(({ id }) => id === model.runtime))
        .map((model): Model => ({ id: model.id, name: model.name, engine: model.runtime }));
      return [...builtIn, ...downloaded];
    },
    prepare,
    async chat(model, messages, signal, events) {
      const onToken = typeof events === 'function' ? events : events.onToken;
      await prepare(model, signal);
      if (signal.aborted) throw new DaemonError('Chat stopped.');
      const current = loaded;
      if (current?.id !== model) throw new DaemonError('The model was changed before it could reply.');
      const answer = await current.runtime.generate(chatTurns(messages), signal, onToken);
      if (signal.aborted) throw new DaemonError('Chat stopped.');
      return answer;
    },
    unload(modelId) {
      const next = loading.catch(() => undefined).then(async () => {
        if (!loaded || (modelId && loaded.id !== modelId)) return;
        const previous = loaded;
        loaded = null;
        await previous.runtime.unload();
      });
      loading = next;
      return next;
    },
  };
}
