import { DaemonError, type ChatEvents, type Conversation, type Message, type Model, type OpenAiMessage, type Tool } from '../api.ts';
import type { ToolResult } from './tools.ts';
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
  /** On this device: whether the model's runtime can call tools. A host decides this itself. */
  supportsTools?(modelId: string): Promise<boolean>;
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
  /**
   * One round of a reply that may call tools: the text written and the calls asked for. Runtimes without
   * it chat without tools.
   */
  generateWithTools?(
    messages: ToolTurn[], tools: readonly object[], signal: AbortSignal, onToken: (token: string) => void,
  ): Promise<{ content: string; calls: RequestedCall[] }>;
  unload(): Promise<void>;
};

export type RequestedCall = { id?: string; name: string; arguments: string };
type WireCall = { id: string; type: 'function'; function: { name: string; arguments: string } };
/** A message of a conversation that keeps its tool rounds. */
export type ToolTurn =
  | ChatTurn
  | { role: 'assistant'; content: string; tool_calls: WireCall[] }
  | { role: 'tool'; content: string; tool_call_id: string };

/** The tools a chat on this device can run, and how to run one. */
export type LocalTools = {
  definitions(names: readonly string[]): object[];
  run(name: string, args: string, signal: AbortSignal): Promise<ToolResult>;
};

/** Rounds of tool calls in one reply; the last round is asked without tools so it ends in an answer. */
export const MAX_TOOL_ROUNDS = 4;
/** A result this long would crowd the question out of a phone model's context. */
const MAX_RESULT_CHARS = 4_000;

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
 * The conversation for a runtime that calls tools. A tool round is kept only when it is whole (every call
 * has its result), because chat templates reject a call without one; a stopped reply can leave such a round.
 */
export function toolTurns(messages: readonly OpenAiMessage[]): ToolTurn[] {
  const turns: ToolTurn[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message.role === 'assistant' && message.tool_calls?.length) {
      const calls = message.tool_calls.flatMap((value): WireCall[] => {
        const call = value as { id?: unknown; function?: { name?: unknown; arguments?: unknown } };
        if (typeof call?.id !== 'string' || typeof call.function?.name !== 'string') return [];
        const args = call.function.arguments;
        return [{ id: call.id, type: 'function', function: {
          name: call.function.name, arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}),
        } }];
      });
      const results: ToolTurn[] = [];
      while (messages[index + 1]?.role === 'tool') {
        index += 1;
        const result = messages[index];
        if (calls.some(({ id }) => id === result.tool_call_id)) {
          results.push({ role: 'tool', content: result.content, tool_call_id: result.tool_call_id! });
        }
      }
      if (calls.length && calls.length === message.tool_calls.length && results.length === calls.length) {
        turns.push({ role: 'assistant', content: message.content, tool_calls: calls }, ...results);
      } else if (message.content.trim()) {
        turns.push({ role: 'assistant', content: message.content });
      }
      continue;
    }
    if ((message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') || !message.content.trim()) continue;
    const last = turns.at(-1);
    if (last?.role === message.role && !('tool_calls' in last)) last.content += `\n\n${message.content}`;
    else turns.push({ role: message.role, content: message.content });
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

export function localEngine(
  runtimes: readonly Runtime[], installed: () => Promise<InstalledModel[]>, localTools?: LocalTools,
): LocalEngine & { supportsTools(modelId: string): Promise<boolean> } {
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
    async supportsTools(modelId) {
      return !!localTools && !!(await resolve(modelId).catch(() => null))?.runtime.generateWithTools;
    },
    async chat(model, messages, signal, events, tools = []) {
      const { onToken, onToolCall, onTranscript } = typeof events === 'function' ? { onToken: events } as ChatEvents : events;
      const stopped = () => { if (signal.aborted) throw new DaemonError('Chat stopped.'); };
      await prepare(model, signal);
      stopped();
      const current = loaded;
      if (current?.id !== model) throw new DaemonError('The model was changed before it could reply.');
      const { runtime } = current;
      const definitions = localTools && runtime.generateWithTools ? localTools.definitions(tools) : [];
      if (!definitions.length) {
        const answer = await runtime.generate(chatTurns(messages), signal, onToken);
        stopped();
        return answer;
      }
      const turns = toolTurns(messages);
      let made = 0;
      for (let round = 1; ; round += 1) {
        const { content, calls } = await runtime.generateWithTools!(
          turns, round < MAX_TOOL_ROUNDS ? definitions : [], signal, onToken);
        stopped();
        if (!calls.length || round === MAX_TOOL_ROUNDS) return content;
        const wire = calls.map((call): WireCall => ({
          // Ids are made unique within the conversation: some models reuse `call_0` in every round.
          id: `call-${Date.now().toString(36)}-${made += 1}`, type: 'function',
          function: { name: call.name, arguments: call.arguments || '{}' },
        }));
        turns.push({ role: 'assistant', content, tool_calls: wire });
        onTranscript?.({ role: 'assistant', content, tool_calls: wire });
        for (const call of wire) {
          const { name, arguments: args } = call.function;
          const result = tools.includes(name) ? await localTools!.run(name, args, signal)
            : { output: `Error: the tool ${name} is not available.`, is_error: true };
          stopped();
          const output = result.output.length > MAX_RESULT_CHARS
            ? `${result.output.slice(0, MAX_RESULT_CHARS)}\n[Cut short.]` : result.output;
          turns.push({ role: 'tool', content: output, tool_call_id: call.id });
          onToolCall?.({ call_id: call.id, name, arguments: args, output, is_error: result.is_error });
          onTranscript?.({ role: 'tool', content: output, tool_call_id: call.id });
        }
      }
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
