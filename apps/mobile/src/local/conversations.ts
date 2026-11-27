import type { Conversation, Message } from '../api.ts';

/** Named text files; the app keeps them in its documents directory. */
export type TextFiles = {
  read(name: string): Promise<string | null>;
  write(name: string, text: string): Promise<void>;
  remove(name: string): Promise<void>;
};
export type NewMessage = {
  parent_id: string | null; role: string; content: unknown; model?: string;
  tool_calls?: unknown[] | null; tool_call_id?: string | null;
};

const indexFile = 'conversations.json';
const messagesFile = (id: string) => `conversation-${id}.json`;
const validId = (id: unknown): id is string => typeof id === 'string' && /^[A-Za-z0-9-]{1,64}$/.test(id);

export function parseConversations(raw: string | null): Conversation[] {
  if (!raw) return [];
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return []; }
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  return value.flatMap((entry): Conversation[] => {
    const { id, title } = entry as Partial<Conversation>;
    if (!validId(id) || typeof title !== 'string' || seen.has(id)) return [];
    seen.add(id);
    return [{ id, title }];
  });
}

export function parseMessages(raw: string | null): Message[] {
  if (!raw) return [];
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return []; }
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry): Message[] => {
    const message = entry as Partial<Message>;
    if (typeof message?.id !== 'string' || typeof message.role !== 'string') return [];
    return [{
      id: message.id, role: message.role, content: message.content ?? '',
      parent_id: typeof message.parent_id === 'string' ? message.parent_id : null,
      ...(Array.isArray(message.tool_calls) && message.tool_calls.length ? { tool_calls: message.tool_calls } : {}),
      ...(typeof message.tool_call_id === 'string' ? { tool_call_id: message.tool_call_id } : {}),
    }];
  });
}

/**
 * Conversations kept on the phone, with the same calls Chat makes to a host.
 * The newest conversation is listed first, as hosts do.
 */
export function conversationStore(files: TextFiles, newId: () => string) {
  // Writes to one file are whole-file replacements, so they run one at a time.
  let queue: Promise<unknown> = Promise.resolve();
  const serial = <T>(work: () => Promise<T>): Promise<T> => {
    const result = queue.then(work, work);
    queue = result.catch(() => undefined);
    return result;
  };
  const list = async () => parseConversations(await files.read(indexFile));
  const messages = async (id: string) => validId(id) ? parseMessages(await files.read(messagesFile(id))) : [];
  const createMessage = (id: string, message: NewMessage): Promise<Message> => serial(async () => {
    if (!(await list()).some((chat) => chat.id === id)) throw new Error('This conversation is no longer on this device.');
    const saved: Message = {
      id: newId(), parent_id: message.parent_id, role: message.role, content: message.content,
      ...(message.tool_calls?.length ? { tool_calls: message.tool_calls } : {}),
      ...(message.tool_call_id ? { tool_call_id: message.tool_call_id } : {}),
    };
    await files.write(messagesFile(id), JSON.stringify([...await messages(id), saved]));
    return saved;
  });
  return {
    conversations: list,
    createConversation: (title: string): Promise<Conversation> => serial(async () => {
      const chat = { id: newId(), title: Array.from(title).slice(0, 72).join('') };
      await files.write(indexFile, JSON.stringify([chat, ...await list()]));
      return chat;
    }),
    messages,
    createMessage,
    async appendUserMessage(id: string, viewedBranch: readonly Message[], content: string) {
      const user = await createMessage(id, { parent_id: viewedBranch.at(-1)?.id ?? null, role: 'user', content });
      return { user, history: [...viewedBranch, user] };
    },
    deleteConversation: (id: string): Promise<void> => serial(async () => {
      await files.write(indexFile, JSON.stringify((await list()).filter((chat) => chat.id !== id)));
      if (validId(id)) await files.remove(messagesFile(id));
    }),
  };
}
