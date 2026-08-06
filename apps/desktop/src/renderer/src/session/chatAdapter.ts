/**
 * Chat adapter: the shared conversation, persisted by the daemon.
 *
 * There is no second conversation store. Every message the coordinator records
 * — typed, spoken, or agent-authored — goes into the same SQLite message graph
 * the chat UI already reads, with `source`, `correlation_id`, and `status`
 * saying where it came from and what became of it.
 */

import { createMessage, updateMessage } from '../api'
import type { ContentPart, Message, Role } from '../types'
import type { ChatAdapter } from './adapters'
import type {
  ConversationMessage,
  MessagePatch,
  MessageSource,
  MessageStatus,
  NewMessage
} from './types'

export type ChatAdapterHooks = {
  /** Show a message in the transcript as soon as it is stored. */
  onMessage?: (message: Message) => void
  /** Transient status line, or null to clear it. */
  onStatus?: (status: string | null) => void
  /** Parent for the next message; the chat UI owns branch selection. */
  parentId?: () => string | null
  /** Model to attribute assistant messages to. */
  model?: () => string | undefined
}

function contentToText(content: string | ContentPart[]): string {
  if (typeof content === 'string') return content
  return content
    .filter((part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

const SOURCES: readonly MessageSource[] = [
  'user_text',
  'user_voice',
  'assistant_chat',
  'assistant_agent',
  'assistant_voice',
  'tool',
  'system'
]

const STATUSES: readonly MessageStatus[] = [
  'partial',
  'final',
  'cancelled',
  'superseded',
  'failed'
]

/** Normalize a stored daemon message into the shared conversation shape. */
export function toConversationMessage(message: Message): ConversationMessage {
  const source = SOURCES.find((candidate) => candidate === message.source)
  const status = STATUSES.find((candidate) => candidate === message.status)
  return {
    id: message.id,
    conversationId: message.conversation_id,
    role: message.role as ConversationMessage['role'],
    // Messages written before the integration, and by the plain chat path, have
    // no source recorded; infer the obvious one rather than inventing a label.
    source: source ?? defaultSource(message.role),
    content: contentToText(message.content),
    createdAt: message.created_at,
    correlationId: message.correlation_id ?? undefined,
    status: status ?? 'final',
    metadata: message.metadata ?? undefined
  }
}

function defaultSource(role: Role): MessageSource {
  if (role === 'user') return 'user_text'
  if (role === 'assistant') return 'assistant_chat'
  if (role === 'tool') return 'tool'
  return 'system'
}

export class DaemonChatAdapter implements ChatAdapter {
  private lastMessageId: string | null = null

  constructor(
    private readonly conversationId: string,
    private readonly hooks: ChatAdapterHooks = {}
  ) {}

  async appendMessage(message: NewMessage): Promise<ConversationMessage> {
    const parentId = this.hooks.parentId?.() ?? this.lastMessageId
    const stored = await createMessage(this.conversationId, {
      parent_id: parentId,
      role: message.role,
      content: message.content,
      model: message.role === 'assistant' ? this.hooks.model?.() : undefined,
      source: message.source,
      correlation_id: message.correlationId,
      status: message.status,
      metadata: message.metadata
    })
    this.lastMessageId = stored.id
    this.hooks.onMessage?.(stored)
    return toConversationMessage(stored)
  }

  async updateMessage(messageId: string, patch: MessagePatch): Promise<ConversationMessage> {
    const stored = await updateMessage(this.conversationId, messageId, {
      content: patch.content,
      status: patch.status,
      metadata: patch.metadata
    })
    this.hooks.onMessage?.(stored)
    return toConversationMessage(stored)
  }

  showStatus(status: string | null): void {
    this.hooks.onStatus?.(status)
  }

  markQueued(messageId: string): void {
    this.hooks.onStatus?.('Queued behind the turn already running.')
    void messageId
  }

  markCancelled(messageId: string): void {
    this.hooks.onStatus?.('Cancelled.')
    void messageId
  }
}

/**
 * Chat adapter for incognito sessions: nothing is written to the daemon and
 * nothing may reach memory. Messages live only in the renderer and are
 * discarded when the session ends.
 */
export class InMemoryChatAdapter implements ChatAdapter {
  private lastMessageId: string | null = null

  constructor(private readonly hooks: ChatAdapterHooks = {}) {}

  async appendMessage(message: NewMessage): Promise<ConversationMessage> {
    const parentId = this.hooks.parentId?.() ?? this.lastMessageId
    const stored: Message = {
      id: `ephemeral-${crypto.randomUUID()}`,
      conversation_id: 'incognito',
      parent_id: parentId,
      role: message.role,
      content: message.content,
      model: message.role === 'assistant' ? (this.hooks.model?.() ?? null) : null,
      source: message.source,
      correlation_id: message.correlationId,
      status: message.status,
      metadata: message.metadata ?? null,
      created_at: new Date().toISOString()
    }
    this.lastMessageId = stored.id
    this.hooks.onMessage?.(stored)
    return toConversationMessage(stored)
  }

  async updateMessage(messageId: string, patch: MessagePatch): Promise<ConversationMessage> {
    this.hooks.onStatus?.(null)
    return {
      id: messageId,
      conversationId: 'incognito',
      role: 'assistant',
      source: 'assistant_chat',
      content: patch.content ?? '',
      createdAt: new Date().toISOString(),
      status: patch.status ?? 'final',
      metadata: patch.metadata
    }
  }

  showStatus(status: string | null): void {
    this.hooks.onStatus?.(status)
  }

  markQueued(messageId: string): void {
    this.hooks.onStatus?.('Queued behind the turn already running.')
    void messageId
  }

  markCancelled(messageId: string): void {
    this.hooks.onStatus?.('Cancelled.')
    void messageId
  }
}
