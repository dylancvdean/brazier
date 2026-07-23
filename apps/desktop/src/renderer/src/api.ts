import type { ContentPart, Conversation, HubModel, Message, Role } from './types'

type Connection = Awaited<ReturnType<typeof window.brazier.getConnection>>
let connectionPromise: Promise<Connection> | undefined

async function connection(): Promise<Connection> {
  connectionPromise ??= window.brazier.getConnection()
  return connectionPromise
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const daemon = await connection()
  const headers = new Headers(init?.headers)
  headers.set('content-type', 'application/json')
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}${path}`, { ...init, headers })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(payload?.error?.message ?? `Request failed with status ${response.status}.`)
  }
  return response.json() as Promise<T>
}

export async function listConversations(): Promise<Conversation[]> {
  return (await request<{ data: Conversation[] }>('/api/v1/conversations')).data
}

export function createConversation(title = 'New conversation'): Promise<Conversation> {
  return request('/api/v1/conversations', {
    method: 'POST',
    body: JSON.stringify({ title })
  })
}

export async function listMessages(conversationId: string): Promise<Message[]> {
  return (
    await request<{ data: Message[] }>(`/api/v1/conversations/${conversationId}/messages`)
  ).data
}

export function createMessage(
  conversationId: string,
  message: {
    parent_id: string | null
    role: Role
    content: string | ContentPart[]
    model?: string
  }
): Promise<Message> {
  return request(`/api/v1/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify(message)
  })
}

export async function streamCompletion(
  messages: Message[],
  signal: AbortSignal,
  onToken: (token: string) => void
): Promise<void> {
  const daemon = await connection()
  const response = await fetch(`${daemon.address}/v1/chat/completions`, {
    method: 'POST',
    signal,
    headers: {
      'content-type': 'application/json',
      ...(daemon.api_key ? { authorization: `Bearer ${daemon.api_key}` } : {})
    },
    body: JSON.stringify({
      model: 'brazier/mock',
      stream: true,
      messages: messages.map(({ role, content }) => ({ role, content }))
    })
  })
  if (!response.ok || !response.body) throw new Error(`Generation failed (${response.status}).`)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const data = frame
        .split('\n')
        .find((line) => line.startsWith('data:'))
        ?.slice(5)
        .trim()
      if (!data || data === '[DONE]') continue
      const chunk = JSON.parse(data) as {
        choices?: Array<{ delta?: { content?: string } }>
      }
      const token = chunk.choices?.[0]?.delta?.content
      if (token) onToken(token)
    }
  }
}

export async function searchHub(query: string, engine: string): Promise<HubModel[]> {
  const parameters = new URLSearchParams({ q: query, engine, limit: '40' })
  return (
    await request<{ data: HubModel[] }>(`/api/v1/huggingface/models?${parameters.toString()}`)
  ).data
}
