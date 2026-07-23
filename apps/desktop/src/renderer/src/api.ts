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

export type LocalModel = {
  id: string
  object: string
  owned_by: string
  size_bytes?: number | null
  capabilities?: {
    input_modalities: string[]
    output_modalities: string[]
    streaming: boolean
    tools: boolean
    reasoning: boolean
  }
}

export type RuntimeTarget =
  | 'auto'
  | 'cpu'
  | 'cuda'
  | 'rocm'
  | 'metal'
  | 'vulkan'

export type RuntimeSettings = {
  target: RuntimeTarget
  context_size: number
  batch_size: number
  threads: number | null
  gpu_layers: number
  flash_attention: boolean
  kv_cache_type_k: string
  kv_cache_type_v: string
  jinja: boolean
  temperature: number
  top_p: number
  max_tokens: number | null
  enable_reasoning: boolean
  binary_override: string | null
}

export type HardwareInfo = {
  os: string
  architecture: string
  logical_cpus: number
  memory_bytes: number | null
  gpu: string | null
  recommended_target: RuntimeTarget
  targets: Array<{
    id: RuntimeTarget
    name: string
    available: boolean
    recommended: boolean
    managed_install: boolean
    detail: string
  }>
}

export type EngineStatus = {
  id: string
  llama_binary: string | null
  llama_server: { base_url: string; model_path: string } | null
  managed_binary_path: string
  platform_asset_tag: string | null
  settings: RuntimeSettings
  hardware: HardwareInfo
}

export function health(): Promise<{ status: string; engine: string; version: string }> {
  return request('/health')
}

export function hardwareInfo(): Promise<HardwareInfo> {
  return request('/api/v1/hardware')
}

export function engineStatus(): Promise<EngineStatus> {
  return request('/api/v1/engines')
}

export function runtimeSettings(): Promise<RuntimeSettings> {
  return request('/api/v1/runtime/settings')
}

export function saveRuntimeSettings(settings: RuntimeSettings): Promise<RuntimeSettings> {
  return request('/api/v1/runtime/settings', {
    method: 'PUT',
    body: JSON.stringify(settings)
  })
}

export async function listModels(): Promise<LocalModel[]> {
  return (await request<{ data: LocalModel[] }>('/v1/models')).data
}

export async function listConversations(query?: string): Promise<Conversation[]> {
  const suffix =
    query && query.trim() ? `?q=${encodeURIComponent(query.trim())}` : ''
  return (await request<{ data: Conversation[] }>(`/api/v1/conversations${suffix}`)).data
}

export type ConversationExport = {
  schema_version: number
  exported_at: string
  conversation: Conversation
  messages: Message[]
}

export async function exportConversation(conversationId: string): Promise<ConversationExport> {
  return request(`/api/v1/conversations/${conversationId}/export`)
}

export async function importConversation(exportBundle: ConversationExport): Promise<Conversation> {
  return request('/api/v1/conversations/import', {
    method: 'POST',
    body: JSON.stringify(exportBundle)
  })
}

export type ModelTrust = {
  repo_id: string
  gated: boolean
  license: string | null
  remote_code: boolean
  requires_acknowledgement: boolean
}

export async function fetchModelTrust(repoId: string): Promise<ModelTrust> {
  const [owner, name] = repoId.split('/')
  return request(`/api/v1/huggingface/models/${owner}/${name}/trust`)
}

export type DownloadJob = {
  id: string
  repo_id: string
  filename: string
  revision: string
  status: string
  bytes_downloaded: number | null
  total_bytes: number | null
  sha256: string | null
  error: string | null
  created_at: string
  updated_at: string
}

export async function listDownloadJobs(): Promise<DownloadJob[]> {
  return (await request<{ data: DownloadJob[] }>('/api/v1/models/downloads')).data
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

export type ToolCallRecord = {
  call_id: string
  name: string
  arguments: string
  output: string
  is_error: boolean
}

export async function streamCompletion(
  messages: Message[],
  model: string,
  signal: AbortSignal,
  onToken: (token: string) => void,
  options?: {
    builtinTools?: boolean
    onToolCall?: (record: ToolCallRecord) => void
  }
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
      model,
      stream: true,
      ...(options?.builtinTools ? { builtin_tools: true } : {}),
      messages: messages
        .filter(({ role }) => role !== 'tool')
        .map(({ role, content }) => ({ role, content }))
    })
  })
  if (!response.ok || !response.body) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(payload?.error?.message ?? `Generation failed (${response.status}).`)
  }

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
        brazier?: { tool_call?: ToolCallRecord }
        error?: { message?: string }
      }
      if (chunk.error?.message) throw new Error(chunk.error.message)
      if (chunk.brazier?.tool_call) options?.onToolCall?.(chunk.brazier.tool_call)
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

export type HubFile = {
  path: string
  size: number | null
}

export type HubFilesResponse = {
  repo_id: string
  data: HubFile[]
  preferred_filename: string | null
}

export async function listHubFiles(repoId: string): Promise<HubFilesResponse> {
  const [owner, name] = repoId.split('/')
  if (!owner || !name) throw new Error('Repository id must be owner/name.')
  return request(`/api/v1/huggingface/models/${owner}/${name}/files`)
}

export type DownloadResult = {
  model_id: string
  path: string
  bytes: number
  sha256: string
  resumed: boolean
}

export type ProgressEvent = {
  phase: string
  bytes?: number
  total?: number
  percent?: number
  message?: string
  done?: boolean
  error?: string
  result?: Record<string, unknown>
}

async function readProgressSse(
  path: string,
  init: RequestInit,
  onProgress: (event: ProgressEvent) => void
): Promise<ProgressEvent> {
  const daemon = await connection()
  const headers = new Headers(init.headers)
  headers.set('content-type', 'application/json')
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}${path}`, { ...init, headers })
  if (!response.ok || !response.body) {
    throw new Error(`Request failed with status ${response.status}.`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let last: ProgressEvent | null = null

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
      if (!data) continue
      const event = JSON.parse(data) as ProgressEvent
      last = event
      onProgress(event)
      if (event.error) {
      const error = new Error(event.error) as Error & {
        diagnostics?: Record<string, unknown>
      }
      if (event.result) error.diagnostics = event.result
      throw error
    }
      if (event.done) return event
    }
  }

  if (last?.error) throw new Error(last.error)
  if (last?.done) return last
  throw new Error('Download ended without a completion event.')
}

export async function downloadModel(
  repoId: string,
  filename: string,
  onProgress: (event: ProgressEvent) => void,
  revision = 'main'
): Promise<DownloadResult> {
  const final = await readProgressSse(
    '/api/v1/models/download?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, filename, revision })
    },
    onProgress
  )
  const result = final.result as DownloadResult | undefined
  if (!result?.model_id) throw new Error('Download completed without a model id.')
  return result
}

export async function ensureLlamaEngine(
  onProgress: (event: ProgressEvent) => void
): Promise<{ binary: string; status: string }> {
  const final = await readProgressSse(
    '/api/v1/engines/llama.cpp/ensure?stream=true',
    { method: 'POST', body: '{}' },
    onProgress
  )
  const result = final.result as { binary?: string; status?: string } | undefined
  if (!result?.binary) throw new Error('Engine install completed without a binary path.')
  return { binary: result.binary, status: result.status ?? 'ready' }
}

export type RuntimeEntry = {
  id: string
  kind: 'managed' | 'source' | 'system'
  label: string
  target: string | null
  version: string | null
  path: string
  active: boolean
  deletable: boolean
}

export async function listRuntimes(): Promise<{
  data: RuntimeEntry[]
  active_binary: string | null
}> {
  return request('/api/v1/runtimes')
}

export function activateRuntime(id: string): Promise<{ active_binary: string; id: string }> {
  return request('/api/v1/runtimes/activate', {
    method: 'POST',
    body: JSON.stringify({ id })
  })
}

export function deleteRuntime(id: string): Promise<{ deleted: string }> {
  return request('/api/v1/runtimes', {
    method: 'DELETE',
    body: JSON.stringify({ id })
  })
}

export function deleteModel(modelId: string): Promise<{ deleted: string }> {
  return request('/api/v1/models', {
    method: 'DELETE',
    body: JSON.stringify({ model_id: modelId })
  })
}

export type BundledTool = {
  name: string
  title: string
  description: string
  network: boolean
}

export async function listTools(): Promise<BundledTool[]> {
  return (await request<{ data: BundledTool[] }>('/api/v1/tools')).data
}

export async function buildRuntime(
  repository: string,
  revision: string,
  target: string,
  onProgress: (event: ProgressEvent) => void,
  options?: { onBuildId?: (buildId: string) => void }
): Promise<{ binary: string; build_id: string }> {
  const final = await readProgressSse(
    '/api/v1/runtimes/build?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({
        engine: 'llama.cpp',
        repository,
        revision,
        target
      })
    },
    (event) => {
      const buildId = event.result?.build_id
      if (typeof buildId === 'string') {
        options?.onBuildId?.(buildId)
      }
      onProgress(event)
    }
  )
  const result = final.result as { binary?: string; build_id?: string } | undefined
  if (!result?.binary) throw new Error('Build completed without a binary path.')
  return { binary: result.binary, build_id: result.build_id ?? '' }
}

export function cancelBuild(buildId: string): Promise<{ cancelled: string }> {
  return request('/api/v1/runtimes/build/cancel', {
    method: 'POST',
    body: JSON.stringify({ build_id: buildId })
  })
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || Number.isNaN(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

export type StoredBlob = {
  sha256: string
  mime_type: string
  size_bytes: number
  original_name?: string | null
}

const blobUrlCache = new Map<string, string>()

export async function fetchBlobObjectUrl(sha256: string): Promise<string> {
  const cached = blobUrlCache.get(sha256)
  if (cached) return cached
  const daemon = await connection()
  const headers = new Headers()
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}/api/v1/blobs/${sha256}`, { headers })
  if (!response.ok) throw new Error(`Could not load attachment (${response.status}).`)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  blobUrlCache.set(sha256, url)
  return url
}

export async function uploadAttachmentBlob(file: File): Promise<StoredBlob> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('Could not read attachment.'))
    reader.onload = () => resolve(String(reader.result))
    reader.readAsDataURL(file)
  })
  const dataBase64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl
  const stored = await request<StoredBlob>('/api/v1/blobs', {
    method: 'POST',
    body: JSON.stringify({
      mime_type: file.type || 'application/octet-stream',
      data_base64: dataBase64,
      filename: file.name
    })
  })
  return stored
}

export function recordRun(
  conversationId: string,
  body: {
    parent_message_id?: string | null
    assistant_message_id?: string | null
    model: string
    settings: RuntimeSettings
    tool_calls?: ToolCallRecord[]
    response_text?: string
    error?: string
  }
): Promise<{ id: string }> {
  return request(`/api/v1/conversations/${conversationId}/runs`, {
    method: 'POST',
    body: JSON.stringify({
      parent_message_id: body.parent_message_id ?? null,
      assistant_message_id: body.assistant_message_id ?? null,
      model: body.model,
      settings: body.settings,
      tool_calls: body.tool_calls?.length ? body.tool_calls : null,
      response_text: body.response_text ?? null,
      error: body.error ?? null
    })
  })
}

export function huggingFaceTokenStatus(): Promise<{ configured: boolean; source: string }> {
  return request('/api/v1/huggingface/token')
}

export function setHuggingFaceToken(token: string): Promise<{ configured: boolean }> {
  return request('/api/v1/huggingface/token', {
    method: 'PUT',
    body: JSON.stringify({ token })
  })
}

export function clearHuggingFaceToken(): Promise<{ configured: boolean }> {
  return request('/api/v1/huggingface/token', { method: 'DELETE' })
}
