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
  engine?: string
  size_bytes?: number | null
  read_only?: boolean
  library_label?: string | null
  capabilities?: {
    input_modalities: string[]
    output_modalities: string[]
    streaming: boolean
    tools: boolean
    reasoning: boolean
    max_context_length?: number | null
    reasoning_modes?: string[]
    harmony?: boolean
    /** `native` when the chat model consumes audio tokens directly. */
    audio_input?: string | null
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
  reasoning_budget_tokens?: number | null
  binary_override: string | null
  mlx_lm_python?: string | null
  mlx_vlm_python?: string | null
  whisper_binary?: string | null
  whisper_model?: string | null
  streaming_asr_python?: string | null
  streaming_asr_model?: string | null
  sdcpp_binary?: string | null
  /** Show generated images back to a vision model so it can iterate. */
  show_generated_images_to_model?: boolean
  /** Same for video, by sampling frames — far more context per clip. */
  show_generated_video_to_model?: boolean
  default_image_gen_model?: string | null
  default_video_gen_model?: string | null
  voice_python?: string | null
  default_voice_model?: string | null
  default_voice_persona?: string | null
  build_jobs: number
  extra_model_library_paths: string[]
  generation_memory_policy: 'auto' | 'coresident' | 'exclusive'
  generation_memory_headroom_mb: number
  reload_llm_after_generation: boolean
}

export type PipelineFeatures = {
  asr: boolean
  video_preprocess: boolean
  whisper_cpp_engine?: boolean
  native_model_audio?: boolean
  streaming_asr?: boolean
  realtime_voice?: boolean
}

export type CapabilitiesResponse = {
  schema_version: number
  features: Record<string, unknown> & {
    asr?: boolean
    video_preprocess?: boolean
    audio_interfaces?: {
      batch_asr?: { available?: boolean; summary?: string }
      native_model_audio?: { available?: boolean; summary?: string }
      streaming_asr?: { available?: boolean; planned?: boolean; summary?: string }
      realtime_voice?: { available?: boolean; planned?: boolean; summary?: string }
    }
    generation_interfaces?: {
      image_gen?: { available?: boolean; summary?: string }
      video_gen?: { available?: boolean; summary?: string }
    }
  }
}

export async function fetchCapabilities(): Promise<CapabilitiesResponse> {
  return request('/api/v1/capabilities')
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

export type ToolchainTool = {
  id: string
  label: string
  available: boolean
  required_for: string
  install_hint: string | null
}

export type ToolchainStatus = {
  os: {
    family: string
    id: string
    pretty_name: string
  }
  tools: ToolchainTool[]
  platforms: {
    mlx: boolean
    streaming_asr: boolean
    whisper_cpp: boolean
    llama_cpp: boolean
  }
}

export function fetchToolchainStatus(): Promise<ToolchainStatus> {
  return request('/api/v1/toolchain')
}

export function engineStatus(options?: { probe?: boolean }): Promise<EngineStatus> {
  const suffix = options?.probe ? '?probe=true' : ''
  return request(`/api/v1/engines${suffix}`)
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

export type ModelLibraryPathSuggestion = {
  id: string
  label: string
  path: string
  exists: boolean
  gguf_count: number
  mlx_count: number
  configured: boolean
}

export async function modelLibraryPathSuggestions(): Promise<{
  configured: string[]
  suggestions: ModelLibraryPathSuggestion[]
}> {
  return request('/api/v1/models/library-paths/suggestions')
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
  blobs?: ExportBlob[]
  run_snapshots?: RunSnapshot[]
}

export type ExportBlob = {
  sha256: string
  mime_type: string
  data_base64: string
  original_name?: string | null
}

export type RunSnapshot = {
  id: string
  conversation_id: string
  parent_message_id: string | null
  assistant_message_id: string | null
  model: string
  settings: RuntimeSettings
  tool_calls: ToolCallRecord[] | null
  response_text: string | null
  error: string | null
  created_at: string
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

export async function fetchModelDescription(repoId: string): Promise<string> {
  const [owner, name] = repoId.split('/')
  const response = await request<{ description: string }>(
    `/api/v1/huggingface/models/${owner}/${name}/description`
  )
  return response.description
}

export type DownloadJob = {
  kind?: string
  label?: string | null
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

export async function cancelDownloadJob(jobId: string): Promise<void> {
  await request('/api/v1/models/download/cancel', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId })
  })
}

export async function pauseDownloadJob(jobId: string): Promise<void> {
  await request('/api/v1/models/download/pause', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId })
  })
}

/** Put a paused, failed, or cancelled job back in line; it resumes in place. */
export async function resumeDownloadJob(jobId: string): Promise<void> {
  await request('/api/v1/models/download/resume', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId })
  })
}

/** Queue a multi-file snapshot download (MLX, PersonaPlex, streaming ASR). */
export async function queueSnapshotDownload(
  kind: 'mlx' | 'personaplex' | 'streaming-asr',
  repoId: string,
  engine?: string
): Promise<{ job_id: string }> {
  return request(`/api/v1/models/download/queue/snapshot/${kind}`, {
    method: 'POST',
    body: JSON.stringify({ repo_id: repoId, engine })
  })
}

/** Queue a stable-diffusion.cpp bundle install. */
export async function queueSdcppInstall(
  target: { id: string } | { bundle: SdcppBundle }
): Promise<{ job_id: string }> {
  return request('/api/v1/models/sdcpp/install/queue', {
    method: 'POST',
    body: JSON.stringify(target)
  })
}

export async function queueModelDownload(
  repoId: string,
  filename: string,
  revision = 'main'
): Promise<{ job_id: string }> {
  return request('/api/v1/models/download/queue', {
    method: 'POST',
    body: JSON.stringify({ repo_id: repoId, filename, revision })
  })
}

export async function listRunSnapshots(conversationId: string): Promise<RunSnapshot[]> {
  return (
    await request<{ data: RunSnapshot[] }>(`/api/v1/conversations/${conversationId}/runs`)
  ).data
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
    tool_calls?: unknown[] | null
    tool_call_id?: string | null
  }
): Promise<Message> {
  return request(`/api/v1/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify(message)
  })
}

export type ClientToolCall = {
  id: string
  name: string
  arguments: string
}

export type TranscriptMessagePayload = {
  role: string
  content: string | null
  tool_calls?: unknown[] | null
  tool_call_id?: string | null
}

export type StreamCompletionResult = {
  responseText: string
  toolRecords: ToolCallRecord[]
  clientToolCalls: ClientToolCall[]
  transcript: TranscriptMessagePayload[]
}

export type ToolCallRecord = {
  call_id: string
  name: string
  arguments: string
  output: string
  is_error: boolean
}

export type RuntimeForkHint = {
  engine: string
  display_name: string
  repository: string
  trusted: boolean
  summary: string
}

export class GenerationFailure extends Error {
  forkHints: RuntimeForkHint[]

  constructor(message: string, forkHints: RuntimeForkHint[] = []) {
    super(message)
    this.name = 'GenerationFailure'
    this.forkHints = forkHints
  }
}

function forkHintsFromPayload(payload: unknown): RuntimeForkHint[] {
  if (!payload || typeof payload !== 'object') return []
  const brazier = (payload as { brazier?: { fork_hints?: RuntimeForkHint[] } }).brazier
  return Array.isArray(brazier?.fork_hints) ? brazier.fork_hints : []
}

export async function fetchForkHints(repoId: string): Promise<RuntimeForkHint[]> {
  const [owner, name] = repoId.split('/')
  if (!owner || !name) throw new Error('Repository id must be owner/name.')
  const response = await request<{ fork_hints: RuntimeForkHint[] }>(
    `/api/v1/huggingface/models/${owner}/${name}/fork-hints`
  )
  return response.fork_hints ?? []
}

export async function streamCompletion(
  messages: Message[],
  model: string,
  signal: AbortSignal,
  onToken: (token: string) => void,
  options?: {
    builtinTools?: boolean
    builtinToolNames?: string[]
    toolChoice?: 'auto' | 'none' | { type: 'function'; function: { name: string } }
    onToolCall?: (record: ToolCallRecord) => void
    onLoad?: (event: { phase: string; message: string }) => void
  }
): Promise<StreamCompletionResult> {
  const daemon = await connection()
  const toolChoice =
    options?.toolChoice === 'auto' || options?.toolChoice === 'none'
      ? options.toolChoice
      : options?.toolChoice
        ? options.toolChoice
        : undefined
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
      ...(options?.builtinTools && options.builtinToolNames
        ? { builtin_tool_names: options.builtinToolNames }
        : {}),
      ...(toolChoice ? { tool_choice: toolChoice } : {}),
      messages: messagesForCompletion(messages)
    })
  })
  if (!response.ok || !response.body) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
      brazier?: { fork_hints?: RuntimeForkHint[] }
    } | null
    throw new GenerationFailure(
      payload?.error?.message ?? `Generation failed (${response.status}).`,
      forkHintsFromPayload(payload)
    )
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const toolRecords: ToolCallRecord[] = []
  const clientToolCalls: ClientToolCall[] = []
  const transcript: TranscriptMessagePayload[] = []
  let responseText = ''
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
        choices?: Array<{
          delta?: {
            content?: string
            tool_calls?: Array<{
              index?: number
              id?: string
              type?: string
              function?: { name?: string; arguments?: string }
            }>
          }
          finish_reason?: string | null
        }>
        brazier?: {
          tool_call?: ToolCallRecord
          transcript_message?: TranscriptMessagePayload
          fork_hints?: RuntimeForkHint[]
          load?: { phase: string; message: string }
        }
        error?: { message?: string }
      }
      if (chunk.error?.message) {
        throw new GenerationFailure(chunk.error.message, chunk.brazier?.fork_hints ?? [])
      }
      if (chunk.brazier?.load) options?.onLoad?.(chunk.brazier.load)
      if (chunk.brazier?.tool_call) {
        toolRecords.push(chunk.brazier.tool_call)
        options?.onToolCall?.(chunk.brazier.tool_call)
      }
      if (chunk.brazier?.transcript_message) {
        transcript.push(chunk.brazier.transcript_message)
      }
      const finishReason = chunk.choices?.[0]?.finish_reason
      const toolCalls = chunk.choices?.[0]?.delta?.tool_calls
      if (finishReason === 'tool_calls' && toolCalls?.length) {
        for (const call of toolCalls) {
          if (call.id && call.function?.name) {
            clientToolCalls.push({
              id: call.id,
              name: call.function.name,
              arguments: call.function.arguments ?? ''
            })
          }
        }
      }
      const token = chunk.choices?.[0]?.delta?.content
      if (token) {
        responseText += token
        onToken(token)
      }
    }
  }
  return { responseText, toolRecords, clientToolCalls, transcript }
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
  engine?: string
  notice?: string
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

  const consumeFrame = (frame: string): ProgressEvent | null => {
    const data = frame
      .split('\n')
      .find((line) => line.startsWith('data:'))
      ?.slice(5)
      .trim()
    if (!data || data === '[DONE]') return null
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
    if (isTerminalProgress(event)) return event
    return null
  }

  const drainFrames = (final = false): ProgressEvent | null => {
    const parts = buffer.split('\n\n')
    if (final) {
      buffer = ''
      for (const frame of parts) {
        if (!frame.trim()) continue
        const terminal = consumeFrame(frame)
        if (terminal) return terminal
      }
    } else {
      buffer = parts.pop() ?? ''
      for (const frame of parts) {
        if (!frame.trim()) continue
        const terminal = consumeFrame(frame)
        if (terminal) return terminal
      }
    }
    return null
  }

  while (true) {
    const { done, value } = await reader.read()
    if (value) {
      buffer += decoder.decode(value, { stream: true })
      const terminal = drainFrames(false)
      if (terminal) return terminal
    }
    if (done) {
      buffer += decoder.decode()
      const terminal = drainFrames(true)
      if (terminal) return terminal
      break
    }
  }

  if (last && isTerminalProgress(last)) return last
  throw new Error('Operation ended without a completion event.')
}

function isTerminalProgress(event: ProgressEvent): boolean {
  if (event.done) return true
  if (event.phase === 'done' && event.result) return true
  if (event.phase === 'error') return true
  return false
}

export async function downloadModel(
  repoId: string,
  filename: string,
  onProgress: (event: ProgressEvent) => void,
  revision = 'main',
  engine: 'llama.cpp' | 'whisper.cpp' = 'llama.cpp'
): Promise<DownloadResult> {
  const final = await readProgressSse(
    '/api/v1/models/download?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, filename, revision, engine })
    },
    onProgress
  )
  const result = final.result as DownloadResult | undefined
  if (!result?.model_id) throw new Error('Download completed without a model id.')
  return result
}

export async function downloadMlxModel(
  repoId: string,
  engine: 'mlx-lm' | 'mlx-vlm',
  onProgress: (event: ProgressEvent) => void,
  revision = 'main'
): Promise<DownloadResult> {
  const final = await readProgressSse(
    '/api/v1/models/download/mlx?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, engine, revision })
    },
    onProgress
  )
  const result = final.result as DownloadResult | undefined
  if (!result?.model_id) throw new Error('Download completed without a model id.')
  return result
}

export async function downloadStreamingAsrModel(
  repoId: string,
  onProgress: (event: ProgressEvent) => void,
  revision = 'main'
): Promise<DownloadResult> {
  const final = await readProgressSse(
    '/api/v1/models/download/streaming-asr?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, engine: 'streaming-asr', revision })
    },
    onProgress
  )
  const result = final.result as DownloadResult | undefined
  if (!result?.model_id) throw new Error('Download completed without a model id.')
  return result
}

export async function downloadPersonaplexModel(
  repoId: string,
  onProgress: (event: ProgressEvent) => void,
  revision = 'main'
): Promise<DownloadResult> {
  const final = await readProgressSse(
    '/api/v1/models/download/personaplex?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, engine: 'personaplex', revision })
    },
    onProgress
  )
  const result = final.result as DownloadResult | undefined
  if (!result?.model_id) throw new Error('Download completed without a model id.')
  return result
}

export type ManagedLlamaTargetStatus = {
  target: string
  installed: boolean
  installed_version: string | null
  latest_version: string | null
  update_available: boolean
}

export type ManagedEngineStatus = {
  latest_version: string | null
  /** True while the daemon is still checking upstream for the newest release. */
  latest_pending?: boolean
  targets: ManagedLlamaTargetStatus[]
}

export async function fetchManagedLlamaStatus(): Promise<ManagedEngineStatus> {
  return request('/api/v1/engines/llama.cpp/managed-status')
}

export async function ensureLlamaEngine(
  onProgress: (event: ProgressEvent) => void,
  options?: { target?: RuntimeTarget; force?: boolean }
): Promise<{ binary: string; status: string }> {
  const body = JSON.stringify({
    ...(options?.target ? { target: options.target } : {}),
    ...(options?.force ? { force: true } : {})
  })
  const final = await readProgressSse(
    '/api/v1/engines/llama.cpp/ensure?stream=true',
    { method: 'POST', body },
    onProgress
  )
  const result = final.result as { binary?: string; status?: string } | undefined
  if (!result?.binary) throw new Error('Engine install completed without a binary path.')
  return { binary: result.binary, status: result.status ?? 'ready' }
}

export async function fetchManagedWhisperStatus(): Promise<
  ManagedEngineStatus & { managed_supported: boolean; note?: string | null }
> {
  return request('/api/v1/engines/whisper.cpp/managed-status')
}

export async function ensureWhisperEngine(
  onProgress: (event: ProgressEvent) => void,
  options?: { target?: RuntimeTarget; force?: boolean }
): Promise<{ binary: string; status: string }> {
  const body = JSON.stringify({
    ...(options?.target ? { target: options.target } : {}),
    ...(options?.force ? { force: true } : {})
  })
  const final = await readProgressSse(
    '/api/v1/engines/whisper.cpp/ensure?stream=true',
    { method: 'POST', body },
    onProgress
  )
  const result = final.result as { binary?: string; status?: string } | undefined
  if (!result?.binary) throw new Error('Whisper install completed without a binary path.')
  return { binary: result.binary, status: result.status ?? 'ready' }
}

export async function fetchManagedSdcppStatus(): Promise<ManagedEngineStatus> {
  return request('/api/v1/engines/stable-diffusion.cpp/managed-status')
}

export async function ensureSdcppEngine(
  onProgress: (event: ProgressEvent) => void,
  options?: { target?: RuntimeTarget; force?: boolean }
): Promise<{ binary: string; status: string }> {
  const body = JSON.stringify({
    ...(options?.target ? { target: options.target } : {}),
    ...(options?.force ? { force: true } : {})
  })
  const final = await readProgressSse(
    '/api/v1/engines/stable-diffusion.cpp/ensure?stream=true',
    { method: 'POST', body },
    onProgress
  )
  const result = final.result as { binary?: string; status?: string } | undefined
  if (!result?.binary) throw new Error('sd.cpp install completed without a binary path.')
  return { binary: result.binary, status: result.status ?? 'ready' }
}

export type GenerateBlobResult = {
  blob: { sha256: string; mime_type: string; size_bytes: number; original_name?: string | null }
  metadata: unknown
  engine: string
}

export type GenerateBody = {
  prompt: string
  model_id?: string
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  seed?: number
  /** Classifier-free guidance. Distilled models such as Flux want 1.0. */
  cfg_scale?: number
  /** Distilled guidance, used by Flux-family models instead of CFG. */
  guidance?: number
  video_frames?: number
  fps?: number
}

export function generateImage(body: GenerateBody): Promise<GenerateBlobResult> {
  return request('/api/v1/generate/image', {
    method: 'POST',
    body: JSON.stringify(body)
  })
}

export function generateVideo(body: GenerateBody): Promise<GenerateBlobResult> {
  return request('/api/v1/generate/video', {
    method: 'POST',
    body: JSON.stringify(body)
  })
}

/** Generation settings that suit a curated model, used to prefill the panel. */
export type SdcppDefaults = {
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  guidance?: number
  video_frames?: number
  fps?: number
}

export type SdcppBundleComponent = {
  repo_id: string
  path: string
  /** sd-cli flag without the dashes; null for a self-contained checkpoint. */
  flag?: string | null
  role: string
  gated: boolean
  approx_bytes?: number | null
}

export type SdcppBundle = {
  id: string
  label: string
  modality: 'image' | 'video'
  key: string
  summary: string
  license?: string | null
  model_id: string
  installed: boolean
  gated: boolean
  approx_bytes?: number | null
  /** `builtin` ships with the app; `custom` lives in the data directory. */
  origin: 'builtin' | 'custom'
  /** Whether the model can start from a supplied image. */
  supports_init_image?: boolean
  defaults: SdcppDefaults
  components: SdcppBundleComponent[]
}

/** A bundle proposed for an arbitrary checkpoint, before it is installed. */
export type SdcppProposal = {
  bundle: SdcppBundle
  architecture: string | null
  architecture_label: string | null
  variant: string | null
  /** How the architecture was identified: GGUF metadata, tensor names, … */
  detected_by: string
  self_contained: boolean
  warnings: string[]
}

/** Inspect a checkpoint's header on the Hub and propose a bundle for it. */
export function assembleSdcppBundle(body: {
  repo_id: string
  path: string
  modality?: 'image' | 'video'
}): Promise<SdcppProposal> {
  return request('/api/v1/models/sdcpp/assemble', {
    method: 'POST',
    body: JSON.stringify(body)
  })
}

export function saveSdcppBundle(bundle: SdcppBundle): Promise<SdcppBundle> {
  return request('/api/v1/models/sdcpp/bundles', {
    method: 'PUT',
    body: JSON.stringify(bundle)
  })
}

export function deleteSdcppBundle(id: string): Promise<{ deleted: string }> {
  return request(`/api/v1/models/sdcpp/bundles/${encodeURIComponent(id)}`, {
    method: 'DELETE'
  })
}

/** Curated sd.cpp bundles: a model plus the VAE and text encoders it needs. */
export async function listSdcppBundles(): Promise<SdcppBundle[]> {
  return (await request<{ data: SdcppBundle[] }>('/api/v1/models/sdcpp/catalog')).data
}

/** Install a saved bundle by id, or a one-off bundle passed inline. */
export async function installSdcppBundle(
  target: { id: string } | { bundle: SdcppBundle },
  onProgress: (event: ProgressEvent) => void
): Promise<{ model_id: string; path: string; bytes: number }> {
  const final = await readProgressSse(
    '/api/v1/models/sdcpp/install?stream=true',
    { method: 'POST', body: JSON.stringify(target) },
    onProgress
  )
  return final.result as { model_id: string; path: string; bytes: number }
}

export type VoiceSessionInfo = {
  id: string
  ws_url: string
  persona_text: string
  voice_prompt?: string | null
  protocol?: { handshake: number; audio: number; text: number }
  engine?: string
}

export function createVoiceSession(body?: {
  model_id?: string
  persona_text?: string
  voice_prompt_path?: string
}): Promise<VoiceSessionInfo> {
  return request('/api/v1/voice/sessions', {
    method: 'POST',
    body: JSON.stringify(body ?? {})
  })
}

export function getVoiceSession(): Promise<{ session: VoiceSessionInfo | null }> {
  return request('/api/v1/voice/sessions')
}

export function endVoiceSession(id: string): Promise<{ ended: string }> {
  return request(`/api/v1/voice/sessions/${encodeURIComponent(id)}`, {
    method: 'DELETE'
  })
}

export type RuntimeEntry = {
  id: string
  engine: string
  kind: 'managed' | 'source' | 'system'
  label: string
  target: string | null
  version: string | null
  repository?: string | null
  path: string
  active: boolean
  deletable: boolean
}

export async function listRuntimes(options?: {
  includeSystem?: boolean
}): Promise<{
  data: RuntimeEntry[]
  active_binary: string | null
}> {
  const suffix = options?.includeSystem ? '?include_system=true' : ''
  return request(`/api/v1/runtimes${suffix}`)
}

export function activateRuntime(id: string): Promise<{ active_binary: string; id: string }> {
  return request('/api/v1/runtimes/activate', {
    method: 'POST',
    body: JSON.stringify({ id })
  })
}

export type SourceRuntimeUpdate = {
  id: string
  engine: string
  label: string
  repository: string
  revision: string
  current_commit: string | null
  upstream_commit: string | null
  update_available: boolean
  pinned: boolean
  error: string | null
}

/** Query upstream refs for every source-built runtime (network; on demand). */
export async function checkRuntimeUpdates(): Promise<SourceRuntimeUpdate[]> {
  return (
    await request<{ data: SourceRuntimeUpdate[] }>('/api/v1/runtimes/check-updates', {
      method: 'POST'
    })
  ).data
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

export async function fetchModelBindings(): Promise<Record<string, string>> {
  const response = await request<{ bindings: Record<string, string> }>('/api/v1/models/bindings')
  return response.bindings ?? {}
}

export async function setModelBinding(
  modelId: string,
  runtimeId: string | null
): Promise<Record<string, string>> {
  const response = await request<{ bindings: Record<string, string> }>(
    '/api/v1/models/bindings',
    {
      method: 'PUT',
      body: JSON.stringify({ model_id: modelId, runtime_id: runtimeId })
    }
  )
  return response.bindings ?? {}
}

export async function prepareModel(
  modelId: string,
  options?: {
    signal?: AbortSignal
    onLoad?: (event: { phase: string; message: string }) => void
  }
): Promise<void> {
  const daemon = await connection()
  const response = await fetch(`${daemon.address}/api/v1/models/prepare?stream=true`, {
    method: 'POST',
    signal: options?.signal,
    headers: {
      'content-type': 'application/json',
      ...(daemon.api_key ? { authorization: `Bearer ${daemon.api_key}` } : {})
    },
    body: JSON.stringify({ model_id: modelId })
  })
  if (!response.ok || !response.body) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
      brazier?: { fork_hints?: RuntimeForkHint[] }
    } | null
    throw new GenerationFailure(
      payload?.error?.message ?? `Model prepare failed (${response.status}).`,
      forkHintsFromPayload(payload)
    )
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
        phase?: string
        message?: string
        status?: string
        error?: { message?: string }
        brazier?: { fork_hints?: RuntimeForkHint[] }
      }
      if (chunk.error?.message) {
        throw new GenerationFailure(chunk.error.message, chunk.brazier?.fork_hints ?? [])
      }
      if (chunk.phase && chunk.message) {
        options?.onLoad?.({ phase: chunk.phase, message: chunk.message })
      }
      if (chunk.status === 'ready') return
    }
  }
}

export type BundledTool = {
  name: string
  title: string
  description?: string
  network: boolean
  source?: string
}

export type McpServer = {
  id: string
  name: string
  command: string
  args: string[]
  enabled: boolean
  tools: BundledTool[]
}

export async function listTools(): Promise<BundledTool[]> {
  return (await request<{ data: BundledTool[] }>('/api/v1/tools')).data
}

export async function listMcpServers(): Promise<McpServer[]> {
  return (await request<{ data: McpServer[] }>('/api/v1/mcp/servers')).data
}

export async function createMcpServer(body: {
  id: string
  name: string
  command: string
  args?: string[]
  enabled?: boolean
}): Promise<{ id: string }> {
  return request('/api/v1/mcp/servers', {
    method: 'POST',
    body: JSON.stringify(body)
  })
}

export async function updateMcpServer(
  id: string,
  body: {
    id: string
    name: string
    command: string
    args?: string[]
    enabled?: boolean
  }
): Promise<{ id: string }> {
  return request(`/api/v1/mcp/servers/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify(body)
  })
}

export async function deleteMcpServer(id: string): Promise<void> {
  await request(`/api/v1/mcp/servers/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function refreshMcpServer(id: string): Promise<{ tools: unknown[] }> {
  return request(`/api/v1/mcp/servers/${encodeURIComponent(id)}/refresh`, { method: 'POST' })
}

type OpenAiToolCall = {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

type OpenAiChatMessage = {
  role: Role
  content?: string | ContentPart[]
  tool_calls?: OpenAiToolCall[]
  tool_call_id?: string
}

function messageContentForApi(content: string | ContentPart[]): string | ContentPart[] {
  return content
}

function messagesForCompletion(messages: Message[]): OpenAiChatMessage[] {
  const payload: OpenAiChatMessage[] = []
  for (const message of messages) {
    if (message.role === 'tool' && message.tool_call_id) {
      payload.push({
        role: 'tool',
        tool_call_id: message.tool_call_id,
        content:
          typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content)
      })
      continue
    }
    if (message.role === 'assistant' && message.tool_calls?.length) {
      payload.push({
        role: 'assistant',
        content:
          typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content),
        tool_calls: message.tool_calls as OpenAiToolCall[]
      })
      continue
    }
    if (message.role === 'tool') {
      let records: ToolCallRecord[] | null = null
      if (typeof message.content === 'string') {
        try {
          const parsed = JSON.parse(message.content) as { brazier_tool_calls?: ToolCallRecord[] }
          records = Array.isArray(parsed.brazier_tool_calls) ? parsed.brazier_tool_calls : null
        } catch {
          records = null
        }
      }
      if (records && records.length > 0) {
        payload.push({
          role: 'assistant',
          content: '',
          tool_calls: records.map((record) => ({
            id: record.call_id,
            type: 'function',
            function: { name: record.name, arguments: record.arguments }
          }))
        })
        for (const record of records) {
          payload.push({
            role: 'tool',
            tool_call_id: record.call_id,
            content: record.output
          })
        }
        continue
      }
    }
    payload.push({
      role: message.role,
      content: messageContentForApi(message.content)
    })
  }
  return payload
}

export async function buildRuntime(
  engine: string,
  repository: string,
  revision: string,
  target: string,
  jobs: number,
  onProgress: (event: ProgressEvent) => void,
  options?: { onBuildId?: (buildId: string) => void }
): Promise<{ binary: string; build_id: string }> {
  const final = await readProgressSse(
    '/api/v1/runtimes/build?stream=true',
    {
      method: 'POST',
      body: JSON.stringify({
        engine,
        repository,
        revision,
        target,
        jobs
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
