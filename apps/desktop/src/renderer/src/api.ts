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
  // Successful commands such as stop and safety-authority intentionally use
  // 204 No Content. Calling Response.json() for those responses throws
  // "Unexpected end of JSON input" and can make a completed control-plane
  // operation look like an inference failure.
  if (response.status === 204 || response.status === 205) return undefined as T
  return response.json() as Promise<T>
}

export async function fetchWelcomePreference(): Promise<{ completed: boolean }> {
  return request('/api/v1/preferences/welcome')
}

export async function saveWelcomePreference(completed: boolean): Promise<{ completed: boolean }> {
  return request('/api/v1/preferences/welcome', {
    method: 'PUT',
    body: JSON.stringify({ completed })
  })
}

/** Which workspace modes appear in the top-bar mode switch. */
export type WorkspaceModesPreference = {
  chat: boolean
  agent: boolean
  generate: boolean
  voice: boolean
  computer: boolean
}

export async function fetchWorkspacePreference(): Promise<{ modes: WorkspaceModesPreference }> {
  return request('/api/v1/preferences/workspace')
}

export async function saveWorkspacePreference(
  modes: WorkspaceModesPreference
): Promise<{ modes: WorkspaceModesPreference }> {
  return request('/api/v1/preferences/workspace', {
    method: 'PUT',
    body: JSON.stringify({ modes })
  })
}

export type ComputerTarget = 'browser' | 'desktop'

export type ComputerPermissionMode = 'ask' | 'browser-only' | 'skip-permissions' | 'allow-all'

export type OsPermissionState =
  | 'granted'
  | 'denied'
  | 'missing'
  | 'unsupported'
  | 'unknown'

export type OsPermissionStatus = {
  platform: string
  display_server: string
  screen_capture: OsPermissionState
  input_injection: OsPermissionState
  detail?: string | null
  settings_hint?: string | null
}

export async function requestComputerPermissions(): Promise<OsPermissionStatus> {
  return request('/api/v1/computer/permissions', { method: 'POST' })
}

export type ComputerUsePreference = {
  action_settle_delay_ms: number
  /** Screenshots retained in a computer-use trajectory (Fara defaults to 3). */
  max_screenshots_kept: number
}

export function fetchComputerUsePreference(): Promise<ComputerUsePreference> {
  return request('/api/v1/preferences/computer')
}

export function saveComputerUsePreference(
  preference: ComputerUsePreference
): Promise<ComputerUsePreference> {
  return request('/api/v1/preferences/computer', {
    method: 'PUT',
    body: JSON.stringify(preference)
  })
}

export type ComputerViewport = {
  width: number
  height: number
  device_pixel_ratio?: number | null
}

/** Normalized computer-use action (tagged by `type`, snake_case). */
export type ComputerAction =
  | { type: 'screenshot' }
  | { type: 'left_click'; x: number; y: number }
  | { type: 'right_click'; x: number; y: number }
  | { type: 'double_click'; x: number; y: number }
  | { type: 'triple_click'; x: number; y: number }
  | { type: 'mouse_move'; x: number; y: number }
  | {
      type: 'left_click_drag'
      start_x: number
      start_y: number
      end_x: number
      end_y: number
    }
  | { type: 'type'; text: string }
  | { type: 'keypress'; keys: string[] }
  | { type: 'scroll'; x: number; y: number; delta_x: number; delta_y: number }
  | { type: 'wait'; milliseconds?: number }
  | { type: 'visit_url'; url: string }
  | { type: 'web_search'; query: string }
  | { type: 'memorize'; fact: string }
  | { type: 'ask_user'; question: string }
  | { type: 'terminate'; response?: string | null }

export type ComputerActionStatus =
  | 'ok'
  | 'needs_approval'
  | 'refused'
  | 'error'
  | 'finished'
  | 'waiting_for_user'

export type ComputerActionResult = {
  status: ComputerActionStatus
  message?: string | null
  screenshot_base64?: string | null
  mime_type?: string | null
  viewport?: ComputerViewport | null
  url?: string | null
  title?: string | null
  needs_approval?: boolean
  approval_id?: string | null
}

export type ComputerSession = {
  id: string
  title: string
  target: ComputerTarget
  model_id?: string | null
  permission_mode: ComputerPermissionMode
  viewport: ComputerViewport
  created_at: string
  updated_at: string
  url?: string | null
  title_page?: string | null
  running?: boolean
  memories?: string[]
}

export type ComputerStep = {
  id: string
  session_id: string
  role: string
  content: string
  thought?: string | null
  action?: ComputerAction | null
  result?: ComputerActionResult | null
  created_at: string
}

export type FaraParseResult = {
  thought?: string | null
  actions: ComputerAction[]
  raw_tool_calls?: string[]
}

export async function fetchComputerPermissions(): Promise<OsPermissionStatus> {
  return request('/api/v1/computer/permissions')
}

export async function listComputerSessions(): Promise<ComputerSession[]> {
  const payload = await request<{ sessions: ComputerSession[] }>('/api/v1/computer/sessions')
  return payload.sessions ?? []
}

export async function createComputerSession(body: {
  title?: string
  target?: ComputerTarget
  model_id?: string | null
  permission_mode?: ComputerPermissionMode
  viewport?: ComputerViewport
}): Promise<ComputerSession> {
  const elevated =
    body.permission_mode === 'skip-permissions' || body.permission_mode === 'allow-all'
  return request('/api/v1/computer/sessions', {
    method: 'POST',
    body: JSON.stringify({
      ...body,
      ...(elevated ? { confirm_elevated_permissions: true } : {})
    })
  })
}

export async function fetchComputerSession(id: string): Promise<ComputerSession> {
  return request(`/api/v1/computer/sessions/${encodeURIComponent(id)}`)
}

/** Change the permission mode of a live session. Elevated modes are confirmed. */
export async function updateComputerSession(
  id: string,
  permission_mode: ComputerPermissionMode
): Promise<ComputerSession> {
  const elevated =
    permission_mode === 'skip-permissions' || permission_mode === 'allow-all'
  return request(`/api/v1/computer/sessions/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify({
      permission_mode,
      ...(elevated ? { confirm_elevated_permissions: true } : {})
    })
  })
}

export async function deleteComputerSession(id: string): Promise<void> {
  const daemon = await connection()
  const headers = new Headers({ 'content-type': 'application/json' })
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(
    `${daemon.address}/api/v1/computer/sessions/${encodeURIComponent(id)}`,
    { method: 'DELETE', headers }
  )
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(payload?.error?.message ?? `Request failed with status ${response.status}.`)
  }
}

export async function listComputerSteps(sessionId: string): Promise<ComputerStep[]> {
  const payload = await request<{ steps: ComputerStep[] }>(
    `/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/steps`
  )
  return payload.steps ?? []
}

export async function appendComputerStep(
  sessionId: string,
  body: {
    role: string
    content: string
    thought?: string | null
    action?: ComputerAction | null
    result?: ComputerActionResult | null
  }
): Promise<ComputerStep> {
  return request(`/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/steps`, {
    method: 'POST',
    body: JSON.stringify(body)
  })
}

export async function computerScreenshot(sessionId: string): Promise<ComputerActionResult> {
  return request(`/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/screenshot`, {
    method: 'POST'
  })
}

/** Live viewport capture that never writes a step into the trajectory. */
export async function computerPreview(sessionId: string): Promise<ComputerActionResult> {
  return request(`/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/preview`, {
    method: 'POST'
  })
}

export async function stopComputerSession(sessionId: string): Promise<void> {
  await request(`/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/stop`, {
    method: 'POST'
  })
}

export async function setComputerSafetyAuthority(
  sessionId: string,
  active: boolean
): Promise<void> {
  await request(
    `/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/safety-authority`,
    {
      method: 'POST',
      body: JSON.stringify({ active })
    }
  )
}

export async function computerExec(body: {
  session_id: string
  action: ComputerAction
  approval_id?: string | null
  /** Override the broker's settle delay; the renderer sends a short one for direct user input. */
  settle_delay_ms?: number
}): Promise<ComputerActionResult> {
  return request('/api/v1/computer/exec', {
    method: 'POST',
    body: JSON.stringify(body)
  })
}

/**
 * Live browser viewport. Opens the daemon's SSE screencast stream and invokes
 * `onFrame` with each base64-encoded JPEG frame as it arrives. Resolves when
 * the stream ends (normally or because `signal` aborted it).
 */
export async function streamComputerPreview(
  sessionId: string,
  onFrame: (data: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const daemon = await connection()
  const headers = new Headers({ accept: 'text/event-stream' })
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(
    `${daemon.address}/api/v1/computer/sessions/${encodeURIComponent(sessionId)}/stream`,
    { headers, signal }
  )
  if (!response.ok || !response.body) {
    throw new Error(`Preview stream failed with status ${response.status}.`)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let separator = buffer.indexOf('\n\n')
    while (separator >= 0) {
      const frame = buffer.slice(0, separator)
      buffer = buffer.slice(separator + 2)
      const line = frame.split('\n').find((entry) => entry.startsWith('data:'))
      if (line) {
        const payload = line.slice(5).trimStart()
        if (payload) onFrame(payload)
      }
      separator = buffer.indexOf('\n\n')
    }
  }
}

export async function decideComputerApproval(
  approvalId: string,
  approve: boolean
): Promise<{ result: ComputerActionResult | null }> {
  return request(`/api/v1/computer/approvals/${encodeURIComponent(approvalId)}`, {
    method: 'POST',
    body: JSON.stringify({ approve })
  })
}

export async function parseFaraOutput(text: string): Promise<FaraParseResult> {
  return request('/api/v1/computer/parse-fara', {
    method: 'POST',
    body: JSON.stringify({ text })
  })
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
    /** Screenshot→action specialists (Fara1.5 and similar). */
    computer_use?: boolean
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
  /** When true, omit prior-turn reasoning from the next model request. Default keeps it. */
  drop_reasoning_between_turns?: boolean
  binary_override: string | null
  mlx_lm_python?: string | null
  mlx_vlm_python?: string | null
  vllm_python?: string | null
  vllm_model?: string | null
  vllm_models?: VllmModelSettings[]
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
  /** Flat ceiling for one generation job; 0 derives it from frames and steps. */
  generation_timeout_secs?: number
  voice_python?: string | null
  default_voice_model?: string | null
  default_voice_persona?: string | null
  build_jobs: number
  extra_model_library_paths: string[]
  generation_memory_policy: 'auto' | 'coresident' | 'exclusive'
  generation_memory_headroom_mb: number
  reload_llm_after_generation: boolean
  /** Chat `run_javascript` sandbox profile and optional limit overrides. */
  javascript_sandbox?: JavascriptSandboxSettings
  /** Web search backend: `duckduckgo` (keyless) or `brave` (paid API). */
  web_search_provider?: 'duckduckgo' | 'brave'
  /** Brave Search API key. Required when web_search_provider is `brave`. */
  brave_api_key?: string | null
  /** SafeSearch level: `moderate`, `strict`, or `off`. */
  web_safesearch?: 'moderate' | 'strict' | 'off'
  /** Default region/locale for web search, e.g. `us-en`, `wt-wt`. */
  web_search_region?: string | null
}

export type VllmModelSettings = {
  repository: string
  revision?: string | null
  context_size?: number | null
  dtype?: string | null
  gpu_memory_utilization?: number | null
  tensor_parallel_size?: number | null
  trust_remote_code: boolean
  /** When true (default), vLLM starts with prefix caching enabled. */
  prefix_caching?: boolean
  extra_args: string[]
}

export type JsSandboxProfile = 'strict' | 'default' | 'roomy' | 'custom'

export type JavascriptSandboxSettings = {
  profile?: JsSandboxProfile
  capture_console?: boolean | null
  timeout_ms?: number | null
  memory_mb?: number | null
  max_code_bytes?: number | null
  max_output_chars?: number | null
  max_stack_kb?: number | null
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
  vram_bytes: number | null
  gpu_offload_memory_bytes: number | null
  usable_model_memory_bytes: number | null
  gpu: string | null
  gpu_arch: string | null
  amd_apu: boolean
  intel_igpu: boolean
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
  /** Where the tool was found, which is not always where it was expected. */
  path?: string | null
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

export type ToolchainNeeds = {
  customRuntimes: boolean
  voice: boolean
  computerUse: boolean
  video: boolean
}

function toolchainQuery(needs: ToolchainNeeds): string {
  const params = new URLSearchParams({
    custom_runtimes: String(needs.customRuntimes),
    voice: String(needs.voice),
    computer_use: String(needs.computerUse),
    video: String(needs.video)
  })
  return `?${params.toString()}`
}

export function fetchToolchainStatus(needs?: ToolchainNeeds): Promise<ToolchainStatus> {
  return request(`/api/v1/toolchain${needs ? toolchainQuery(needs) : ''}`)
}

export function setupToolchain(
  needs: ToolchainNeeds
): Promise<{ status: ToolchainStatus; output: string }> {
  return request('/api/v1/toolchain', {
    method: 'POST',
    body: JSON.stringify({
      custom_runtimes: needs.customRuntimes,
      voice: needs.voice,
      computer_use: needs.computerUse,
      video: needs.video
    })
  })
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

/** Forget a finished, failed, or cancelled job so it leaves the list. */
export async function dismissDownloadJob(jobId: string): Promise<void> {
  await request('/api/v1/models/download/dismiss', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId })
  })
}

/** Forget every settled job at once. Resolves how many were cleared. */
export async function dismissFinishedDownloadJobs(): Promise<number> {
  const payload = await request<{ dismissed: number }>('/api/v1/models/downloads/finished', {
    method: 'DELETE'
  })
  return payload.dismissed
}

/** Put a paused, failed, or cancelled job back in line; it resumes in place. */
export async function resumeDownloadJob(jobId: string): Promise<void> {
  await request('/api/v1/models/download/resume', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId })
  })
}

/** Cancel a source build represented by a row in the shared activity tray. */
export async function cancelBuildJob(jobId: string): Promise<void> {
  await request('/api/v1/runtimes/build/cancel-job', {
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
  revision = 'main',
  engine: 'llama.cpp' | 'whisper.cpp' = 'llama.cpp'
): Promise<{ job_id: string }> {
  return request('/api/v1/models/download/queue', {
    method: 'POST',
    body: JSON.stringify({ repo_id: repoId, filename, revision, engine })
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
    source?: string
    correlation_id?: string
    status?: string
    metadata?: Record<string, unknown>
  }
): Promise<Message> {
  return request(`/api/v1/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify(message)
  })
}

/**
 * Relabel or finalize a stored message. Used to mark a turn queued, cancelled,
 * or superseded; it never removes the message.
 */
export function updateMessage(
  conversationId: string,
  messageId: string,
  patch: {
    content?: string | ContentPart[]
    status?: string
    metadata?: Record<string, unknown>
  }
): Promise<Message> {
  return request(`/api/v1/conversations/${conversationId}/messages/${messageId}`, {
    method: 'PATCH',
    body: JSON.stringify(patch)
  })
}

export function getConversation(conversationId: string): Promise<Conversation> {
  return request(`/api/v1/conversations/${conversationId}`)
}

/**
 * Bind an agent session to a conversation, retitle it, or store the compact
 * summary. Pass `agent_session_id: null` to unbind.
 */
export function updateConversation(
  conversationId: string,
  update: {
    title?: string
    agent_session_id?: string | null
    summary?: string
  }
): Promise<Conversation> {
  return request(`/api/v1/conversations/${conversationId}`, {
    method: 'PATCH',
    body: JSON.stringify(update)
  })
}

export function deleteConversation(conversationId: string): Promise<void> {
  return request(`/api/v1/conversations/${conversationId}`, { method: 'DELETE' })
}

/**
 * Transcribe one finished utterance through the daemon's ASR path.
 *
 * `engine` picks between the installed interfaces: `streaming-asr` runs the
 * Nemotron worker, and omitting it takes the daemon's default, which is
 * whisper.cpp or WhisperKit. The utterance is already complete either way, so
 * this asks for the collected text rather than an SSE stream.
 */
export type Transcription = {
  text: string
  /**
   * Which ASR interface actually served this, which is not always the one asked
   * for: `auto` sends no preference and the daemon picks. Reported so a session
   * can say what transcribed it rather than what it hoped would.
   */
  engine: string
  /** How long the daemon spent on the audio: decode, convert, and engine. */
  durationMs: number | null
}

export async function transcribeAudio(
  wav: Uint8Array,
  options: { signal?: AbortSignal; engine?: string } = {}
): Promise<Transcription> {
  let binary = ''
  for (let index = 0; index < wav.length; index += 1) binary += String.fromCharCode(wav[index])
  const payload = await request<{ text?: string; engine?: string; duration_ms?: number }>(
    '/v1/audio/transcriptions',
    {
      method: 'POST',
      signal: options.signal,
      body: JSON.stringify({
        file_base64: btoa(binary),
        mime_type: 'audio/wav',
        ...(options.engine ? { engine: options.engine } : {})
      })
    }
  )
  return {
    text: (payload.text ?? '').trim(),
    engine: payload.engine ?? options.engine ?? 'unknown',
    durationMs: typeof payload.duration_ms === 'number' ? payload.duration_ms : null
  }
}

export type ClientToolCall = {
  id: string
  name: string
  arguments: string
}

export type TranscriptMessagePayload = {
  role: string
  content: string | ContentPart[] | null
  tool_calls?: unknown[] | null
  tool_call_id?: string | null
  reasoning_content?: string | null
}

export function reasoningAfterTranscriptBoundary(
  reasoning: string,
  message: TranscriptMessagePayload
): string {
  return message.role === 'assistant' ? '' : reasoning
}

export type StreamCompletionResult = {
  responseText: string
  reasoningText: string
  toolRecords: ToolCallRecord[]
  clientToolCalls: ClientToolCall[]
  transcript: TranscriptMessagePayload[]
  /** Present for local engines, measured from their token stream. */
  generationStats?: {
    prompt_tokens?: number | null
    completion_tokens: number
    decode_duration_ms: number
  }
}

export type ToolCallRecord = {
  call_id: string
  name: string
  arguments: string
  output: string
  is_error: boolean
  /** Media produced by this tool, available immediately while it is running. */
  media?: Array<{
    sha256: string
    mime_type: string
    name?: string | null
    /** Attachments re-hung on a record by chatDisplay use this shape. */
    original_name?: string | null
  }>
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

export type PrefillProgress = {
  total: number
  cached: number
  processed: number
  elapsed_ms: number
  context_total?: number | null
}

export function prefillProgressLabel(progress: PrefillProgress): string {
  const processed = Math.min(progress.processed, progress.total)
  const prompt = `Prefilling ${processed.toLocaleString()} / ${progress.total.toLocaleString()} tokens`
  if (!progress.context_total) return prompt
  return `${prompt} · context ${progress.total.toLocaleString()} / ${progress.context_total.toLocaleString()}`
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

function isHarmonyModel(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return lower.includes('gpt-oss') || lower.includes('gpt_oss') || lower.includes('gptoss')
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
    onReasoning?: (token: string) => void
    onLoad?: (event: { phase: string; message: string }) => void
    onPrefill?: (event: PrefillProgress) => void
    /** Override the runtime's default thinking mode for this completion. */
    enableReasoning?: boolean
    /** When set, overrides a live settings fetch for drop_reasoning_between_turns. */
    dropReasoningBetweenTurns?: boolean
  }
): Promise<StreamCompletionResult> {
  const daemon = await connection()
  const toolChoice =
    options?.toolChoice === 'auto' || options?.toolChoice === 'none'
      ? options.toolChoice
      : options?.toolChoice
        ? options.toolChoice
        : undefined
  const dropReasoningBetweenTurns =
    options?.dropReasoningBetweenTurns ??
    (await runtimeSettings()
      .then((settings) => settings.drop_reasoning_between_turns ?? false)
      .catch(() => false))
  const response = await fetch(`${daemon.address}/v1/chat/completions`, {
    method: 'POST',
    signal,
    headers: {
      'content-type': 'application/json',
      'x-brazier-mode': 'chat',
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
      ...(typeof options?.enableReasoning === 'boolean'
        ? { enable_reasoning: options.enableReasoning }
        : {}),
      messages: messagesForCompletion(messages, {
        dropReasoningBetweenTurns,
        harmony: isHarmonyModel(model)
      })
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
  let reasoningText = ''
  let generationStats: StreamCompletionResult['generationStats']
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
            reasoning_content?: string
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
          prefill?: PrefillProgress
          generation?: {
            prompt_tokens?: number | null
            completion_tokens?: number
            decode_duration_ms?: number
          }
        }
        error?: { message?: string }
      }
      if (chunk.error?.message) {
        throw new GenerationFailure(chunk.error.message, chunk.brazier?.fork_hints ?? [])
      }
      if (chunk.brazier?.load) options?.onLoad?.(chunk.brazier.load)
      if (chunk.brazier?.prefill) options?.onPrefill?.(chunk.brazier.prefill)
      const generation = chunk.brazier?.generation
      if (
        typeof generation?.completion_tokens === 'number' &&
      typeof generation.decode_duration_ms === 'number'
      ) {
        generationStats = {
          prompt_tokens: generation.prompt_tokens,
          completion_tokens: generation.completion_tokens,
          decode_duration_ms: generation.decode_duration_ms
        }
      }
      if (chunk.brazier?.tool_call) {
        toolRecords.push(chunk.brazier.tool_call)
        options?.onToolCall?.(chunk.brazier.tool_call)
      }
      if (chunk.brazier?.transcript_message) {
        transcript.push(chunk.brazier.transcript_message)
        // An assistant transcript message commits the reasoning accumulated for
        // that internal tool round. Keep only subsequent reasoning for the
        // eventual final assistant message instead of saving every earlier
        // round a second time on the final response.
        reasoningText = reasoningAfterTranscriptBoundary(
          reasoningText,
          chunk.brazier.transcript_message
        )
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
      const reasoningToken = chunk.choices?.[0]?.delta?.reasoning_content
      if (reasoningToken) {
        reasoningText += reasoningToken
        options?.onReasoning?.(reasoningToken)
      }
      const token = chunk.choices?.[0]?.delta?.content
      if (token) {
        responseText += token
        onToken(token)
      }
    }
  }
  return { responseText, reasoningText, toolRecords, clientToolCalls, transcript, generationStats }
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

export async function fetchManagedLlamaStatus(force = false): Promise<ManagedEngineStatus> {
  return request(`/api/v1/engines/llama.cpp/managed-status${force ? '?force=1' : ''}`)
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

export async function fetchManagedWhisperStatus(
  force = false
): Promise<ManagedEngineStatus & { managed_supported: boolean; note?: string | null }> {
  return request(`/api/v1/engines/whisper.cpp/managed-status${force ? '?force=1' : ''}`)
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

export async function fetchManagedSdcppStatus(force = false): Promise<ManagedEngineStatus> {
  return request(`/api/v1/engines/stable-diffusion.cpp/managed-status${force ? '?force=1' : ''}`)
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
  /** First-frame conditioning (image-to-video). */
  init_image_blob?: string
  /** Last-frame conditioning for first/last-frame models (`--end-img`). */
  end_image_blob?: string
  /** Reference images for Ref2VA conditioning. */
  ref_image_blobs?: string[]
  /** Reference videos for Ref2VA conditioning; frames are sampled to 24 fps. */
  ref_video_blobs?: string[]
  /** Soundtracks paired by index with `ref_video_blobs`. */
  ref_video_audio_blobs?: string[]
  /** Standalone audio references for Ref2VA conditioning. */
  ref_audio_blobs?: string[]
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

/** A generation in flight, including one a model started on its own. */
export type ActiveGeneration = {
  id: string
  modality: 'image' | 'video'
  model_id: string
  prompt: string
  negative_prompt?: string | null
  /** Blob of the conditioning image, when the job was given one. */
  init_image_blob?: string | null
  /** Blob of the ending image, for first/last-frame conditioning. */
  end_image_blob?: string | null
  /** Blobs behind reference conditioning (images, videos, audio). */
  conditioning_blobs?: string[]
  /** Whether the person or a model asked for this. */
  origin: 'user' | 'model'
  elapsed_secs: number
  timeout_secs: number
  current_step: number
  total_steps: number
}

export async function fetchActiveGeneration(): Promise<ActiveGeneration | null> {
  const payload = await request<{ active: ActiveGeneration | null }>('/api/v1/generate/active')
  return payload.active
}

/** Stop the running generation. Resolves false when nothing was running. */
export async function cancelGeneration(): Promise<boolean> {
  const payload = await request<{ cancelled: boolean }>('/api/v1/generate/cancel', {
    method: 'POST'
  })
  return payload.cancelled
}

// ---------------------------------------------------------------------------
// Startup model recommendations
// ---------------------------------------------------------------------------

/** The things a first-run installation can be set up for. */
export type RecommendationCategory = 'text' | 'agent' | 'image' | 'video' | 'voice' | 'computer_use'

/** A chat or agent model, resolved to the exact files to download. */
export type RepoRecommendation = {
  id: string
  label: string
  repo_id: string
  summary?: string | null
  /** The rung chosen for this machine, when one was recognised. */
  quant?: string | null
  /** Files to fetch, in order. More than one when the quant is sharded. */
  files?: string[]
  bytes?: number
  /** Nothing fitted the memory budget; this is simply the smallest build. */
  tight?: boolean
  /** Why this could not be sized — a missing model, or an unreachable Hub. */
  unresolved?: string
  /** Why the chat model is standing in for the tier's own agent model. */
  substituted?: string
  /** Extra files to fetch after the main weights, in order. E.g. a vision
   *  projector that `llama.cpp` auto-attaches when it sits next to the model. */
  companion_files?: string[]
  /** Automatically associated speculative-decoding draft GGUFs. */
  draft_files?: string[]
  /** Companion file(s) requested but not published by the repository. */
  unresolved_companions?: string
  /** Draft file(s) requested but not published by the repository. */
  unresolved_drafts?: string
  /** A runtime fork that is built and activated with this recommendation. */
  runtime_build?: {
    engine: string
    repository: string
    revision: string
    target: string
    label: string
  }
  /** Context window in tokens this model should default to on this machine. */
  context_tokens?: number
  /** This repository requires a Hugging Face token before it can be downloaded. */
  gated?: boolean
}

/** An image or video model, named by stable-diffusion.cpp bundle. */
export type BundleRecommendation = {
  id: string
  label: string
  bundle_id?: string | null
  variant?: string | null
  /** Separate text-to-video and image-to-video models, when it is split. */
  parts?: Array<{ bundle_id: string; role: string; label: string; variant?: string | null }>
  summary?: string | null
  unresolved?: string
  /** At least one bundle component requires a Hugging Face token. */
  gated?: boolean
}

export type VoiceRecommendationModel = {
  id: string
  label: string
  /** `personaplex` or `whisper`. */
  kind: string
  repo_id: string
  /** This snapshot requires a Hugging Face token and accepted access terms. */
  gated?: boolean
  filename?: string | null
  summary?: string | null
}

/** A category whose recommendation moved on since it was installed. */
export type PendingSwap = {
  category: RecommendationCategory
  installed_id: string
  recommended_id: string
  recommended_label: string
  summary?: string | null
}

export type RecommendationState = {
  suppressed: boolean
  installed: Record<
    string,
    { recommendation_id: string; model_id?: string | null; installed_at: string }
  >
  dismissed?: string[]
}

export type Recommendations = {
  memory_bytes: number | null
  /** Whether the tier was chosen by video memory or by system memory. */
  memory_source?: 'vram' | 'system'
  tier_gb: number | null
  /** Why there is nothing to recommend, when there is nothing. */
  reason?: string
  categories: {
    text?: RepoRecommendation
    agent?: RepoRecommendation
    image?: BundleRecommendation
    video?: BundleRecommendation
    computer_use?: RepoRecommendation
  }
  /** Alternative agent models for this tier, including the default first. */
  agent_options?: RepoRecommendation[]
  voice?: { summary?: string | null; models: VoiceRecommendationModel[] } | null
  state: RecommendationState
  swaps: PendingSwap[]
}

export function fetchRecommendations(): Promise<Recommendations> {
  return request('/api/v1/recommendations')
}

/** Record that a category was set up from a recommendation. */
export function recordRecommendationInstall(
  category: RecommendationCategory,
  recommendationId: string,
  modelId?: string
): Promise<RecommendationState> {
  return request('/api/v1/recommendations/installed', {
    method: 'POST',
    body: JSON.stringify({
      category,
      recommendation_id: recommendationId,
      model_id: modelId ?? null
    })
  })
}

export type RecommendationSetup = {
  id: string
  recommendation_id: string
  categories: string[]
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
  error?: string | null
  steps: Array<{ label: string; kind: string; job_id?: string | null; status: string }>
}

type QueuedGgufWork = {
  kind: 'gguf'
  repo_id: string
  filename: string
  revision: string
  engine: 'llama.cpp' | 'whisper.cpp'
}

export async function startRecommendationSetup(body: {
  recommendation_id: string
  categories: RecommendationCategory[]
  works: QueuedGgufWork[]
  required_bytes: number
  build?: { engine: string; repository: string; revision: string; target: string; jobs?: number }
}): Promise<RecommendationSetup> {
  return (await request<{ setup: RecommendationSetup }>('/api/v1/recommendations/setups', {
    method: 'POST',
    body: JSON.stringify(body)
  })).setup
}

export async function listRecommendationSetups(): Promise<RecommendationSetup[]> {
  return (await request<{ data: RecommendationSetup[] }>('/api/v1/recommendations/setups')).data
}

/** Stop mentioning changed recommendations, or decline one particular swap. */
export function updateRecommendationState(patch: {
  suppressed?: boolean
  dismiss?: string
}): Promise<RecommendationState> {
  return request('/api/v1/recommendations/state', {
    method: 'PUT',
    body: JSON.stringify(patch)
  })
}

// ---------------------------------------------------------------------------
// Adapters and per-model configuration
// ---------------------------------------------------------------------------

export type AdapterKind = 'lora' | 'controlnet'

/** One installed LoRA or ControlNet, with the engines that can load it. */
export type Adapter = {
  id: string
  kind: AdapterKind
  name: string
  path: string
  size_bytes?: number
  /** `llama.cpp`, `mlx-lm`, `mlx-vlm`, or `stable-diffusion.cpp`. */
  engines: string[]
  /** Registered from outside the library, so Brazier will not delete it. */
  external: boolean
  source_repo?: string
}

export type LoraBinding = {
  adapter_id: string
  path?: string | null
  scale: number
  enabled: boolean
}

export type ControlNetBinding = {
  adapter_id: string
  path?: string | null
  strength: number
  image_path?: string | null
  /** Run the ControlNet on the CPU, keeping VRAM for the main model. */
  cpu: boolean
  enabled: boolean
}

/** Which family of settings a model takes. */
export type ModelKind = 'text' | 'image' | 'video' | 'transcription' | 'voice'

/**
 * Every field is optional and means *override*: left unset, a setting falls
 * through to the global inference defaults and then to the engine's own.
 */
export type TextProfile = {
  context_size?: number | null
  batch_size?: number | null
  ubatch_size?: number | null
  threads?: number | null
  gpu_layers?: number | null
  flash_attention?: boolean | null
  kv_cache_type_k?: string | null
  kv_cache_type_v?: string | null
  jinja?: boolean | null
  /** Custom Jinja chat template. Null keeps the GGUF-bundled template. */
  chat_template?: string | null
  mlock?: boolean | null
  no_mmap?: boolean | null
  rope_scaling?: string | null
  rope_freq_base?: number | null
  rope_freq_scale?: number | null
  yarn_orig_ctx?: number | null
  n_cpu_moe?: number | null
  main_gpu?: number | null
  tensor_split?: string | null
  split_mode?: string | null
  cache_reuse?: number | null
  defrag_threshold?: number | null
  /** Use llama.cpp multi-token prediction. Null auto-detects MTP GGUFs. */
  mtp?: boolean | null
  /** Maximum MTP draft tokens per step (default 2). */
  mtp_draft_tokens?: number | null
  /** Draft GGUF filename beside the model, or an absolute path. Null auto-detects. */
  speculative_draft_model?: string | null
  /** draft-simple, draft-dflash, or draft-dspark. Null infers from the filename. */
  speculative_draft_type?: string | null
  temperature?: number | null
  top_p?: number | null
  top_k?: number | null
  min_p?: number | null
  typical_p?: number | null
  repeat_penalty?: number | null
  repeat_last_n?: number | null
  presence_penalty?: number | null
  frequency_penalty?: number | null
  dry_multiplier?: number | null
  dry_base?: number | null
  dry_allowed_length?: number | null
  mirostat?: number | null
  mirostat_tau?: number | null
  mirostat_eta?: number | null
  seed?: number | null
  max_tokens?: number | null
  stop?: string[]
  enable_reasoning?: boolean | null
  reasoning_budget_tokens?: number | null
  system_prompt?: string | null
  /** Model id for agent subagents. Null/unset means the parent model. */
  subagent_model?: string | null
  /** Per-subagent context. Null/unset inherits the parent context. */
  /** Max concurrent subagents. Null inherits the default of 2. */
  max_subagents?: number | null
  /**
   * When true, llama.cpp starts with enough `--parallel` slots for concurrent
   * subagent generation. Reloads the server and allocates the configured
   * per-agent context to every slot.
   */
  parallel_subagents?: boolean | null
  loras?: LoraBinding[]
  extra_args?: string[]
}

export type DiffusionProfile = {
  width?: number | null
  height?: number | null
  steps?: number | null
  cfg_scale?: number | null
  guidance?: number | null
  img_cfg_scale?: number | null
  sampling_method?: string | null
  schedule?: string | null
  clip_skip?: number | null
  seed?: number | null
  batch_count?: number | null
  strength?: number | null
  eta?: number | null
  slg_scale?: number | null
  skip_layers?: string | null
  skip_layer_start?: number | null
  skip_layer_end?: number | null
  flow_shift?: number | null
  threads?: number | null
  vae_tiling?: boolean | null
  vae_on_cpu?: boolean | null
  clip_on_cpu?: boolean | null
  diffusion_fa?: boolean | null
  auto_fit?: boolean | null
  max_vram?: number | null
  params_backend?: string | null
  stream_layers?: boolean | null
  offload_to_cpu?: boolean | null
  rng?: string | null
  negative_prompt?: string | null
  video_frames?: number | null
  fps?: number | null
  loras?: LoraBinding[]
  control_nets?: ControlNetBinding[]
  extra_args?: string[]
}

export type TranscriptionProfile = {
  language?: string | null
  translate?: boolean | null
  beam_size?: number | null
  best_of?: number | null
  temperature?: number | null
  max_context?: number | null
  max_len?: number | null
  split_on_word?: boolean | null
  word_threshold?: number | null
  entropy_threshold?: number | null
  logprob_threshold?: number | null
  no_speech_threshold?: number | null
  no_fallback?: boolean | null
  suppress_nst?: boolean | null
  threads?: number | null
  flash_attention?: boolean | null
  initial_prompt?: string | null
  lookahead?: number | null
  extra_args?: string[]
}

export type VoiceProfile = {
  persona_text?: string | null
  voice_id?: string | null
  voice_prompt_path?: string | null
  quantization?: number | null
  extra_args?: string[]
}

/** A model's overrides, tagged with the kind of model they apply to. */
export type ModelProfile =
  | ({ kind: 'text' } & TextProfile)
  | ({ kind: 'image' } & DiffusionProfile)
  | ({ kind: 'video' } & DiffusionProfile)
  | ({ kind: 'transcription' } & TranscriptionProfile)
  | ({ kind: 'voice' } & VoiceProfile)

export type ModelSettingsResponse = {
  models: Record<string, ModelProfile>
  /** The kind each installed model takes, so the right fields are offered. */
  kinds: Record<string, ModelKind>
}

export function fetchModelSettings(): Promise<ModelSettingsResponse> {
  return request('/api/v1/models/settings')
}

export async function saveModelProfile(
  modelId: string,
  profile: ModelProfile
): Promise<Record<string, ModelProfile>> {
  const response = await request<{ models: Record<string, ModelProfile> }>(
    '/api/v1/models/settings',
    { method: 'PUT', body: JSON.stringify({ model_id: modelId, profile }) }
  )
  return response.models
}

export async function resetModelProfile(
  modelId: string
): Promise<Record<string, ModelProfile>> {
  const response = await request<{ models: Record<string, ModelProfile> }>(
    '/api/v1/models/settings/reset',
    { method: 'POST', body: JSON.stringify({ model_id: modelId }) }
  )
  return response.models
}

export type ModelChatTemplateResponse = {
  model_id: string
  chat_template: string | null
  source: 'gguf' | 'missing' | 'unsupported'
}

/** Jinja chat template embedded in a GGUF (`tokenizer.chat_template`). */
export function fetchModelChatTemplate(modelId: string): Promise<ModelChatTemplateResponse> {
  const parameters = new URLSearchParams({ model_id: modelId })
  return request(`/api/v1/models/chat-template?${parameters.toString()}`)
}

export async function listAdapters(): Promise<Adapter[]> {
  return (await request<{ data: Adapter[] }>('/api/v1/adapters')).data
}

export async function registerAdapter(
  kind: AdapterKind,
  path: string,
  name?: string
): Promise<Adapter> {
  const response = await request<{ adapter: Adapter }>('/api/v1/adapters/register', {
    method: 'POST',
    body: JSON.stringify({ kind, path, name })
  })
  return response.adapter
}

export async function forgetAdapter(id: string): Promise<void> {
  await request('/api/v1/adapters/forget', {
    method: 'POST',
    body: JSON.stringify({ id })
  })
}

export async function deleteAdapter(id: string): Promise<void> {
  await request('/api/v1/adapters/delete', {
    method: 'POST',
    body: JSON.stringify({ id })
  })
}

export async function downloadAdapter(
  kind: AdapterKind,
  repoId: string,
  filename: string,
  onProgress?: (event: ProgressEvent) => void
): Promise<{ path: string; bytes: number }> {
  const final = await readProgressSse(
    '/api/v1/adapters/download?stream=true',
    { method: 'POST', body: JSON.stringify({ kind, repo_id: repoId, filename }) },
    onProgress ?? (() => {})
  )
  const result = final.result as { path?: string; bytes?: number } | undefined
  if (!result?.path) throw new Error('The adapter download finished without a file.')
  return { path: result.path, bytes: result.bytes ?? 0 }
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

/** One size a component is offered in — the GGUF quant choice, for bundles. */
export type SdcppComponentVariant = {
  label: string
  path: string
  repo_id?: string | null
  /** Optional sd-cli flag override, e.g. `tae` instead of `vae`. */
  flag?: string | null
  /** Replace this component in the existing model directory. */
  in_place?: boolean
  approx_bytes?: number | null
  note?: string | null
}

export type SdcppBundleComponent = {
  repo_id: string
  path: string
  /** sd-cli flag without the dashes; null for a self-contained checkpoint. */
  flag?: string | null
  role: string
  gated: boolean
  approx_bytes?: number | null
  /** Sizes on offer. Empty or absent means the single file named above. */
  variants?: SdcppComponentVariant[]
}

/** The license agreement a bundle requires the user to accept before install. */
export type SdcppConsentRequirement = {
  /** Stable license identifier, shared across bundles under the same terms. */
  id: string
  /** Version of the terms the user must have seen. */
  version: string
  /** Link to the full agreement text. */
  url: string
  /** Plain-language summary shown in the consent dialog. */
  summary: string
  /** Whether this machine has recorded acceptance. */
  accepted?: boolean
}

export type SdcppBundle = {
  id: string
  label: string
  modality: 'image' | 'video'
  key: string
  summary: string
  license?: string | null
  /** Full agreement text the bundle's license links to. */
  license_url?: string | null
  /** Version of the terms the person must have accepted. */
  license_version?: string | null
  /** Plain-language summary of what accepting the agreement means. */
  license_summary?: string | null
  /** Gated behind an explicit license acceptance before it can be installed. */
  requires_license_acceptance?: boolean
  /** The agreement to accept, when the bundle requires one. */
  consent?: SdcppConsentRequirement | null
  model_id: string
  installed: boolean
  gated: boolean
  approx_bytes?: number | null
  /** `builtin` ships with the app; `custom` lives in the data directory. */
  origin: 'builtin' | 'custom'
  /** Whether the model can start from a supplied image. */
  supports_init_image?: boolean
  /** Conditioning surface the installed checkpoint offers, when installed. */
  conditioning?: 'text' | 'init_image' | 'first_last_frame' | 'references' | null
  /** Shown in the short list rather than behind "show every model". */
  featured?: boolean
  defaults: SdcppDefaults
  components: SdcppBundleComponent[]
}

/** Record acceptance of a bundle's license agreement. */
export function acceptSdcppLicense(bundleId: string): Promise<{
  license_id: string
  version: string
  accepted: boolean
  bundle_id: string
}> {
  return request('/api/v1/models/sdcpp/consent', {
    method: 'POST',
    body: JSON.stringify({ bundle_id: bundleId })
  })
}

/**
 * Resolve a bundle's chosen sizes into the concrete one to install.
 *
 * Picking a quant produces a different set of files, so the bundle is sent in
 * full rather than by id, and its key gains the chosen sizes — two quants of
 * one model are separate installs, exactly as two GGUF quants are.
 */
export function resolveBundleVariants(
  bundle: SdcppBundle,
  choices: Record<number, string>
): SdcppBundle {
  const picked: string[] = []
  const components = bundle.components.map((component, index) => {
    const wanted = choices[index]
    const variant = component.variants?.find((option) => option.label === wanted)
    if (!variant) return component
    if (!variant.in_place) picked.push(variant.label)
    return {
      ...component,
      repo_id: variant.repo_id || component.repo_id,
      flag: variant.flag ?? component.flag,
      path: variant.path,
      approx_bytes: variant.approx_bytes ?? component.approx_bytes
    }
  })
  if (picked.length === 0) return { ...bundle, components }
  const slug = picked
    .join('-')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
  return {
    ...bundle,
    id: `${bundle.id}-${slug}`,
    key: `${bundle.key}-${slug}`,
    components
  }
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

export function deactivateRuntime(id: string): Promise<{ id: string; deactivated: boolean }> {
  return request('/api/v1/runtimes/deactivate', {
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

export type ModelResidency = {
  placement: 'gpu' | 'cpu_offload' | 'unavailable'
  backend: string
  description: string
  gpu_bytes: number | null
  cpu_bytes: number | null
  gpu_kv_bytes: number | null
  cpu_kv_bytes: number | null
  gpu_capacity_bytes: number | null
  aggregate_context_tokens: number | null
  parallel_slots: number | null
}

export type ModelLoadMode = 'chat' | 'agent'

export async function prepareModel(
  modelId: string,
  options?: {
    signal?: AbortSignal
    onLoad?: (event: { phase: string; message: string }) => void
    mode?: ModelLoadMode
  }
): Promise<ModelResidency | null> {
  const daemon = await connection()
  const response = await fetch(`${daemon.address}/api/v1/models/prepare?stream=true`, {
    method: 'POST',
    signal: options?.signal,
    headers: {
      'content-type': 'application/json',
      ...(daemon.api_key ? { authorization: `Bearer ${daemon.api_key}` } : {})
    },
    body: JSON.stringify({ model_id: modelId, mode: options?.mode ?? 'chat' })
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
        residency?: ModelResidency
        error?: { message?: string }
        brazier?: { fork_hints?: RuntimeForkHint[] }
      }
      if (chunk.error?.message) {
        throw new GenerationFailure(chunk.error.message, chunk.brazier?.fork_hints ?? [])
      }
      if (chunk.phase && chunk.message) {
        options?.onLoad?.({ phase: chunk.phase, message: chunk.message })
      }
      if (chunk.status === 'ready') return chunk.residency ?? null
    }
  }
  return null
}

/** Stop the currently resident local chat model, if any. */
export async function unloadModel(): Promise<void> {
  await request('/api/v1/models/loaded', { method: 'DELETE' })
}

export type BundledTool = {
  name: string
  title: string
  description?: string
  network: boolean
  source?: string
  /** Display name of the MCP server that supplied this tool. */
  server_name?: string
}

export type McpServer = {
  id: string
  name: string
  command: string
  args: string[]
  enabled: boolean
  tools: BundledTool[]
}

/**
 * An OpenAI-compatible server someone else is running.
 *
 * `has_api_key` rather than the key: the daemon never sends one back, so the UI
 * can say a key is set without ever holding it.
 */
export type RemoteConnection = {
  id: string
  label: string
  base_url: string
  enabled: boolean
  has_api_key: boolean
  /** When true, Brazier sends llama.cpp KV cache hints (`cache_prompt`, `id_slot`). */
  llama_cpp_compatible: boolean
}

export async function listRemoteConnections(): Promise<RemoteConnection[]> {
  return (await request<{ data: RemoteConnection[] }>('/api/v1/remote/connections')).data
}

export async function saveRemoteConnection(connection: {
  id: string
  label: string
  base_url: string
  /** Omit to keep the stored key; empty string clears it. */
  api_key?: string
  enabled: boolean
  llama_cpp_compatible?: boolean
}): Promise<RemoteConnection[]> {
  return (
    await request<{ data: RemoteConnection[] }>('/api/v1/remote/connections', {
      method: 'PUT',
      body: JSON.stringify(connection)
    })
  ).data
}

export async function deleteRemoteConnection(id: string): Promise<RemoteConnection[]> {
  return (
    await request<{ data: RemoteConnection[] }>(
      `/api/v1/remote/connections/${encodeURIComponent(id)}`,
      { method: 'DELETE' }
    )
  ).data
}

export async function testRemoteConnection(
  id: string
): Promise<{ reachable: boolean; models: string[]; error?: string }> {
  return request(`/api/v1/remote/connections/${encodeURIComponent(id)}/test`, { method: 'POST' })
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
  reasoning_content?: string
}

function messageContentForApi(content: string | ContentPart[]): string | ContentPart[] {
  return content
}

function reasoningFromMessage(message: Message): string | undefined {
  const value = message.metadata?.reasoning_content
  return typeof value === 'string' && value.trim() ? value : undefined
}

export type MessagesForCompletionOptions = {
  /**
   * When true, omit reasoning_content from assistant messages before the latest
   * user turn. Current-turn tool-round reasoning is kept for Jinja.
   */
  dropReasoningBetweenTurns?: boolean
  /** gpt-oss / Harmony models need reasoning on prior tool-call turns too. */
  harmony?: boolean
}

export function messagesForCompletion(
  messages: Message[],
  options?: MessagesForCompletionOptions
): OpenAiChatMessage[] {
  const payload: OpenAiChatMessage[] = []
  for (const message of messages) {
    // This assistant-role message exists only to place generated media in the
    // human transcript. When enabled, a separate hidden system message carries
    // the same blob into model context on the next real user turn.
    if (message.metadata?.generated_media_display === true) continue
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
      const reasoning = reasoningFromMessage(message)
      payload.push({
        role: 'assistant',
        content:
          typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content),
        tool_calls: message.tool_calls as OpenAiToolCall[],
        ...(reasoning ? { reasoning_content: reasoning } : {})
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
    const reasoning = message.role === 'assistant' ? reasoningFromMessage(message) : undefined
    payload.push({
      role: message.role,
      content: messageContentForApi(message.content),
      ...(reasoning ? { reasoning_content: reasoning } : {})
    })
  }
  if (options?.dropReasoningBetweenTurns) {
    let lastUser = -1
    for (let index = 0; index < payload.length; index += 1) {
      if (payload[index]?.role === 'user') lastUser = index
    }
    if (lastUser >= 0) {
      for (let index = 0; index <= lastUser; index += 1) {
        const entry = payload[index]
        if (!entry || !('reasoning_content' in entry)) continue
        if (
          options.harmony &&
          entry.role === 'assistant' &&
          entry.tool_calls &&
          entry.tool_calls.length > 0
        ) {
          continue
        }
        delete entry.reasoning_content
      }
    }
  }
  return payload
}

export async function buildRuntime(
  engine: string,
  repository: string,
  revision: string,
  target: string,
  jobs: number,
  name: string | undefined,
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
        ...(name?.trim() ? { name: name.trim() } : {}),
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

/** Extension to suggest for a stored blob, from its MIME type. */
function extensionForMime(mimeType: string): string {
  const known: Record<string, string> = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/webp': 'webp',
    'image/gif': 'gif',
    'image/svg+xml': 'svg',
    'video/mp4': 'mp4',
    'video/webm': 'webm',
    'video/quicktime': 'mov',
    'application/pdf': 'pdf',
    'application/rtf': 'rtf',
    'text/rtf': 'rtf',
    'application/msword': 'doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'text/plain': 'txt',
    'text/markdown': 'md',
    'text/csv': 'csv',
    'application/json': 'json',
    'application/xml': 'xml'
  }
  const normalized = mimeType.split(';', 1)[0].trim().toLowerCase()
  const suffix = normalized.split('/').at(-1)
  return known[normalized] ?? (suffix && /^[a-z0-9]+$/.test(suffix) ? suffix : 'bin')
}

export function filenameWithExtension(
  name: string | null | undefined,
  mimeType: string,
  fallback: string
): string {
  const trimmed = name?.trim()
  if (!trimmed) return fallback
  return /\.[^./\\]+$/.test(trimmed) ? trimmed : `${trimmed}.${extensionForMime(mimeType)}`
}

/**
 * Save a stored blob to disk through a native save dialog.
 *
 * Resolves the chosen path, or null when the dialog was dismissed. Generated
 * media otherwise only exists inside the app's blob store, where it is
 * addressed by hash and effectively unreachable.
 */
export async function saveBlobToDisk(
  sha256: string,
  mimeType: string,
  suggestedName?: string | null
): Promise<string | null> {
  const daemon = await connection()
  const headers = new Headers()
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}/api/v1/blobs/${sha256}`, { headers })
  if (!response.ok) throw new Error(`Could not read that file (${response.status}).`)
  const bytes = await response.arrayBuffer()
  const extension = extensionForMime(mimeType)
  const fallback = `brazier-${sha256.slice(0, 12)}.${extension}`
  return window.brazier.saveFile(filenameWithExtension(suggestedName, mimeType, fallback), bytes)
}

/** Download the daemon's privacy-filtered diagnostic archive and save it natively. */
export async function saveSupportBundle(): Promise<string | null> {
  const daemon = await connection()
  const headers = new Headers()
  if (daemon.api_key) headers.set('authorization', `Bearer ${daemon.api_key}`)
  const response = await fetch(`${daemon.address}/api/v1/support/bundle`, { headers })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    throw new Error(
      payload?.error?.message ?? `Could not create support bundle (${response.status}).`
    )
  }
  const bytes = await response.arrayBuffer()
  return window.brazier.saveFile('brazier-support.zip', bytes)
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
