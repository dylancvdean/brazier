import type { LocalModel, ModelKind, ModelProfile, RuntimeEntry } from './api'

export function modelEngine(model: LocalModel | undefined): string {
  if (!model) return ''
  if (model.engine) return model.engine
  if (model.id.startsWith('sdcpp-image:') || model.id.startsWith('sdcpp-video:')) {
    return 'stable-diffusion.cpp'
  }
  if (model.id.startsWith('personaplex:')) return 'personaplex'
  if (model.id.startsWith('whisper:')) return 'whisper.cpp'
  if (model.id.startsWith('streaming-asr:')) return 'streaming-asr'
  if (model.id.startsWith('remote:')) return 'remote'
  return model.owned_by.replace(/^brazier:/, '')
}

export function engineLabel(engine: string): string {
  switch (engine) {
    case 'mlx-lm':
      return 'MLX · text'
    case 'mlx-vlm':
      return 'MLX · vision'
    case 'llama.cpp':
      return 'GGUF'
    case 'whisper.cpp':
      return 'ASR · whisper'
    case 'streaming-asr':
      return 'ASR · streaming'
    case 'stable-diffusion.cpp':
      return 'Gen · sd.cpp'
    case 'personaplex':
      return 'Voice · PersonaPlex'
    case 'remote':
      return 'Remote'
    default:
      return engine
  }
}

export function engineBadgeClass(engine: string): string {
  switch (engine) {
    case 'mlx-lm':
      return 'engine-badge mlx-lm'
    case 'mlx-vlm':
      return 'engine-badge mlx-vlm'
    case 'llama.cpp':
      return 'engine-badge gguf'
    case 'whisper.cpp':
      return 'engine-badge whisper'
    case 'streaming-asr':
      return 'engine-badge whisper'
    case 'stable-diffusion.cpp':
      return 'engine-badge gguf'
    case 'personaplex':
      return 'engine-badge mlx-lm'
    case 'remote':
      return 'engine-badge remote'
    default:
      return 'engine-badge'
  }
}

export function isChatModel(model: LocalModel | undefined): boolean {
  const engine = modelEngine(model)
  return (
    engine !== 'whisper.cpp' &&
    engine !== 'streaming-asr' &&
    engine !== 'stable-diffusion.cpp' &&
    engine !== 'personaplex' &&
    !model?.id.startsWith('sdcpp-image:') &&
    !model?.id.startsWith('sdcpp-video:') &&
    !model?.id.startsWith('personaplex:') &&
    !model?.id.startsWith('whisper:') &&
    !model?.id.startsWith('streaming-asr:')
  )
}

export function isImageGenModel(model: LocalModel | undefined): boolean {
  return Boolean(model?.id.startsWith('sdcpp-image:'))
}

export function isVideoGenModel(model: LocalModel | undefined): boolean {
  return Boolean(model?.id.startsWith('sdcpp-video:'))
}

export function isVoiceModel(model: LocalModel | undefined): boolean {
  return Boolean(model?.id.startsWith('personaplex:'))
}

/**
 * Merge per-model TextProfile capability overrides onto a model's advertised
 * capabilities. Null/undefined fields inherit whatever the daemon reported.
 */
export function mergeModelCapabilities(
  model: LocalModel | undefined,
  profile?: ModelProfile
): LocalModel['capabilities'] {
  if (!model) return undefined
  if (!profile || profile.kind !== 'text') return model.capabilities
  const hasOverride =
    profile.capability_vision != null ||
    profile.capability_reasoning != null ||
    profile.capability_tools != null ||
    profile.capability_computer_use != null
  if (!hasOverride) return model.capabilities

  const base = model.capabilities ?? {
    input_modalities: ['text'],
    output_modalities: ['text'],
    streaming: true,
    tools: false,
    reasoning: false
  }

  let inputModalities = [...base.input_modalities]
  if (profile.capability_vision === true && !inputModalities.includes('image')) {
    inputModalities.push('image')
  } else if (profile.capability_vision === false) {
    inputModalities = inputModalities.filter((modality) => modality !== 'image')
  }

  const reasoning =
    profile.capability_reasoning != null ? profile.capability_reasoning : base.reasoning
  const reasoningModes =
    profile.capability_reasoning === false
      ? []
      : profile.capability_reasoning === true
        ? base.reasoning_modes && base.reasoning_modes.length > 0
          ? base.reasoning_modes
          : ['off', 'on']
        : base.reasoning_modes

  const tools = profile.capability_tools != null ? profile.capability_tools : base.tools
  const computerUse =
    profile.capability_computer_use != null
      ? profile.capability_computer_use
      : base.computer_use

  return {
    ...base,
    input_modalities: inputModalities,
    reasoning,
    ...(reasoningModes !== undefined ? { reasoning_modes: reasoningModes } : {}),
    tools,
    ...(computerUse !== undefined ? { computer_use: computerUse } : {})
  }
}

export function withMergedCapabilities(model: LocalModel, profile?: ModelProfile): LocalModel {
  const capabilities = mergeModelCapabilities(model, profile)
  if (capabilities === model.capabilities) return model
  return { ...model, capabilities }
}

/**
 * Chat-class models that can drive Computer Use (screenshot → action).
 *
 * Allows any dedicated `computer_use` model as well as any vision-capable chat
 * model (Qwen3-VL, Qwen3.5, Gemma 4, Pixtral, vLLM Fara, and remote VLMs).
 */
export function isComputerUseModel(
  model: LocalModel | undefined,
  profile?: ModelProfile
): boolean {
  if (!model || !isChatModel(model)) return false
  const effective = profile ? withMergedCapabilities(model, profile) : model
  if (profile?.kind === 'text' && profile.capability_computer_use != null) {
    return profile.capability_computer_use
  }
  if (effective.capabilities?.computer_use === true) return true
  return Boolean(effective.capabilities?.input_modalities?.includes('image'))
}

/** Sort dedicated `computer_use` specialists ahead of general vision chat models. */
export function sortComputerUseModels(
  models: LocalModel[],
  profiles?: Record<string, ModelProfile>
): LocalModel[] {
  const isSpecialist = (model: LocalModel): boolean => {
    const profile = profiles?.[model.id]
    if (profile?.kind === 'text' && profile.capability_computer_use != null) {
      return profile.capability_computer_use
    }
    return model.capabilities?.computer_use === true
  }
  return [...models].sort((left, right) => {
    const leftRank = isSpecialist(left) ? 0 : 1
    const rightRank = isSpecialist(right) ? 0 : 1
    if (leftRank !== rightRank) return leftRank - rightRank
    return modelDisplayName(left.id, left).title.localeCompare(
      modelDisplayName(right.id, right).title
    )
  })
}

/**
 * Which family of advanced settings a model takes.
 *
 * Mirrors the daemon's own classification, which is the one that decides what
 * a stored profile is allowed to contain.
 */
export function modelKindFor(modelId: string): ModelKind {
  if (modelId.startsWith('sdcpp-image:')) return 'image'
  if (modelId.startsWith('sdcpp-video:')) return 'video'
  if (modelId.startsWith('whisper:') || modelId.startsWith('streaming-asr:')) {
    return 'transcription'
  }
  if (modelId.startsWith('personaplex:')) return 'voice'
  return 'text'
}

export function modelLibraryKey(modelId: string): string {
  if (modelId.startsWith('gguf-ext:')) {
    return modelId.replace(/^gguf-ext:\d+:/, '')
  }
  if (modelId.startsWith('mlx-vlm-ext:')) {
    return modelId.replace(/^mlx-vlm-ext:\d+:/, '')
  }
  if (modelId.startsWith('mlx-ext:')) {
    return modelId.replace(/^mlx-ext:\d+:/, '')
  }
  if (modelId.startsWith('gguf:')) {
    return modelId.slice('gguf:'.length)
  }
  if (modelId.startsWith('mlx-vlm:') || modelId.startsWith('mlx:')) {
    return modelId.replace(/^mlx(-vlm)?:/, '')
  }
  if (modelId.startsWith('sdcpp-image:')) {
    return modelId.slice('sdcpp-image:'.length)
  }
  if (modelId.startsWith('sdcpp-video:')) {
    return modelId.slice('sdcpp-video:'.length)
  }
  if (modelId.startsWith('personaplex:')) {
    return modelId.slice('personaplex:'.length)
  }
  if (modelId.startsWith('whisper:')) {
    return modelId.slice('whisper:'.length)
  }
  if (modelId.startsWith('streaming-asr:')) {
    return modelId.slice('streaming-asr:'.length)
  }
  if (modelId.startsWith('remote:')) {
    const rest = modelId.slice('remote:'.length)
    const slash = rest.indexOf('/')
    return slash >= 0 ? rest.slice(slash + 1) : rest
  }
  return modelId
}

/** Drop `-00001-of-00003` or `-1-of-3` so split GGUFs read as one model name. */
export function stripGgufShardSuffix(filename: string): string {
  return filename.replace(/-\d{1,5}-of-\d{1,5}(?=\.gguf$|$)/i, '')
}

export function modelDisplayName(
  modelId: string,
  model?: LocalModel
): { title: string; subtitle: string } {
  if (!modelId) {
    return { title: 'Select a model', subtitle: 'Download a model to get started' }
  }

  const engine = modelEngine(model)
  const engineText = engine ? engineLabel(engine) : ''

  if (modelId.startsWith('gguf-ext:') || modelId.startsWith('gguf:')) {
    const key = modelLibraryKey(modelId)
    const file = key.split('/').at(-1) ?? modelId
    const source = model?.library_label
      ? `${model.library_label} · External`
      : modelId.startsWith('gguf-ext:')
        ? 'External library'
        : 'Local library'
    return {
      title: stripGgufShardSuffix(file),
      subtitle: engineText ? `${engineText} · ${source}` : source
    }
  }

  if (
    modelId.startsWith('mlx-ext:') ||
    modelId.startsWith('mlx-vlm-ext:') ||
    modelId.startsWith('mlx:') ||
    modelId.startsWith('mlx-vlm:')
  ) {
    const key = modelLibraryKey(modelId)
    const name = key.split('/').at(-1) ?? modelId
    const source = model?.library_label
      ? `${model.library_label} · External`
      : modelId.includes('-ext:')
        ? 'External library'
        : 'Local library'
    return {
      title: name,
      subtitle: engineText ? `${engineText} · ${source}` : source
    }
  }

  if (modelId.startsWith('sdcpp-image:') || modelId.startsWith('sdcpp-video:')) {
    const key = modelLibraryKey(modelId)
    const name = key.split('/').at(-1) || key
    const modality = modelId.startsWith('sdcpp-video:') ? 'Video' : 'Image'
    const label = engineText || 'Gen · sd.cpp'
    return {
      title: stripGgufShardSuffix(name),
      subtitle: `${label} · ${modality}`
    }
  }

  if (modelId.startsWith('personaplex:')) {
    const key = modelLibraryKey(modelId)
    const name = key.split('/').at(-1) || key
    const label = engineText || 'Voice · PersonaPlex'
    return {
      title: name,
      subtitle: `${label} · Local library`
    }
  }

  if (modelId.startsWith('whisper:')) {
    const key = modelLibraryKey(modelId)
    const name = key.split('/').at(-1) || key
    const label = engineText || 'ASR · whisper'
    return {
      title: stripGgufShardSuffix(name),
      subtitle: `${label} · Local library`
    }
  }

  if (modelId.startsWith('streaming-asr:')) {
    const key = modelLibraryKey(modelId)
    const name = key.split('/').at(-1) || key
    const label = engineText || 'ASR · streaming'
    return {
      title: name,
      subtitle: `${label} · Local library`
    }
  }

  if (modelId.startsWith('remote:')) {
    const rest = modelId.slice('remote:'.length)
    const slash = rest.indexOf('/')
    const connectionName = slash >= 0 ? rest.slice(0, slash) : ''
    const rawModel = slash >= 0 ? rest.slice(slash + 1) : rest
    const title = rawModel.split('/').at(-1) || rawModel || rest
    const hostLabel = model?.library_label?.trim() || connectionName
    return {
      title,
      subtitle: hostLabel ? `Remote · ${hostLabel}` : 'Remote'
    }
  }

  return {
    title: stripGgufShardSuffix(modelId),
    subtitle: engineText || model?.owned_by.replace(/^brazier:/, '') || 'Local'
  }
}

/**
 * Derive the daemon's `data_dir` from `EngineStatus.managed_binary_path`
 * (`<data_dir>/engines/llama.cpp/bin/llama-server[.exe]`).
 */
export function dataDirFromManagedBinaryPath(
  managedBinaryPath: string | null | undefined
): string | null {
  if (!managedBinaryPath) return null
  const normalized = managedBinaryPath.replace(/\\/g, '/')
  const marker = '/engines/llama.cpp/'
  const index = normalized.lastIndexOf(marker)
  if (index <= 0) return null
  const useBackslash = managedBinaryPath.includes('\\') && !managedBinaryPath.includes('/')
  const prefix = normalized.slice(0, index)
  return useBackslash ? prefix.replace(/\//g, '\\') : prefix
}

/**
 * Resolve an installed local model ID to its on-disk file or folder path,
 * matching `crates/brazier-runtime/src/models_store.rs::path_for_model_id`.
 */
export function resolveLocalModelPath(
  modelId: string,
  dataDir: string | null | undefined,
  extraLibraryPaths: string[] = []
): string | null {
  if (!modelId || modelId.startsWith('remote:')) return null

  const joinParts = (root: string, relative: string): string => {
    const sep = root.includes('\\') && !root.includes('/') ? '\\' : '/'
    const cleanRoot = root.replace(/[\\/]+$/, '')
    const cleanRel = relative
      .split('/')
      .filter((part) => part.length > 0 && part !== '.' && part !== '..')
      .join(sep)
    return cleanRel ? `${cleanRoot}${sep}${cleanRel}` : cleanRoot
  }

  const externalMatch = /^(?:gguf-ext|mlx-ext|mlx-vlm-ext):(\d+):(.+)$/.exec(modelId)
  if (externalMatch) {
    const root = extraLibraryPaths[Number(externalMatch[1])]
    if (!root) return null
    return joinParts(root, externalMatch[2])
  }

  if (!dataDir) return null
  if (modelId.startsWith('gguf:')) {
    return joinParts(dataDir, `models/gguf/${modelId.slice('gguf:'.length)}`)
  }
  if (modelId.startsWith('mlx:') || modelId.startsWith('mlx-vlm:')) {
    return joinParts(dataDir, `models/mlx/${modelLibraryKey(modelId)}`)
  }
  if (modelId.startsWith('sdcpp-image:')) {
    return joinParts(dataDir, `models/sdcpp/image/${modelId.slice('sdcpp-image:'.length)}`)
  }
  if (modelId.startsWith('sdcpp-video:')) {
    return joinParts(dataDir, `models/sdcpp/video/${modelId.slice('sdcpp-video:'.length)}`)
  }
  if (modelId.startsWith('personaplex:')) {
    return joinParts(dataDir, `models/personaplex/${modelId.slice('personaplex:'.length)}`)
  }
  if (modelId.startsWith('whisper:')) {
    return joinParts(dataDir, `models/whisper/${modelId.slice('whisper:'.length)}`)
  }
  if (modelId.startsWith('streaming-asr:')) {
    return joinParts(dataDir, `models/streaming-asr/${modelId.slice('streaming-asr:'.length)}`)
  }
  return null
}

export function runtimeNoticeForModel(
  modelId: string,
  models: LocalModel[],
  runtimes: RuntimeEntry[] | null | undefined,
  bindings?: Record<string, string> | null
): string | null {
  if (!modelId || !runtimes) return null
  const model = models.find((entry) => entry.id === modelId)
  if (!model) return null
  const engine = modelEngine(model)
  if (!engine) return null
  const boundId = bindings?.[modelId]
  if (boundId) {
    const bound = runtimes.find((entry) => entry.id === boundId)
    if (bound) return null
    const fallback = fallbackRuntimeForEngine(runtimes, engine)
    if (fallback) {
      return `The paired runtime was removed. This model will use ${fallback.label}.`
    }
    return 'The paired runtime is missing. Choose another runtime below.'
  }
  const active = runtimes.some((entry) => entry.engine === engine && entry.active)
  if (active) return null
  if (engine === 'llama.cpp') {
    return 'This model needs an active llama.cpp runtime. Open Manage → Runtimes to install one.'
  }
  if (engine === 'mlx-lm' || engine === 'mlx-vlm') {
    return `This model needs ${engineLabel(engine)}. Open Manage → Runtimes to build and activate it.`
  }
  return null
}

/** Active runtime of this engine, else a managed install, else any remaining one. */
export function fallbackRuntimeForEngine(
  runtimes: RuntimeEntry[],
  engine: string
): RuntimeEntry | undefined {
  return (
    runtimes.find((entry) => entry.engine === engine && entry.active) ??
    runtimes.find((entry) => entry.engine === engine && entry.kind === 'managed') ??
    runtimes.find((entry) => entry.engine === engine)
  )
}

export function runtimesForModel(
  model: LocalModel | undefined,
  runtimes: RuntimeEntry[] | null | undefined
): RuntimeEntry[] {
  const engine = modelEngine(model)
  if (!engine || !runtimes) return []
  return runtimes.filter((entry) => entry.engine === engine)
}

export function visionCapabilityTitle(
  modelId: string,
  models: LocalModel[],
  canAttach: boolean
): string {
  const engine = modelEngine(models.find((model) => model.id === modelId))
  if (engine === 'mlx-vlm') return 'Vision model (MLX)'
  if (engine === 'mlx-lm') return 'Text-only MLX model'
  if (engine === 'remote') {
    return canAttach
      ? 'Vision enabled for this remote model'
      : 'Vision not advertised — toggle Vision in model settings if supported'
  }
  if (canAttach) return 'Vision-capable (projector installed)'
  return 'Needs a vision/mmproj GGUF pair for image input'
}
