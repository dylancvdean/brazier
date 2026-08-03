import type { LocalModel, ModelKind, RuntimeEntry } from './api'

export function modelEngine(model: LocalModel | undefined): string {
  if (!model) return ''
  if (model.engine) return model.engine
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
    engine !== 'vllm-omni' &&
    engine !== 'personaplex' &&
    !model?.id.startsWith('sdcpp-image:') &&
    !model?.id.startsWith('sdcpp-video:') &&
    !model?.id.startsWith('vllm-omni-image:') &&
    !model?.id.startsWith('vllm-omni-video:') &&
    !model?.id.startsWith('personaplex:')
  )
}

export function isImageGenModel(model: LocalModel | undefined): boolean {
  return Boolean(
    model?.id.startsWith('sdcpp-image:') || model?.id.startsWith('vllm-omni-image:')
  )
}

export function isVideoGenModel(model: LocalModel | undefined): boolean {
  return Boolean(
    model?.id.startsWith('sdcpp-video:') || model?.id.startsWith('vllm-omni-video:')
  )
}

export function isVoiceModel(model: LocalModel | undefined): boolean {
  return Boolean(model?.id.startsWith('personaplex:'))
}

/**
 * Chat-class models that can drive Computer Use (screenshot → action).
 *
 * True when the daemon advertises `capabilities.computer_use`, or when the
 * model accepts image input and its id/name looks like Fara / computer-use.
 */
export function isComputerUseModel(model: LocalModel | undefined): boolean {
  if (!model || !isChatModel(model)) return false
  if (model.capabilities?.computer_use) return true
  const hasImage = Boolean(model.capabilities?.input_modalities?.includes('image'))
  if (!hasImage) return false
  const needle = `${model.id} ${model.library_label ?? ''}`.toLowerCase()
  return (
    needle.includes('fara') ||
    needle.includes('computer-use') ||
    needle.includes('computer_use') ||
    needle.includes('cua-') ||
    needle.includes('-cua')
  )
}

/**
 * Which family of advanced settings a model takes.
 *
 * Mirrors the daemon's own classification, which is the one that decides what
 * a stored profile is allowed to contain.
 */
export function modelKindFor(modelId: string): ModelKind {
  if (modelId.startsWith('sdcpp-image:') || modelId.startsWith('vllm-omni-image:')) return 'image'
  if (modelId.startsWith('sdcpp-video:') || modelId.startsWith('vllm-omni-video:')) return 'video'
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
  return modelId
}

/** Drop `-00001-of-00003` so split GGUFs read as one model name. */
export function stripGgufShardSuffix(filename: string): string {
  return filename.replace(/-\d{5}-of-\d{5}(?=\.gguf$)/i, '')
}

export function modelDisplayName(
  modelId: string,
  model?: LocalModel
): { title: string; subtitle: string } {
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

  if (!modelId) {
    return { title: 'Select a model', subtitle: 'Download a model to get started' }
  }

  return {
    title: modelId,
    subtitle: engineText || model?.owned_by.replace(/^brazier:/, '') || 'Local'
  }
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
  if (canAttach) return 'Vision-capable (projector installed)'
  return 'Needs a vision/mmproj GGUF pair for image input'
}
