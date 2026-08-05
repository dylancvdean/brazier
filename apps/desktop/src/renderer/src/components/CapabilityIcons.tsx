import { Brain, Image, Music, Video, Wrench } from 'lucide-react'
import type { LocalModel } from '../api'

/** What a model can take in and put out, as shown in model lists. */
export type CapabilityFlags = {
  imageIn?: boolean
  videoIn?: boolean
  reasoning?: boolean
  tools?: boolean
  imageOut?: boolean
  videoOut?: boolean
  /** Model also generates synchronized audio (e.g. MiniMax-H3). */
  audioOut?: boolean
}

/** Which set of icons a model is judged by. */
export type CapabilityKind = 'chat' | 'generator'

/**
 * Capabilities of an installed model.
 *
 * Video input is not a model capability of its own: any vision model can take
 * video once ffmpeg is present to sample it into frames, which is the same
 * rule the composer uses to decide whether video can be attached.
 */
export function capabilityFlags(
  model: LocalModel | undefined,
  videoPipeline = false
): CapabilityFlags {
  const caps = model?.capabilities
  if (!caps) return {}
  const imageIn = caps.input_modalities.includes('image')
  const outputs = caps.output_modalities
  return {
    imageIn,
    videoIn: imageIn && videoPipeline,
    reasoning: Boolean(caps.reasoning) || (caps.reasoning_modes?.length ?? 0) > 0,
    tools: caps.tools,
    imageOut: outputs.includes('image'),
    videoOut: outputs.includes('video')
  }
}

/** Whether a model generates media or consumes it. */
export function capabilityKind(flags: CapabilityFlags): CapabilityKind {
  return flags.imageOut || flags.videoOut ? 'generator' : 'chat'
}

/**
 * Best guess for a model that is not installed yet, from its Hub pipeline tag.
 *
 * Only the pipeline tag is dependable before download. Reasoning and tool use
 * are not expressed in tags with any consistency, so they stay unlit rather
 * than being guessed at — the UI marks the whole row as inferred.
 */
export function hubCapabilityFlags(tags: string[]): CapabilityFlags {
  const lower = tags.map((tag) => tag.toLowerCase())
  const has = (...needles: string[]) => needles.some((needle) => lower.includes(needle))
  return {
    imageIn: has('image-text-to-text', 'visual-question-answering', 'image-to-text', 'vision'),
    imageOut: has('text-to-image', 'image-to-image'),
    videoOut: has('text-to-video', 'image-to-video'),
    reasoning: false,
    tools: false
  }
}

type Entry = {
  key: keyof CapabilityFlags
  icon: React.JSX.Element
  label: string
  direction: 'in' | 'out'
}

const CHAT_ENTRIES: Entry[] = [
  { key: 'imageIn', icon: <Image size={12} />, label: 'images in', direction: 'in' },
  { key: 'videoIn', icon: <Video size={12} />, label: 'video in', direction: 'in' },
  { key: 'reasoning', icon: <Brain size={12} />, label: 'reasoning', direction: 'in' },
  { key: 'tools', icon: <Wrench size={12} />, label: 'tool calling', direction: 'in' }
]

const GENERATOR_ENTRIES: Entry[] = [
  { key: 'imageOut', icon: <Image size={12} />, label: 'images out', direction: 'out' },
  { key: 'videoOut', icon: <Video size={12} />, label: 'video out', direction: 'out' },
  { key: 'audioOut', icon: <Music size={12} />, label: 'audio out', direction: 'out' },
  { key: 'imageIn', icon: <Image size={12} />, label: 'accepts a starting image', direction: 'in' }
]

/**
 * Compact capability row, always laid out horizontally.
 *
 * The whole set is drawn so lists line up and a model's shape is readable at a
 * glance; unsupported capabilities are dimmed rather than omitted. For a model
 * that is not installed yet, dim means "not indicated by its Hub tags", which
 * the tooltip spells out.
 */
export function CapabilityIcons({
  flags,
  kind,
  inferred = false
}: {
  flags: CapabilityFlags
  /** Defaults to whichever set suits the flags. */
  kind?: CapabilityKind
  /** Marks guesses made from Hub tags before a model is installed. */
  inferred?: boolean
}): React.JSX.Element | null {
  const entries = (kind ?? capabilityKind(flags)) === 'generator' ? GENERATOR_ENTRIES : CHAT_ENTRIES
  if (entries.length === 0) return null
  return (
    <span className={`capability-icons ${inferred ? 'inferred' : ''}`}>
      {entries.map((entry) => {
        const on = Boolean(flags[entry.key])
        const title = on
          ? `Supports ${entry.label}`
          : inferred
            ? `No ${entry.label} indicated by its Hugging Face tags`
            : `No ${entry.label}`
        return (
          <span
            key={entry.key}
            className={`capability-icon ${entry.direction} ${on ? 'on' : 'off'}`}
            title={title}
            aria-label={title}
          >
            {entry.icon}
          </span>
        )
      })}
    </span>
  )
}
