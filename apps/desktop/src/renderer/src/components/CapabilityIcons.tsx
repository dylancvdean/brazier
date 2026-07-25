import { Brain, Image, Video, Wrench } from 'lucide-react'
import type { LocalModel } from '../api'

/** What a model can take in and put out, as shown in model lists. */
export type CapabilityFlags = {
  imageIn?: boolean
  videoIn?: boolean
  reasoning?: boolean
  tools?: boolean
  imageOut?: boolean
  videoOut?: boolean
}

/**
 * Capabilities of an installed model.
 *
 * Video input is not a model capability of its own: any vision model can take
 * video once ffmpeg is present to sample it into frames, which is the same
 * rule the composer uses to decide whether video can be attached.
 */
export function capabilityFlags(model: LocalModel | undefined, videoPipeline = false): CapabilityFlags {
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

/**
 * Best guess for a model that is not installed yet, from its Hub tags.
 *
 * Only the pipeline tag is reliable before download, so this deliberately
 * claims less than [`capabilityFlags`] and is labelled as a guess in the UI.
 */
export function hubCapabilityFlags(tags: string[]): CapabilityFlags {
  const lower = tags.map((tag) => tag.toLowerCase())
  const has = (...needles: string[]) => needles.some((needle) => lower.includes(needle))
  return {
    imageIn: has('image-text-to-text', 'visual-question-answering', 'image-to-text', 'vision'),
    imageOut: has('text-to-image', 'image-to-image'),
    videoOut: has('text-to-video', 'image-to-video'),
    // Reasoning and tool use are not expressed in Hub tags with any
    // consistency, so they are left out rather than guessed at.
    reasoning: false,
    tools: false
  }
}

const ENTRIES: Array<{
  key: keyof CapabilityFlags
  icon: React.JSX.Element
  label: string
  direction: 'in' | 'out'
}> = [
  { key: 'imageIn', icon: <Image size={12} />, label: 'Accepts images', direction: 'in' },
  {
    key: 'videoIn',
    icon: <Video size={12} />,
    label: 'Accepts video (sampled into frames)',
    direction: 'in'
  },
  { key: 'reasoning', icon: <Brain size={12} />, label: 'Reasoning', direction: 'in' },
  { key: 'tools', icon: <Wrench size={12} />, label: 'Tool calling', direction: 'in' },
  { key: 'imageOut', icon: <Image size={12} />, label: 'Generates images', direction: 'out' },
  { key: 'videoOut', icon: <Video size={12} />, label: 'Generates video', direction: 'out' }
]

/**
 * Compact capability row. Only supported capabilities are drawn, so a list
 * stays scannable rather than showing a grid of mostly-greyed icons.
 */
export function CapabilityIcons({
  flags,
  inferred = false
}: {
  flags: CapabilityFlags
  /** Marks guesses made from Hub tags before a model is installed. */
  inferred?: boolean
}): React.JSX.Element | null {
  const shown = ENTRIES.filter((entry) => flags[entry.key])
  if (shown.length === 0) return null
  return (
    <span className={`capability-icons ${inferred ? 'inferred' : ''}`}>
      {shown.map((entry) => (
        <span
          key={`${entry.key}`}
          className={`capability-icon ${entry.direction}`}
          title={inferred ? `${entry.label} (from Hugging Face tags)` : entry.label}
          aria-label={entry.label}
        >
          {entry.icon}
        </span>
      ))}
    </span>
  )
}
