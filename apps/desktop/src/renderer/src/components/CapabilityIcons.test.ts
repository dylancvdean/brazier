import { describe, expect, it } from 'vitest'
import { capabilityFlags, hubCapabilityFlags } from './CapabilityIcons'
import type { LocalModel } from '../api'

function model(capabilities: Partial<NonNullable<LocalModel['capabilities']>>): LocalModel {
  return {
    id: 'test',
    object: 'model',
    owned_by: 'brazier',
    capabilities: {
      input_modalities: ['text'],
      output_modalities: ['text'],
      streaming: true,
      tools: false,
      reasoning: false,
      ...capabilities
    }
  }
}

describe('installed model capabilities', () => {
  it('reports what a vision chat model accepts', () => {
    const flags = capabilityFlags(
      model({ input_modalities: ['text', 'image'], tools: true, reasoning: true }),
      true
    )
    expect(flags).toMatchObject({ imageIn: true, videoIn: true, tools: true, reasoning: true })
    expect(flags.imageOut).toBe(false)
  })

  it('does not claim video input without the ffmpeg pipeline', () => {
    const flags = capabilityFlags(model({ input_modalities: ['text', 'image'] }), false)
    expect(flags.imageIn).toBe(true)
    expect(flags.videoIn).toBe(false)
  })

  it('never claims video input for a text-only model', () => {
    expect(capabilityFlags(model({}), true).videoIn).toBe(false)
  })

  it('counts budget-style reasoning modes as reasoning', () => {
    expect(capabilityFlags(model({ reasoning_modes: ['off', 'budget'] })).reasoning).toBe(true)
  })

  it('separates image and video generators by output', () => {
    expect(capabilityFlags(model({ output_modalities: ['image'] }))).toMatchObject({
      imageOut: true,
      videoOut: false
    })
    expect(capabilityFlags(model({ output_modalities: ['video'] }))).toMatchObject({
      imageOut: false,
      videoOut: true
    })
  })

  it('shows nothing for a model whose capabilities are unknown', () => {
    expect(capabilityFlags(undefined)).toEqual({})
  })
})

describe('capabilities guessed from Hub tags', () => {
  it('recognises generation pipelines', () => {
    expect(hubCapabilityFlags(['text-to-image', 'diffusers']).imageOut).toBe(true)
    expect(hubCapabilityFlags(['text-to-video']).videoOut).toBe(true)
  })

  it('recognises vision inputs', () => {
    expect(hubCapabilityFlags(['image-text-to-text']).imageIn).toBe(true)
  })

  it('does not guess at reasoning or tool support', () => {
    const flags = hubCapabilityFlags(['text-generation', 'conversational'])
    expect(flags.reasoning).toBe(false)
    expect(flags.tools).toBe(false)
    expect(flags.imageIn).toBe(false)
  })
})
