import { describe, expect, it } from 'vitest'

import {
  dataDirFromManagedBinaryPath,
  fallbackRuntimeForEngine,
  isComputerUseModel,
  mergeModelCapabilities,
  modelDisplayName,
  modelLibraryKey,
  resolveLocalModelPath,
  runtimeNoticeForModel,
  sortComputerUseModels,
  stripGgufShardSuffix
} from './model-utils'
import type { LocalModel, ModelProfile, RuntimeEntry } from './api'

function model(
  id: string,
  engine: string,
  extra: Partial<LocalModel> = {}
): LocalModel {
  return {
    id,
    object: 'model',
    owned_by: `brazier:${engine}`,
    engine,
    ...extra
  }
}

function runtime(partial: Partial<RuntimeEntry> & Pick<RuntimeEntry, 'id' | 'engine'>): RuntimeEntry {
  return {
    kind: 'managed',
    label: partial.id,
    target: 'cpu',
    version: 'b1',
    path: `/tmp/${partial.id}`,
    active: false,
    deletable: true,
    ...partial
  }
}

describe('runtimeNoticeForModel', () => {
  const models = [model('gguf:acme/a.gguf', 'llama.cpp')]

  it('explains a missing pairing when another runtime can take over', () => {
    const runtimes = [runtime({ id: 'managed', engine: 'llama.cpp', active: true, label: 'llama.cpp · CPU' })]
    expect(
      runtimeNoticeForModel('gguf:acme/a.gguf', models, runtimes, {
        'gguf:acme/a.gguf': 'source-gone'
      })
    ).toBe('The paired runtime was removed. This model will use llama.cpp · CPU.')
  })

  it('asks the user to choose when nothing remains for that engine', () => {
    expect(
      runtimeNoticeForModel('gguf:acme/a.gguf', models, [], {
        'gguf:acme/a.gguf': 'source-gone'
      })
    ).toBe('The paired runtime is missing. Choose another runtime below.')
  })
})

describe('fallbackRuntimeForEngine', () => {
  it('prefers the active runtime, then a managed install', () => {
    const runtimes = [
      runtime({ id: 'source-old', engine: 'llama.cpp', kind: 'source', active: false }),
      runtime({ id: 'managed', engine: 'llama.cpp', kind: 'managed', active: true })
    ]
    expect(fallbackRuntimeForEngine(runtimes, 'llama.cpp')?.id).toBe('managed')
  })
})

describe('stripGgufShardSuffix', () => {
  it('strips both 5-digit and shorter shard suffixes', () => {
    expect(stripGgufShardSuffix('Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf')).toBe(
      'Qwen3-30B-A3B-Q4_K_M.gguf'
    )
    expect(stripGgufShardSuffix('Model-Q8_0-1-of-2.gguf')).toBe('Model-Q8_0.gguf')
    expect(stripGgufShardSuffix('Model-Q8_0-1-of-2')).toBe('Model-Q8_0')
  })
})

describe('modelLibraryKey and modelDisplayName', () => {
  it('strips internal scheme prefixes cleanly for diffusion, voice, ASR, and remote models', () => {
    expect(modelLibraryKey('sdcpp-image:black-forest-labs/flux-schnell')).toBe(
      'black-forest-labs/flux-schnell'
    )
    expect(modelDisplayName('sdcpp-image:black-forest-labs/flux-schnell')).toEqual({
      title: 'flux-schnell',
      subtitle: 'Gen · sd.cpp · Image'
    })

    expect(modelDisplayName('sdcpp-video:wan-ai/wan2.1-t2v-1.3b')).toEqual({
      title: 'wan2.1-t2v-1.3b',
      subtitle: 'Gen · sd.cpp · Video'
    })

    expect(modelDisplayName('personaplex:nvidia/personaplex-7b-v1')).toEqual({
      title: 'personaplex-7b-v1',
      subtitle: 'Voice · PersonaPlex · Local library'
    })

    expect(modelDisplayName('whisper:ggerganov/ggml-large-v3-turbo.bin')).toEqual({
      title: 'ggml-large-v3-turbo.bin',
      subtitle: 'ASR · whisper · Local library'
    })

    expect(modelDisplayName('streaming-asr:nvidia/nemotron-speech-streaming-en-0.6b')).toEqual({
      title: 'nemotron-speech-streaming-en-0.6b',
      subtitle: 'ASR · streaming · Local library'
    })
  })

  it('uses library_label or connection id for remote models without leaking remote: prefix', () => {
    expect(modelLibraryKey('remote:workstation/Qwen/Qwen3-VL-8B-Instruct')).toBe(
      'Qwen/Qwen3-VL-8B-Instruct'
    )
    expect(
      modelDisplayName(
        'remote:workstation/Qwen/Qwen3-VL-8B-Instruct',
        model('remote:workstation/Qwen/Qwen3-VL-8B-Instruct', 'remote', {
          library_label: 'Workstation (vLLM)'
        })
      )
    ).toEqual({
      title: 'Qwen3-VL-8B-Instruct',
      subtitle: 'Remote · Workstation (vLLM)'
    })

    expect(modelDisplayName('remote:ollama/gemma4:12b')).toEqual({
      title: 'gemma4:12b',
      subtitle: 'Remote · ollama'
    })
  })
})

describe('isComputerUseModel and sortComputerUseModels', () => {
  const fara = model('gguf:microsoft/Fara-7B-Q4_K_M.gguf', 'llama.cpp', {
    capabilities: {
      input_modalities: ['text', 'image'],
      output_modalities: ['text'],
      streaming: true,
      tools: false,
      reasoning: false,
      computer_use: true
    }
  })
  const qwenVl = model('mlx-vlm:mlx-community/Qwen3-VL-8B-4bit', 'mlx-vlm', {
    capabilities: {
      input_modalities: ['text', 'image'],
      output_modalities: ['text'],
      streaming: true,
      tools: true,
      reasoning: true,
      computer_use: false
    }
  })
  const textOnly = model('gguf:acme/Qwen3-8B-Q4_K_M.gguf', 'llama.cpp', {
    capabilities: {
      input_modalities: ['text'],
      output_modalities: ['text'],
      streaming: true,
      tools: true,
      reasoning: true,
      computer_use: false
    }
  })
  const diffusion = model('sdcpp-image:acme/flux', 'stable-diffusion.cpp', {
    capabilities: {
      input_modalities: ['text', 'image'],
      output_modalities: ['image'],
      streaming: false,
      tools: false,
      reasoning: false
    }
  })

  it('accepts both dedicated computer_use models and general vision chat models', () => {
    expect(isComputerUseModel(fara)).toBe(true)
    expect(isComputerUseModel(qwenVl)).toBe(true)
    expect(isComputerUseModel(textOnly)).toBe(false)
    expect(isComputerUseModel(diffusion)).toBe(false)
  })

  it('sorts dedicated computer_use models ahead of general vision models', () => {
    const sorted = sortComputerUseModels([qwenVl, fara])
    expect(sorted.map((entry) => entry.id)).toEqual([fara.id, qwenVl.id])
  })

  it('applies per-model TextProfile capability overrides', () => {
    const remoteModel = model('remote:ollama/qwen3-vl:8b', 'remote', {
      capabilities: {
        input_modalities: ['text'],
        output_modalities: ['text'],
        streaming: true,
        tools: false,
        reasoning: false,
        computer_use: false
      }
    })
    const profile: ModelProfile = {
      kind: 'text',
      capability_vision: true,
      capability_reasoning: true,
      capability_tools: true
    }
    const merged = mergeModelCapabilities(remoteModel, profile)
    expect(merged?.input_modalities).toContain('image')
    expect(merged?.reasoning).toBe(true)
    expect(merged?.reasoning_modes).toEqual(['off', 'on'])
    expect(merged?.tools).toBe(true)
    expect(isComputerUseModel(remoteModel, profile)).toBe(true)
  })
})

describe('resolveLocalModelPath', () => {
  it('derives data_dir and resolves local and external model paths', () => {
    const dataDir = dataDirFromManagedBinaryPath(
      '/Users/alice/Library/Application Support/brazier/engines/llama.cpp/bin/llama-server'
    )
    expect(dataDir).toBe('/Users/alice/Library/Application Support/brazier')
    expect(resolveLocalModelPath('gguf:acme/model.gguf', dataDir, [])).toBe(
      '/Users/alice/Library/Application Support/brazier/models/gguf/acme/model.gguf'
    )
    expect(resolveLocalModelPath('gguf-ext:0:sub/ext.gguf', dataDir, ['/Volumes/Models'])).toBe(
      '/Volumes/Models/sub/ext.gguf'
    )
    expect(resolveLocalModelPath('remote:work/qwen', dataDir, [])).toBeNull()
  })
})
