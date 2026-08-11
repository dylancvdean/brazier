import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'

import {
  VOICE_QUALIFICATION_PHRASES,
  buildVoiceQualificationResult,
  countRecognizedQualificationPhrases
} from './voiceQualification'

describe('voice hardware qualification evidence', () => {
  it('counts distinct fixed phrases rather than treating any capture as recall', () => {
    expect(countRecognizedQualificationPhrases([...VOICE_QUALIFICATION_PHRASES])).toBe(20)
    expect(
      countRecognizedQualificationPhrases([
        ...VOICE_QUALIFICATION_PHRASES.slice(0, 19),
        'unrelated background words that happened to transcribe'
      ])
    ).toBe(19)
    expect(
      countRecognizedQualificationPhrases([
        ...VOICE_QUALIFICATION_PHRASES.slice(0, 7),
        ...VOICE_QUALIFICATION_PHRASES.slice(8)
      ])
    ).toBe(19)
    expect(
      countRecognizedQualificationPhrases([
        ...VOICE_QUALIFICATION_PHRASES.slice(0, 10),
        'unrelated stray segment',
        ...VOICE_QUALIFICATION_PHRASES.slice(10)
      ])
    ).toBe(20)
    expect(countRecognizedQualificationPhrases(Array(20).fill('keyboard noise'))).toBe(0)
  })

  it('passes only a complete in-budget hardware trial', async () => {
    const result = await buildVoiceQualificationResult({
      host: {
        commit: 'abcdef012345', platform: 'macos', arch: 'arm64',
        memory_gib: 32, gpu_vram_gib: null, gpu_vendor: 'apple'
      },
      microphoneClass: 'built-in',
      expectedSpeechUtterances: VOICE_QUALIFICATION_PHRASES.length,
      recognizedSpeechUtterances: 19,
      noiseMinutes: 10,
      falseNoiseUtterances: 1,
      captureVad: 'silero-v5',
      models: { voice: 'voice', background: 'chat' },
      vadWindowP95Ms: 4,
      vadQueueLagP95Ms: 0,
      vadSamples: 100,
      metrics: {
        transcriptWaitMs: Array(19).fill(500),
        interruptToSpeechStopMs: [100, 120, 140]
      },
      createdAt: '2026-08-10T00:00:00.000Z'
    })

    expect(result.passed).toBe(true)
    expect(result.host_id).toBe('macos-apple-silicon')
    expect(result.metrics.speech_recall).toBe(0.95)
    expect(result.corpus_sha256).toMatch(/^[a-f0-9]{64}$/)
    const manifest = JSON.parse(
      readFileSync(
        new URL('../../../../../../qualification/beta-manifest.json', import.meta.url),
        'utf8'
      )
    ) as { voice_corpus_sha256: string }
    expect(result.corpus_sha256).toBe(manifest.voice_corpus_sha256)
  })

  it('fails empty or fallback measurements instead of treating missing percentiles as zero', async () => {
    const result = await buildVoiceQualificationResult({
      host: {
        commit: 'abcdef012345', platform: 'linux', arch: 'x64',
        memory_gib: 32, gpu_vram_gib: 24, gpu_vendor: 'nvidia'
      },
      microphoneClass: 'usb',
      expectedSpeechUtterances: VOICE_QUALIFICATION_PHRASES.length,
      recognizedSpeechUtterances: 20,
      noiseMinutes: 10,
      falseNoiseUtterances: 0,
      captureVad: 'energy-fallback',
      models: { voice: 'voice', background: 'chat' },
      vadWindowP95Ms: Number.POSITIVE_INFINITY,
      vadQueueLagP95Ms: Number.POSITIVE_INFINITY,
      vadSamples: 0,
      metrics: {
        transcriptWaitMs: [],
        interruptToSpeechStopMs: []
      }
    })

    expect(result.passed).toBe(false)
    expect(result.metrics.vad_window_p95_ms).toBe(Number.POSITIVE_INFINITY)
  })
})
