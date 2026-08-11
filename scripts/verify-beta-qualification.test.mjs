import assert from 'node:assert/strict'
import test from 'node:test'

import { verifyQualification } from './verify-beta-qualification.mjs'

const commit = '0123456789abcdef'
const manifest = {
  schema_version: 1,
  required_package_smokes: [{ platform: 'macos', arch: 'arm64', artifact: 'dmg' }],
  required_voice_hosts: [
    {
      id: 'mac',
      platform: 'macos',
      arch: 'arm64',
      minimum_memory_gib: 16,
      microphone_classes: ['built-in']
    }
  ],
  package_budgets: { agent_worker_ready_ms_max: 10_000, agent_session_open_ms_max: 15_000 },
  voice_budgets: {
    speech_recall_min: 0.95,
    false_sustained_interruptions_per_noise_minute_max: 0.1,
    vad_window_p95_ms_max: 32
  },
  voice_minimums: {
    expected_speech_utterances: 20,
    noise_minutes: 5,
    vad_samples: 100,
    interruption_samples: 3
  },
  voice_corpus_sha256: 'a'.repeat(64)
}

function passingResults() {
  return [
    {
      file: 'package.json',
      schema_version: 1,
      commit,
      passed: true,
      kind: 'package-smoke',
      platform: 'macos',
      arch: 'arm64',
      artifact: 'dmg',
      checks: {
        daemon_started: true,
        safety_helper_present: true,
        worker_loaded: true,
        session_opened: true,
        worker_shutdown: true,
        session_deleted: true,
        daemon_stopped: true,
        clean_shutdown: true
      },
      metrics: { agent_worker_ready_ms: 100, agent_session_open_ms: 200 }
    },
    {
      file: 'voice.json',
      schema_version: 1,
      commit,
      passed: true,
      kind: 'voice-hardware',
      host_id: 'mac',
      platform: 'macos',
      arch: 'arm64',
      memory_gib: 32,
      gpu_vram_gib: null,
      gpu_vendor: 'apple',
      microphone_class: 'built-in',
      capture_vad: 'silero-v5',
      corpus_sha256: 'a'.repeat(64),
      samples: {
        expected_speech_utterances: 20,
        recognized_speech_utterances: 20,
        noise_minutes: 10,
        false_noise_utterances: 0,
        vad_samples: 100,
        transcript_samples: 20,
        interruption_samples: 3
      },
      metrics: {
        speech_recall: 1,
        false_sustained_interruptions_per_noise_minute: 0,
        vad_window_p95_ms: 12
      }
    }
  ]
}

test('accepts one fresh passing artifact for every required target', () => {
  assert.deepEqual(verifyQualification(manifest, passingResults(), commit), {
    commit,
    package_smokes: 1,
    voice_hosts: 1
  })
})

test('rejects stale and over-budget evidence', () => {
  const stale = passingResults()
  stale[0].commit = 'ffffffff'
  assert.throws(() => verifyQualification(manifest, stale, commit), /belongs to/)

  const slow = passingResults()
  slow[1].metrics.vad_window_p95_ms = 40
  assert.throws(() => verifyQualification(manifest, slow, commit), /exceeds/)
})

test('rejects missing target evidence', () => {
  assert.throws(() => verifyQualification(manifest, passingResults().slice(0, 1), commit), /voice result/)
})

test('requires the installed AppContainer launcher probe for Windows packages', () => {
  const windowsManifest = structuredClone(manifest)
  windowsManifest.required_package_smokes = [
    { platform: 'windows', arch: 'x64', artifact: 'nsis' }
  ]
  windowsManifest.required_voice_hosts = []
  const [result] = passingResults()
  result.platform = 'windows'
  result.arch = 'x64'
  result.artifact = 'nsis'

  assert.throws(
    () => verifyQualification(windowsManifest, [result], commit),
    /AppContainer launcher/
  )
  result.checks.windows_sandbox_probe = true
  assert.deepEqual(verifyQualification(windowsManifest, [result], commit), {
    commit,
    package_smokes: 1,
    voice_hosts: 0
  })
})

test('allows an experimental platform to be absent while requiring every other target', () => {
  const mixedManifest = structuredClone(manifest)
  mixedManifest.required_package_smokes.push({
    platform: 'windows',
    arch: 'x64',
    artifact: 'nsis'
  })

  assert.deepEqual(
    verifyQualification(mixedManifest, passingResults(), commit, new Set(['windows'])),
    {
      commit,
      package_smokes: 2,
      voice_hosts: 1
    }
  )
  assert.throws(
    () => verifyQualification(mixedManifest, passingResults(), commit),
    /windows-x64-nsis/
  )
})
