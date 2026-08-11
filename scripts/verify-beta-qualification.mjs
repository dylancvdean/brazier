import { readFile, readdir } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'

const root = path.resolve(import.meta.dirname, '..')

function argument(name) {
  const index = process.argv.indexOf(name)
  if (index < 0 || index + 1 >= process.argv.length) return null
  return process.argv[index + 1]
}

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function finiteNumber(result, field) {
  const value = result.metrics?.[field]
  assert(Number.isFinite(value), `${result.file}: metrics.${field} must be a finite number`)
  return value
}

async function loadJson(file) {
  const value = JSON.parse(await readFile(file, 'utf8'))
  value.file = path.basename(file)
  return value
}

export function verifyQualification(manifest, results, commit) {
  assert(typeof commit === 'string' && commit.length >= 7, 'a release commit is required')
  for (const result of results) {
    assert(result.schema_version === manifest.schema_version, `${result.file}: schema mismatch`)
    assert(result.commit === commit, `${result.file}: evidence belongs to ${result.commit}, not ${commit}`)
    assert(result.passed === true, `${result.file}: qualification did not pass`)
  }

  for (const required of manifest.required_package_smokes) {
    const matches = results.filter(
      (result) =>
        result.kind === 'package-smoke' &&
        result.platform === required.platform &&
        result.arch === required.arch &&
        result.artifact === required.artifact
    )
    const label = `${required.platform}-${required.arch}-${required.artifact}`
    assert(matches.length === 1, `expected exactly one package smoke for ${label}, found ${matches.length}`)
    const result = matches[0]
    for (const check of [
      'daemon_started',
      'safety_helper_present',
      'worker_loaded',
      'session_opened',
      'worker_shutdown',
      'session_deleted',
      'daemon_stopped',
      'clean_shutdown'
    ]) {
      assert(result.checks?.[check] === true, `${result.file}: ${check} was not proved`)
    }
    if (required.platform === 'windows') {
      assert(
        result.checks?.windows_sandbox_probe === true,
        `${result.file}: the packaged Windows AppContainer launcher was not proved`
      )
    }
    assert(
      finiteNumber(result, 'agent_worker_ready_ms') <=
        manifest.package_budgets.agent_worker_ready_ms_max,
      `${result.file}: agent worker startup exceeded its budget`
    )
    assert(
      finiteNumber(result, 'agent_session_open_ms') <=
        manifest.package_budgets.agent_session_open_ms_max,
      `${result.file}: agent session startup exceeded its budget`
    )
  }

  for (const required of manifest.required_voice_hosts) {
    const matches = results.filter(
      (result) => result.kind === 'voice-hardware' && result.host_id === required.id
    )
    assert(matches.length === 1, `expected exactly one voice result for ${required.id}, found ${matches.length}`)
    const result = matches[0]
    assert(result.platform === required.platform, `${result.file}: platform mismatch`)
    assert(result.arch === required.arch, `${result.file}: architecture mismatch`)
    assert(result.capture_vad === 'silero-v5', `${result.file}: Silero VAD was not active`)
    assert(
      finiteNumber({ ...result, metrics: result }, 'memory_gib') >= required.minimum_memory_gib,
      `${result.file}: host memory is below ${required.minimum_memory_gib} GiB`
    )
    if (required.minimum_vram_gib !== undefined) {
      assert(
        finiteNumber({ ...result, metrics: result }, 'gpu_vram_gib') >= required.minimum_vram_gib,
        `${result.file}: GPU VRAM is below ${required.minimum_vram_gib} GiB`
      )
    }
    if (required.required_gpu_vendor !== undefined) {
      assert(
        result.gpu_vendor === required.required_gpu_vendor,
        `${result.file}: GPU vendor is not ${required.required_gpu_vendor}`
      )
    }
    assert(
      required.microphone_classes.includes(result.microphone_class),
      `${result.file}: unsupported microphone class ${result.microphone_class}`
    )
    assert(
      result.corpus_sha256 === manifest.voice_corpus_sha256,
      `${result.file}: corpus_sha256 does not match the fixed speech corpus`
    )
    for (const [field, minimum] of Object.entries(manifest.voice_minimums)) {
      assert(
        finiteNumber({ ...result, metrics: result.samples }, field) >= minimum,
        `${result.file}: samples.${field} is below ${minimum}`
      )
    }
    const expected = finiteNumber({ ...result, metrics: result.samples }, 'expected_speech_utterances')
    const recognized = finiteNumber({ ...result, metrics: result.samples }, 'recognized_speech_utterances')
    const noiseMinutes = finiteNumber({ ...result, metrics: result.samples }, 'noise_minutes')
    const falseNoise = finiteNumber({ ...result, metrics: result.samples }, 'false_noise_utterances')
    assert(recognized <= expected, `${result.file}: recognized utterances exceed the fixed corpus`)
    assert(
      result.samples.transcript_samples >= recognized,
      `${result.file}: recognized utterances exceed measured transcript samples`
    )
    assert(
      Math.abs(result.metrics.speech_recall - recognized / expected) < 1e-9,
      `${result.file}: speech recall does not match its sample counts`
    )
    assert(
      Math.abs(
        result.metrics.false_sustained_interruptions_per_noise_minute - falseNoise / noiseMinutes
      ) < 1e-9,
      `${result.file}: false interruption rate does not match its sample counts`
    )
    for (const [field, budget] of Object.entries(manifest.voice_budgets)) {
      const metric = field.replace(/_(min|max)$/, '')
      const value = finiteNumber(result, metric)
      if (field.endsWith('_min')) assert(value >= budget, `${result.file}: ${metric} is below ${budget}`)
      else assert(value <= budget, `${result.file}: ${metric} exceeds ${budget}`)
    }
  }

  return {
    commit,
    package_smokes: manifest.required_package_smokes.length,
    voice_hosts: manifest.required_voice_hosts.length
  }
}

async function main() {
  const resultsDirectory = argument('--results')
  const commit = argument('--commit')
  const only = argument('--only')
  assert(resultsDirectory, 'usage: verify-beta-qualification.mjs --results DIR --commit SHA')
  const manifest = await loadJson(path.join(root, 'qualification', 'beta-manifest.json'))
  if (only === 'voice') manifest.required_package_smokes = []
  else assert(only === null, '--only accepts only `voice`')
  const directory = path.resolve(resultsDirectory)
  const files = (await readdir(directory))
    .filter((file) => file.endsWith('.json'))
    .sort()
  const results = await Promise.all(files.map((file) => loadJson(path.join(directory, file))))
  const summary = verifyQualification(manifest, results, commit)
  process.stdout.write(`${JSON.stringify(summary)}\n`)
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(import.meta.filename)) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  })
}
