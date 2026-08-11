import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'

const outputIndex = process.argv.indexOf('--output')
if (outputIndex < 0 || !process.argv[outputIndex + 1]) {
  throw new Error('usage: materialize-voice-qualification.mjs --output DIR')
}

const output = path.resolve(process.argv[outputIndex + 1])
await mkdir(output, { recursive: true })
for (const [name, variable] of [
  ['macos-apple-silicon.json', 'BRAZIER_MACOS_VOICE_RESULT'],
  ['linux-nvidia-x64.json', 'BRAZIER_LINUX_VOICE_RESULT']
]) {
  const raw = process.env[variable]
  if (!raw) throw new Error(`${variable} is required`)
  const parsed = JSON.parse(raw)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${variable} must contain one JSON object`)
  }
  await writeFile(path.join(output, name), `${JSON.stringify(parsed, null, 2)}\n`, { mode: 0o600 })
}
