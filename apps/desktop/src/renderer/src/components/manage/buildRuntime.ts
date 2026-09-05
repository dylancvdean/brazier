import type { RuntimeTarget } from '../../api'

export type BuildEngine =
  | 'llama.cpp'
  | 'mlx-lm'
  | 'mlx-vlm'
  | 'vllm'
  | 'whisper.cpp'
  | 'streaming-asr'
  | 'stable-diffusion.cpp'
  | 'personaplex'
  | 'personaplex-mlx'
  | 'whisperkit'

export function sourceRuntimeId(engine: BuildEngine, buildId: string): string {
  switch (engine) {
    case 'llama.cpp':
      return `source-${buildId}`
    case 'stable-diffusion.cpp':
      return `sdcpp-source-${buildId}`
    case 'whisper.cpp':
      return `whisper-source-${buildId}`
    case 'whisperkit':
      return `whisperkit-source-${buildId}`
    default:
      return `${engine}-source-${buildId}`
  }
}

export function targetSupportedByBuildEngine(
  engine: BuildEngine,
  target: RuntimeTarget
): boolean {
  return engine !== 'vllm' || (target !== 'vulkan' && target !== 'sycl')
}

export function appendBuildDiagnostics(
  lines: string[],
  diagnostics: Record<string, unknown> | undefined
): string[] {
  if (!diagnostics) return lines
  const hints = diagnostics.hints
  const excerpt = diagnostics.log_excerpt
  let next = [...lines]
  if (typeof excerpt === 'string' && excerpt.trim()) {
    const excerptLines = excerpt.trimEnd().split('\n')
    // Build output is streamed live before the terminal failure event includes
    // its full tail. Replace the overlapping streamed tail with that complete
    // excerpt instead of displaying the same output a second time.
    const recentStart = Math.max(0, next.length - 100)
    const overlap = excerptLines
      .filter((line) => line.length > 0)
      .map((line) => next.lastIndexOf(line))
      .find((index) => index >= recentStart)
    if (overlap != null && overlap >= recentStart) {
      next = [...next.slice(0, overlap), ...excerptLines]
    } else {
      next.push('', '--- last log lines ---', ...excerptLines)
    }
  }
  if (Array.isArray(hints) && hints.length > 0) {
    next.push('', 'Suggested fixes:')
    for (const hint of hints) {
      if (typeof hint === 'string') next.push(`• ${hint}`)
    }
  }
  return next
}
