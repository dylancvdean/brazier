import { pathToFileURL } from 'node:url'

/** The renderer may remain on exactly the document the main process loaded. */
export function isTrustedRendererUrl(
  candidate: string,
  options: { developmentUrl?: string; packagedIndexPath: string }
): boolean {
  let actual: URL
  try {
    actual = new URL(candidate)
  } catch {
    return false
  }
  const expected = options.developmentUrl
    ? new URL(options.developmentUrl)
    : pathToFileURL(options.packagedIndexPath)
  return (
    actual.protocol === expected.protocol &&
    actual.username === '' &&
    actual.password === '' &&
    actual.host === expected.host &&
    actual.pathname === expected.pathname &&
    actual.search === expected.search
  )
}

export function isSafeExternalUrl(candidate: string): boolean {
  try {
    const url = new URL(candidate)
    return url.protocol === 'https:' && url.username === '' && url.password === ''
  } catch {
    return false
  }
}
