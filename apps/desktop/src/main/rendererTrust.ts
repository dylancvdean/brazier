import { pathToFileURL } from 'node:url'

export function isLoopbackHostname(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  return (
    host === 'localhost' ||
    host === '127.0.0.1' ||
    host === '::1' ||
    host === '0:0:0:0:0:0:0:1' ||
    host === '::ffff:127.0.0.1'
  )
}

function effectivePort(url: URL): string {
  if (url.port) return url.port
  if (url.protocol === 'https:' || url.protocol === 'wss:') return '443'
  return '80'
}

function sameHttpFamily(actual: URL, expected: URL): boolean {
  const actualSecure = actual.protocol === 'https:' || actual.protocol === 'wss:'
  const expectedSecure = expected.protocol === 'https:' || expected.protocol === 'wss:'
  return actualSecure === expectedSecure
}

/** Chromium and Node disagree about `localhost` vs `127.0.0.1` vs `::1`. */
function sameHostOrDevLoopback(actual: URL, expected: URL): boolean {
  if (actual.host === expected.host) return true
  return (
    isLoopbackHostname(actual.hostname) &&
    isLoopbackHostname(expected.hostname) &&
    effectivePort(actual) === effectivePort(expected)
  )
}

/**
 * True when `candidate` is traffic to the Vite origin, including loopback
 * aliases and the `ws:`/`wss:` pair of the page's http(s) scheme.
 */
export function isRendererDevelopmentOrigin(candidate: string, developmentOrigin: string): boolean {
  let actual: URL
  let expected: URL
  try {
    actual = new URL(candidate)
    expected = new URL(developmentOrigin)
  } catch {
    return false
  }
  if (actual.username !== '' || actual.password !== '') return false
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(actual.protocol)) return false
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(expected.protocol)) return false
  return sameHttpFamily(actual, expected) && sameHostOrDevLoopback(actual, expected)
}

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
  if (
    actual.protocol !== expected.protocol ||
    actual.username !== '' ||
    actual.password !== '' ||
    actual.pathname !== expected.pathname ||
    actual.search !== expected.search
  ) {
    return false
  }
  if (options.developmentUrl) return sameHostOrDevLoopback(actual, expected)
  return actual.host === expected.host
}

/**
 * Main-frame documents are already constrained by will-navigate/will-redirect.
 * This guard is the renderer's fetch/WebSocket boundary. Cancelling the shell
 * document itself leaves a black window and ERR_BLOCKED_BY_CLIENT.
 */
export function shouldCancelRendererNetworkRequest(
  details: { url: string; webContentsId?: number; resourceType?: string },
  developmentOrigin: string | undefined,
  allows: (url: string, developmentOrigin?: string) => boolean
): boolean {
  if (!details.webContentsId) return false
  if (details.resourceType === 'mainFrame') return false
  if (developmentOrigin && isRendererDevelopmentOrigin(details.url, developmentOrigin)) {
    return false
  }
  return !allows(details.url, developmentOrigin)
}

export function isSafeExternalUrl(candidate: string): boolean {
  try {
    const url = new URL(candidate)
    return url.protocol === 'https:' && url.username === '' && url.password === ''
  } catch {
    return false
  }
}
