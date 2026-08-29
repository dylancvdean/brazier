export type GithubReleaseAsset = {
  name: string
  browser_download_url: string
}

export type GithubRelease = {
  draft: boolean
  prerelease: boolean
  assets: GithubReleaseAsset[]
}

/**
 * Newest complete GitHub release that has this platform's updater manifest.
 *
 * A running prerelease must see other prereleases; otherwise a GitHub flag
 * change would freeze updates until someone bumped a stable version.
 */
export function pickReleaseManifestUrl(
  releases: GithubRelease[],
  manifest: string,
  includePrerelease: boolean
): string | undefined {
  return releases
    .filter((release) => !release.draft && (includePrerelease || !release.prerelease))
    .flatMap((release) => release.assets)
    .find((candidate) => candidate.name === manifest)?.browser_download_url
}

export function versionLooksLikePrerelease(version: string): boolean {
  return version.includes('-')
}
