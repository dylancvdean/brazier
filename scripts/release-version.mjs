const VERSION_PATTERN =
  /^(?<major>0|[1-9]\d*)\.(?<minor>0|[1-9]\d*)\.(?<patch>0|[1-9]\d*)(?:-beta\.(?<beta>0|[1-9]\d*))?$/

export function parseReleaseVersion(version) {
  const match = VERSION_PATTERN.exec(version)
  if (!match?.groups) {
    throw new Error(`Expected a stable version or -beta.N prerelease, received ${version}.`)
  }
  return match.groups
}

/** Compute the next immutable beta candidate; changing the core resets beta to 1. */
export function nextBetaVersion(version, bump = 'beta') {
  const parsed = parseReleaseVersion(version)
  let major = Number(parsed.major)
  let minor = Number(parsed.minor)
  let patch = Number(parsed.patch)
  const beta = parsed.beta == null ? null : Number(parsed.beta)

  switch (bump) {
    case '--major':
      major += 1
      minor = 0
      patch = 0
      break
    case '--minor':
      minor += 1
      patch = 0
      break
    case '--patch':
      patch += 1
      break
    case 'beta':
      if (beta == null) patch += 1
      break
    default:
      throw new Error(`Unsupported release bump ${bump}.`)
  }

  const nextBeta = bump === 'beta' && beta != null ? beta + 1 : 1
  return `${major}.${minor}.${patch}-beta.${nextBeta}`
}
