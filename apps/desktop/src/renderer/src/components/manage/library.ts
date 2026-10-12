import type { LocalModel } from '../../api'
import { modelLibraryKey } from '../../model-utils'

export type LibrarySort = 'largest' | 'smallest' | 'name'

export function libraryModelName(model: LocalModel): string {
  const key = modelLibraryKey(model.id)
  return key.split('/').at(-1) ?? key
}

export function modelWeightBytes(model: LocalModel): number {
  return model.size_bytes ?? 0
}

export function projectorBytes(model: LocalModel): number {
  return (model.projectors ?? []).reduce((total, projector) => total + projector.size_bytes, 0)
}

/**
 * Directory a GGUF model shares with its quants and mmproj files.
 * Null for models that are not local GGUF files.
 */
export function modelFolderKey(model: LocalModel): string | null {
  if (model.id.startsWith('gguf-ext:')) {
    const rest = model.id.slice('gguf-ext:'.length)
    const colon = rest.indexOf(':')
    if (colon === -1) return null
    const index = rest.slice(0, colon)
    const path = rest.slice(colon + 1)
    const slash = path.lastIndexOf('/')
    const dir = slash === -1 ? '' : path.slice(0, slash)
    return `gguf-ext:${index}:${dir}`
  }
  if (model.id.startsWith('gguf:')) {
    const path = model.id.slice('gguf:'.length)
    const slash = path.lastIndexOf('/')
    const dir = slash === -1 ? '' : path.slice(0, slash)
    return `gguf:${dir}`
  }
  return null
}

/** On-disk bytes, counting a shared projector once per folder. */
export function libraryDiskBytes(models: LocalModel[]): number {
  let total = 0
  const seenProjectors = new Set<string>()
  for (const model of models) {
    total += modelWeightBytes(model)
    const folder = modelFolderKey(model)
    if (!folder) continue
    for (const projector of model.projectors ?? []) {
      const key = `${folder}/${projector.name}`
      if (seenProjectors.has(key)) continue
      seenProjectors.add(key)
      total += projector.size_bytes
    }
  }
  return total
}

export function libraryModelFootprint(model: LocalModel): number {
  return modelWeightBytes(model) + projectorBytes(model)
}

export function sortLibraryModels(models: LocalModel[], sort: LibrarySort): LocalModel[] {
  return [...models].sort((left, right) => {
    if (sort === 'name') {
      return libraryModelName(left).localeCompare(libraryModelName(right))
    }
    const bySize =
      sort === 'largest'
        ? libraryModelFootprint(right) - libraryModelFootprint(left)
        : libraryModelFootprint(left) - libraryModelFootprint(right)
    return bySize || libraryModelName(left).localeCompare(libraryModelName(right))
  })
}

export function libraryModelMatches(model: LocalModel, query: string): boolean {
  const needle = query.trim().toLowerCase()
  if (!needle) return true
  const haystack = [
    model.id,
    model.library_label,
    model.engine,
    model.repo_id,
    libraryModelName(model),
    ...(model.projectors ?? []).map((projector) => projector.name)
  ]
    .filter((part): part is string => Boolean(part))
    .join(' ')
    .toLowerCase()
  return haystack.includes(needle)
}

export function isRemoteLibraryModel(model: LocalModel): boolean {
  return model.id.startsWith('remote:') || model.engine === 'remote'
}

export function isGgufLibraryModel(model: LocalModel): boolean {
  return model.id.startsWith('gguf:') || model.id.startsWith('gguf-ext:')
}
