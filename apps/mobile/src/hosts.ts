import type { Credential } from './api.ts';
import type { ThemePreference } from './theme.ts';

export type HostEntry = { id: string; name: string };
export type HostIndex = { active: string | null; hosts: HostEntry[] };
/** Tools are on unless listed in `disabledTools`, so tools a host adds later start enabled. */
export type Preferences = { theme: ThemePreference; disabledTools: string[] };

export const emptyIndex: HostIndex = { active: null, hosts: [] };
export const defaultPreferences: Preferences = { theme: 'system', disabledTools: [] };

/** One saved credential per host: pairing again with the same host replaces it. */
export function hostId(credential: Pick<Credential, 'instanceId'>): string {
  return credential.instanceId.replace(/[^A-Za-z0-9._-]/g, '_');
}

export function parseIndex(raw: string | null): HostIndex {
  if (!raw) return emptyIndex;
  const value = JSON.parse(raw) as HostIndex;
  if (!Array.isArray(value?.hosts)) throw new Error('Invalid saved host list.');
  const seen = new Set<string>();
  const hosts = value.hosts.filter((host) => {
    if (typeof host?.id !== 'string' || !host.id || typeof host.name !== 'string' || seen.has(host.id)) return false;
    seen.add(host.id);
    return true;
  }).map(({ id, name }) => ({ id, name }));
  const active = hosts.some(({ id }) => id === value.active) ? value.active : hosts[0]?.id ?? null;
  return { active, hosts };
}

/** Add or replace a host and make it the active one. */
export function upsertHost(index: HostIndex, entry: HostEntry): HostIndex {
  const existing = index.hosts.some(({ id }) => id === entry.id);
  return {
    active: entry.id,
    hosts: existing ? index.hosts.map((host) => host.id === entry.id ? entry : host) : [...index.hosts, entry],
  };
}

export function removeHost(index: HostIndex, id: string): HostIndex {
  const hosts = index.hosts.filter((host) => host.id !== id);
  return { active: index.active === id ? hosts[0]?.id ?? null : index.active, hosts };
}

export function renameHost(index: HostIndex, id: string, name: string): HostIndex {
  const trimmed = name.trim();
  if (!trimmed) return index;
  return { ...index, hosts: index.hosts.map((host) => host.id === id ? { id, name: trimmed } : host) };
}

export function selectHost(index: HostIndex, id: string): HostIndex {
  return index.hosts.some((host) => host.id === id) ? { ...index, active: id } : index;
}

export function parsePreferences(raw: string | null): Preferences {
  if (!raw) return defaultPreferences;
  try {
    const value = JSON.parse(raw) as Partial<Preferences>;
    return {
      theme: value.theme === 'dark' || value.theme === 'light' ? value.theme : 'system',
      disabledTools: Array.isArray(value.disabledTools)
        ? value.disabledTools.filter((name): name is string => typeof name === 'string') : [],
    };
  } catch {
    return defaultPreferences;
  }
}
