import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';
import { parseCredential, type Credential } from './api.ts';
import {
  emptyIndex, hostId, parseIndex, parsePreferences, removeHost, upsertHost,
  type HostIndex, type Preferences,
} from './hosts.ts';

const legacyKey = 'brazier.host.v1';
const indexKey = 'brazier.hosts.v2';
const preferencesKey = 'brazier.preferences.v1';
const credentialKey = (id: string) => `brazier.host.v2.${id}`;
const androidEmulator = __DEV__ && Platform.OS === 'android';
const options: SecureStore.SecureStoreOptions = { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY };

export type SavedHosts = { index: HostIndex; credentials: Record<string, Credential> };

const writeIndex = (index: HostIndex) => SecureStore.setItemAsync(indexKey, JSON.stringify(index), options);

export async function loadHosts(): Promise<SavedHosts> {
  let index = parseIndex(await SecureStore.getItemAsync(indexKey, options));
  const credentials: Record<string, Credential> = {};
  for (const { id } of index.hosts) {
    const raw = await SecureStore.getItemAsync(credentialKey(id), options);
    if (raw) credentials[id] = parseCredential(raw, androidEmulator);
    else index = removeHost(index, id);
  }
  // Builds before multiple hosts kept a single credential under one key.
  const legacy = await SecureStore.getItemAsync(legacyKey, options);
  if (legacy) {
    const credential = parseCredential(legacy, androidEmulator);
    const id = hostId(credential);
    if (!credentials[id]) {
      await SecureStore.setItemAsync(credentialKey(id), JSON.stringify(credential), options);
      credentials[id] = credential;
      index = upsertHost(index, { id, name: '' });
      await writeIndex(index);
    }
    await SecureStore.deleteItemAsync(legacyKey, options);
  }
  return { index, credentials };
}

/** Save the credential first: an index entry without one is dropped on load. */
export async function saveHost(index: HostIndex, credential: Credential, name: string): Promise<HostIndex> {
  const id = hostId(credential);
  await SecureStore.setItemAsync(credentialKey(id), JSON.stringify(credential), options);
  const next = upsertHost(index, { id, name });
  await writeIndex(next);
  return next;
}

export async function saveIndex(index: HostIndex): Promise<void> {
  await writeIndex(index);
}

export async function deleteHost(index: HostIndex, id: string): Promise<HostIndex> {
  await SecureStore.deleteItemAsync(credentialKey(id), options);
  const next = removeHost(index, id);
  await writeIndex(next);
  return next;
}

export async function loadPreferences(): Promise<Preferences> {
  return parsePreferences(await SecureStore.getItemAsync(preferencesKey).catch(() => null));
}

export async function savePreferences(preferences: Preferences): Promise<void> {
  await SecureStore.setItemAsync(preferencesKey, JSON.stringify(preferences));
}

export { emptyIndex };
