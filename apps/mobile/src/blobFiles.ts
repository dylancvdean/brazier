import { fetch } from 'expo/fetch';
import { Directory, File, Paths } from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import * as Sharing from 'expo-sharing';
import type { Credential } from './api.ts';
import { hostId } from './hosts.ts';
import { blobFileName, mediaClient, parseHistory, type HistoryEntry, type MediaBlob } from './media.ts';

// Images, the video player, Photos and the share sheet cannot send the bearer
// key or use the host's pinned certificate. Blobs are therefore downloaded
// through expo/fetch, which does both, and shown from the app's cache. A blob
// is named by its hash, so a cached copy never goes stale. The bytes are not
// checked against that hash, so each host has its own cache: a host can then
// only supply the media shown for itself.
const hostKey = (credential: Credential) => hostId(credential).replace(/[^A-Za-z0-9_-]/g, '_');

const directory = (credential: Credential) => {
  const cache = new Directory(Paths.cache, 'blobs', hostKey(credential));
  if (!cache.exists) cache.create({ intermediates: true, idempotent: true });
  return cache;
};

const downloads = new Map<string, Promise<string>>();

async function download(credential: Credential, blob: MediaBlob): Promise<string> {
  const target = new File(directory(credential), blobFileName(blob));
  if (target.exists) return target.uri;
  const partial = new File(directory(credential), `${blob.sha256}.${Date.now()}.part`);
  let moved = false;
  try {
    const reader = (await mediaClient(fetch, credential).blob(blob.sha256)).getReader();
    partial.create();
    const handle = partial.open();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value?.length) handle.writeBytes(value);
      }
    } finally {
      handle.close();
    }
    if (!target.exists) {
      // move() repoints `partial` at the destination, so it must not be deleted below.
      partial.move(target);
      moved = true;
    }
    return target.uri;
  } finally {
    if (!moved && partial.exists) partial.delete();
  }
}

/** URI of a local copy of the blob, downloading it from the host if needed. */
export function blobFile(credential: Credential, blob: MediaBlob): Promise<string> {
  const name = `${hostKey(credential)}/${blobFileName(blob)}`;
  let pending = downloads.get(name);
  if (!pending) {
    pending = download(credential, blob).finally(() => downloads.delete(name));
    downloads.set(name, pending);
  }
  return pending;
}

export async function saveToPhotos(uri: string): Promise<void> {
  const permission = await MediaLibrary.requestPermissionsAsync(true, ['photo', 'video']);
  if (!permission.granted) throw new Error('Allow Brazier to add to Photos in the system settings, then try again.');
  await MediaLibrary.saveToLibraryAsync(uri);
}

export async function share(uri: string, mimeType: string): Promise<void> {
  if (!await Sharing.isAvailableAsync()) throw new Error('Sharing is not available on this device.');
  await Sharing.shareAsync(uri, { mimeType });
}

const historyFile = (credential: Credential) =>
  new File(Paths.document, `generate-history-${hostKey(credential)}.json`);

export async function loadHistory(credential: Credential): Promise<HistoryEntry[]> {
  const file = historyFile(credential);
  return file.exists ? parseHistory(await file.text()) : [];
}

export function saveHistory(credential: Credential, history: readonly HistoryEntry[]): void {
  const file = historyFile(credential);
  if (!file.exists) file.create({ intermediates: true });
  file.write(JSON.stringify(history));
}
