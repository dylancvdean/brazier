import { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Pressable, Text, View } from 'react-native';
import { useVideoPlayer, VideoView } from 'expo-video';
import type { Credential } from './api.ts';
import { blobFile, saveToPhotos, share } from './blobFiles.ts';
import { mediaKind, type MediaBlob } from './media.ts';
import { useTheme } from './theme.ts';
import { errorText, useStyles } from './ui.tsx';

type Loaded = { uri?: string; error?: string; retry: () => void };
/** Where a blob's file comes from: a paired host, or something that already has the files. */
export type MediaSource = Credential | { file(blob: MediaBlob): Promise<string> };

export function useBlobFile(credential: MediaSource, blob: MediaBlob): Loaded {
  const [state, setState] = useState<{ uri?: string; error?: string }>({});
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let current = true;
    setState({});
    ('file' in credential ? credential.file(blob) : blobFile(credential, blob)).then(
      (uri) => { if (current) setState({ uri }); },
      (error) => { if (current) setState({ error: errorText(error) }); },
    );
    return () => { current = false; };
  }, [credential, blob.sha256, blob.mime_type, attempt]);
  return { ...state, retry: () => setAttempt((count) => count + 1) };
}

function Picture({ uri, label }: { uri: string; label: string }) {
  const [ratio, setRatio] = useState(1);
  useEffect(() => {
    let current = true;
    Image.getSize(uri, (width, height) => { if (current && width && height) setRatio(width / height); }, () => undefined);
    return () => { current = false; };
  }, [uri]);
  return <Image accessibilityLabel={label} source={{ uri }} resizeMode="contain"
    style={{ width: '100%', aspectRatio: ratio, maxHeight: 520 }} />;
}

function Video({ uri, label }: { uri: string; label: string }) {
  const player = useVideoPlayer(uri, (created) => { created.loop = true; });
  return <VideoView accessibilityLabel={label} player={player} nativeControls contentFit="contain"
    fullscreenOptions={{ enable: true }} style={{ width: '100%', aspectRatio: 16 / 9 }} />;
}

/** One generated image or video with its Save and Share actions. */
export function MediaView({ credential, blob, actions = true }: {
  credential: MediaSource; blob: MediaBlob; actions?: boolean;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const { uri, error, retry } = useBlobFile(credential, blob);
  const [note, setNote] = useState('');
  const [working, setWorking] = useState(false);
  const kind = mediaKind(blob.mime_type);
  const label = blob.name || (kind === 'video' ? 'Generated video' : kind === 'image' ? 'Generated image' : 'File');

  async function run(action: (uri: string) => Promise<void>, done: string) {
    if (!uri || working) return;
    setWorking(true);
    setNote('');
    try {
      await action(uri);
      setNote(done);
    } catch (cause) {
      setNote(errorText(cause));
    } finally {
      setWorking(false);
    }
  }

  const link = (text: string, onPress: () => void) => <Pressable accessibilityRole="button" hitSlop={8}
    disabled={working} onPress={onPress}><Text style={[styles.chipText, { color: palette.accent }]}>{text}</Text></Pressable>;

  return <View style={[styles.card, { padding: 0, gap: 0, overflow: 'hidden' }]}>
    {error ? <View style={[styles.row, { padding: 12 }]}>
      <Text accessibilityRole="alert" style={[styles.listMeta, styles.flex, { color: palette.danger }]}>
        Could not load {label}. {error}
      </Text>
      {link('Retry', retry)}
    </View>
      : !uri ? <View style={{ height: 160, alignItems: 'center', justifyContent: 'center' }}>
        <ActivityIndicator accessibilityLabel={`Loading ${label}`} color={palette.accent} />
      </View>
        : kind === 'image' ? <Picture uri={uri} label={label} />
          : kind === 'video' ? <Video uri={uri} label={label} />
            : <Text style={[styles.listText, { padding: 12 }]}>{label}</Text>}
    {actions && uri ? <View style={[styles.row, { paddingHorizontal: 12, paddingVertical: 9, gap: 18 }]}>
      {kind === 'other' ? null : link('Save to Photos', () => void run(saveToPhotos, 'Saved to Photos.'))}
      {link('Share', () => void run((file) => share(file, blob.mime_type), ''))}
      {note ? <Text numberOfLines={2} style={[styles.listMeta, styles.flex]}>{note}</Text> : null}
    </View> : null}
  </View>;
}

export function MediaList({ credential, blobs }: { credential: MediaSource; blobs: readonly MediaBlob[] }) {
  return <>{blobs.map((blob) => <MediaView key={`${blob.sha256}:${blob.mime_type}`} credential={credential} blob={blob} />)}</>;
}

/** Square preview for the history grid. Videos show a play mark instead of a frame. */
export function MediaThumb({ credential, blob, size }: { credential: MediaSource; blob: MediaBlob; size: number }) {
  const palette = useTheme();
  const video = mediaKind(blob.mime_type) === 'video';
  const box = {
    width: size, height: size, borderRadius: 8, backgroundColor: palette.raised,
    alignItems: 'center' as const, justifyContent: 'center' as const, overflow: 'hidden' as const,
  };
  if (video) return <View style={box}><Text style={{ fontSize: 22, color: palette.soft }}>▶</Text></View>;
  return <View style={box}><Thumb credential={credential} blob={blob} size={size} /></View>;
}

function Thumb({ credential, blob, size }: { credential: MediaSource; blob: MediaBlob; size: number }) {
  const palette = useTheme();
  const { uri, error } = useBlobFile(credential, blob);
  if (uri) return <Image source={{ uri }} style={{ width: size, height: size }} />;
  return error ? <Text style={{ color: palette.muted }}>!</Text> : <ActivityIndicator color={palette.muted} />;
}
