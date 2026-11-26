import { useEffect, useRef, useState } from 'react';
import { Alert, ScrollView, Text, TextInput, View } from 'react-native';
import type { Model } from './api.ts';
import {
  builtInModels, deleteModel, downloadModel, freeSpace, installedModels, repositoryChoices, usableRuntimes, type Download,
} from './local/device.ts';
import { formatBytes, type ModelChoice, type RuntimeId } from './local/huggingface.ts';
import { modelId, type InstalledModel } from './local/models.ts';
import { useTheme } from './theme.ts';
import { Action, errorText, IconButton, useStyles } from './ui.tsx';

const runtimeNames: Record<RuntimeId, string> = { 'llama.cpp': 'llama.cpp', mlx: 'MLX', apple: 'Apple Intelligence' };
const formats: Partial<Record<RuntimeId, string>> = { 'llama.cpp': 'GGUF files (llama.cpp)', mlx: 'MLX repositories' };

/** Models for "This device": add one from a Hugging Face repository, see what is stored, delete. */
export function Models({ onClose, onChanged }: { onClose: () => void; onChanged: () => void }) {
  const styles = useStyles();
  const palette = useTheme();
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [builtIn, setBuiltIn] = useState<Model[]>([]);
  const [usable, setUsable] = useState<RuntimeId[]>([]);
  const [free, setFree] = useState(0);
  const [slug, setSlug] = useState('');
  const [repository, setRepository] = useState<{ slug: string; choices: ModelChoice[] } | null>(null);
  const [listing, setListing] = useState(false);
  const [active, setActive] = useState<{ id: string; name: string; size: number; bytes: number } | null>(null);
  const [error, setError] = useState('');
  const download = useRef<Download | null>(null);

  async function refresh() {
    try {
      setInstalled(await installedModels());
      setFree(freeSpace());
    } catch (cause) {
      setError(errorText(cause));
    }
  }

  useEffect(() => {
    void refresh();
    void usableRuntimes().then(setUsable);
    void builtInModels().then(setBuiltIn);
    // Leaving the screen does not cancel: the download finishes and the model shows up in the chat's list.
  }, []);

  async function list() {
    setListing(true);
    setError('');
    setRepository(null);
    try {
      setRepository(await repositoryChoices(slug));
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setListing(false);
    }
  }

  async function start(repo: string, choice: ModelChoice) {
    const id = modelId(repo, choice);
    setError('');
    setActive({ id, name: choice.name, size: choice.size, bytes: 0 });
    let last = 0;
    const running = downloadModel(repo, choice, (bytes) => {
      // Progress arrives far more often than the screen needs to redraw.
      if (Date.now() - last < 250 && bytes < choice.size) return;
      last = Date.now();
      setActive((current) => current?.id === id ? { ...current, bytes } : current);
    });
    download.current = running;
    try {
      await running.done;
      onChanged();
    } catch (cause) {
      if (errorText(cause) !== 'Download cancelled.') setError(errorText(cause));
    } finally {
      download.current = null;
      setActive(null);
      await refresh();
    }
  }

  const remove = (model: InstalledModel) => Alert.alert(`Delete ${model.name}?`,
    `Frees ${formatBytes(model.size)}. Conversations are kept.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => {
        void deleteModel(model.id).then(onChanged, (cause) => setError(errorText(cause))).then(refresh);
      } },
    ]);

  const accepted = usable.flatMap((runtime) => formats[runtime] ? [formats[runtime]] : []).join(' and ');

  return <View style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="‹" label="Back" onPress={onClose} />
      <Text style={styles.topTitle}>Models on this device</Text>
    </View>
    <ScrollView contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled">
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      <Text style={styles.listMeta}>{formatBytes(free)} free on this device</Text>

      <Text style={styles.section}>Add from Hugging Face</Text>
      <TextInput accessibilityLabel="Hugging Face model" autoCapitalize="none" autoCorrect={false} style={styles.input}
        placeholder="owner/name" placeholderTextColor={palette.muted} value={slug} onChangeText={setSlug}
        onSubmitEditing={() => { if (slug.trim() && !listing) void list(); }} returnKeyType="search" />
      <Action label={listing ? 'Listing files…' : 'List files'} disabled={!slug.trim() || listing} onPress={() => void list()} />
      {accepted ? <Text style={styles.listMeta}>This device runs {accepted}.</Text> : null}
      {repository && repository.choices.length === 0
        ? <Text style={styles.emptyText}>{repository.slug} has no files this device can run.</Text> : null}
      {repository?.choices.map((choice) => {
        const id = modelId(repository.slug, choice);
        const progress = active?.id === id ? active : null;
        const have = installed.some((model) => model.id === id);
        const tooBig = choice.size > free;
        return <View key={id} style={styles.card}>
          <Text style={styles.body}>{choice.name}</Text>
          <Text style={styles.listMeta}>{runtimeNames[choice.runtime]}  ·  {formatBytes(choice.size)}
            {choice.files.length > 1 ? `  ·  ${choice.files.length} files` : ''}</Text>
          {progress ? <>
            <View accessibilityRole="progressbar" style={{ height: 6, borderRadius: 3, backgroundColor: palette.raised, overflow: 'hidden' }}>
              <View style={{ height: 6, backgroundColor: palette.accent, width: `${Math.min(100, progress.size ? progress.bytes / progress.size * 100 : 0)}%` }} />
            </View>
            <Text style={styles.listMeta}>
              {progress.size ? Math.min(100, Math.floor(progress.bytes / progress.size * 100)) : 0}%  ·  {formatBytes(progress.bytes)} of {formatBytes(progress.size)}</Text>
            <View style={styles.row}><Action label="Cancel download" quiet onPress={() => download.current?.cancel()} /></View>
          </> : have ? <Text style={styles.listMeta}>On this device</Text>
            : tooBig ? <Text style={styles.listMeta}>Not enough free space</Text>
            : <View style={styles.row}><Action label="Download" disabled={!!active} onPress={() => void start(repository.slug, choice)} /></View>}
        </View>;
      })}

      <Text style={styles.section}>On this device</Text>
      {builtIn.map((model) => <View key={model.id} style={styles.card}>
        <Text style={styles.body}>{model.name ?? model.id}</Text>
        <Text style={styles.listMeta}>Built in  ·  no download</Text>
      </View>)}
      {installed.length === 0 && builtIn.length === 0
        ? <Text style={styles.emptyText}>No models yet. Add one above to chat on this device.</Text> : null}
      {installed.map((model) => <View key={model.id} style={styles.card}>
        <Text style={styles.body}>{model.name}</Text>
        <Text style={styles.listMeta}>{runtimeNames[model.runtime]}  ·  {formatBytes(model.size)}
          {usable.length && !usable.includes(model.runtime) ? '  ·  cannot run on this device' : ''}</Text>
        <View style={styles.row}><Action label="Delete" danger onPress={() => remove(model)} /></View>
      </View>)}
    </ScrollView>
  </View>;
}
