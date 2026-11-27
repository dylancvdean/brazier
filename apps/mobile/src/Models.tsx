import { useEffect, useState } from 'react';
import { Alert, ScrollView, Text, TextInput, View } from 'react-native';
import type { Model } from './api.ts';
import {
  activeDownload, builtInModels, cancelDownload, deleteModel, DownloadCancelled, downloadModel, freeSpace, installedModels,
  contextMemory, deviceMemory, repositoryChoices, usableRuntimes, watchDownload,
} from './local/device.ts';
import { formatBytes, type ModelChoice, type RuntimeId } from './local/huggingface.ts';
import { memoryWarning, type DeviceMemory } from './local/memory.ts';
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
  const [memory, setMemory] = useState<DeviceMemory>({});
  const [contexts, setContexts] = useState<Record<string, number | null>>({});
  const [slug, setSlug] = useState('');
  const [repository, setRepository] = useState<{ slug: string; choices: ModelChoice[] } | null>(null);
  const [listing, setListing] = useState(false);
  const [active, setActive] = useState(activeDownload);
  const [error, setError] = useState('');

  async function refresh() {
    try {
      const models = await installedModels();
      setInstalled(models);
      setMemory(await deviceMemory());
      setContexts(Object.fromEntries(await Promise.all(
        models.map(async (model) => [model.id, await contextMemory(model)] as const))));
      setFree(freeSpace());
    } catch (cause) {
      setError(errorText(cause));
    }
  }

  useEffect(() => {
    void refresh();
    void usableRuntimes().then(setUsable);
    void builtInModels().then(setBuiltIn);
    // Leaving the screen does not cancel a download; it is picked up again here when the screen reopens.
    return watchDownload(() => {
      const current = activeDownload();
      setActive(current);
      if (!current) { onChanged(); void refresh(); }
    });
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
    setError('');
    try {
      await downloadModel(repo, choice);
    } catch (cause) {
      if (!(cause instanceof DownloadCancelled)) setError(errorText(cause));
    }
  }

  const remove = (model: InstalledModel) => Alert.alert(`Delete ${model.name}?`,
    `Frees ${formatBytes(model.size)}. Conversations are kept.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => {
        void deleteModel(model.id).then(onChanged, (cause) => setError(errorText(cause))).then(refresh);
      } },
    ]);

  const percent = active?.size ? Math.min(100, active.bytes / active.size * 100) : 0;
  const tooLarge = (size: number, context?: number | null) => {
    const warning = memoryWarning(size, memory, context);
    return warning ? <Text style={[styles.listMeta, { color: palette.danger }]}>{warning}</Text> : null;
  };
  const accepted = usable.flatMap((runtime) => formats[runtime] ? [formats[runtime]] : []).join(' and ');

  return <View style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="‹" label="Back" onPress={onClose} />
      <Text style={styles.topTitle}>Models on this device</Text>
    </View>
    <ScrollView contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled">
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      <Text style={styles.listMeta}>{formatBytes(free)} free on this device</Text>

      {active ? <View style={styles.card}>
        <Text style={styles.body}>Downloading {active.name}</Text>
        <View accessibilityRole="progressbar" style={{ height: 6, borderRadius: 3, backgroundColor: palette.raised, overflow: 'hidden' }}>
          <View style={{ height: 6, backgroundColor: palette.accent, width: `${percent}%` }} />
        </View>
        <Text style={styles.listMeta}>{Math.floor(percent)}%  ·  {formatBytes(active.bytes)} of {formatBytes(active.size)}</Text>
        <View style={styles.row}><Action label="Cancel download" quiet onPress={cancelDownload} /></View>
      </View> : null}

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
          {tooLarge(choice.size)}
          {progress ? <Text style={styles.listMeta}>Downloading…</Text>
            : have ? <Text style={styles.listMeta}>On this device</Text>
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
        <Text style={styles.listMeta}>{model.slug}</Text>
        <Text style={styles.listMeta}>{runtimeNames[model.runtime]}  ·  {formatBytes(model.size)}
          {usable.length && !usable.includes(model.runtime) ? '  ·  cannot run on this device' : ''}</Text>
        {tooLarge(model.size, contexts[model.id])}
        <View style={styles.row}><Action label="Delete" danger onPress={() => remove(model)} /></View>
      </View>)}
    </ScrollView>
  </View>;
}
