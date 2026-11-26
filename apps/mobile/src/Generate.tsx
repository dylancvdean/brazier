import { useEffect, useMemo, useRef, useState } from 'react';
import { KeyboardAvoidingView, Platform, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import { fetch } from 'expo/fetch';
import { DaemonError, type Credential, type Model } from './api.ts';
import { loadHistory, saveHistory } from './blobFiles.ts';
import {
  addHistory, entryDetails, generateBody, generationModels, generationProgress, historyEntry, isCancellation,
  mediaClient, modelLabel, sizePresets, type ActiveGeneration, type GenerateForm, type HistoryEntry, type Modality,
} from './media.ts';
import { MediaThumb, MediaView } from './Media.tsx';
import { useTheme } from './theme.ts';
import { Action, errorText, IconButton, Sheet, useStyles } from './ui.tsx';

const blank = (modality: Modality): GenerateForm => ({
  modality, modelId: '', prompt: '', negative: '', steps: '', seed: '', frames: '',
  width: sizePresets[modality][0]![0], height: sizePresets[modality][0]![1],
});

/** Image and video generation on the host, the mobile form of the desktop's Generate mode. */
export function Generate({ credential, hostName, visible, onClose, onUnauthorized }: {
  credential: Credential; hostName: string; visible: boolean; onClose: () => void; onUnauthorized: (error: unknown) => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const api = useMemo(() => mediaClient(fetch, credential), [credential]);
  const [models, setModels] = useState<Model[] | null>(null);
  const [form, setForm] = useState(() => blank('image'));
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [shown, setShown] = useState<HistoryEntry | null>(null);
  const [active, setActive] = useState<ActiveGeneration | null>(null);
  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState('');
  const [showModels, setShowModels] = useState(false);
  const alive = useRef(true);
  // The history as last loaded or stored. A generation that finishes after this
  // screen is gone (the host was switched) still adds its result to the file.
  const stored = useRef<HistoryEntry[]>([]);
  const set = (change: Partial<GenerateForm>) => setForm((previous) => ({ ...previous, ...change }));

  function store(next: HistoryEntry[]) {
    stored.current = next;
    if (alive.current) setHistory(next);
    try { saveHistory(credential, next); } catch { /* History is a convenience; the result is still shown. */ }
  }

  function report(cause: unknown) {
    if (cause instanceof DaemonError && cause.status === 401) onUnauthorized(cause);
    else setError(errorText(cause));
  }

  async function loadModels() {
    try {
      const found = await api.models();
      if (!alive.current) return;
      setModels(found);
      setForm((previous) => previous.modelId ? previous
        : { ...previous, modelId: generationModels(found, previous.modality)[0]?.id ?? '' });
    } catch (cause) {
      if (alive.current) { setModels((previous) => previous ?? []); report(cause); }
    }
  }

  useEffect(() => {
    alive.current = true;
    loadHistory(credential).then((saved) => {
      stored.current = saved;
      if (alive.current) setHistory(saved);
    }, () => undefined);
    return () => { alive.current = false; };
  }, [api]);

  // The screen stays mounted behind the chat, so models installed on the host
  // since it was last open are picked up when it is shown again.
  useEffect(() => {
    if (visible) void loadModels();
  }, [api, visible]);

  // The host runs one generation at a time, so this also shows one started
  // from a chat or another device, and lets it be cancelled here.
  useEffect(() => {
    let current = true;
    const poll = () => api.active().then((found) => { if (current) setActive(found); }, () => undefined);
    void poll();
    const timer = setInterval(poll, running || active ? 1000 : 5000);
    return () => { current = false; clearInterval(timer); };
  }, [api, running, !!active]);

  const choices = generationModels(models ?? [], form.modality);
  const busy = running || !!active;
  const progress = generationProgress(active);

  function switchModality(modality: Modality) {
    if (modality === form.modality) return;
    const [width, height] = sizePresets[modality][0]!;
    set({ modality, width, height, modelId: generationModels(models ?? [], modality)[0]?.id ?? '' });
  }

  async function generate() {
    if (busy) return;
    const body = generateBody(form);
    if (typeof body === 'string') { setError(body); return; }
    setError('');
    setRunning(true);
    try {
      const result = await api.generate(form.modality, body);
      const entry = historyEntry(`${Date.now()}-${result.blob.sha256.slice(0, 8)}`, form.modality, body, result,
        new Date().toISOString());
      store(addHistory(stored.current, entry));
      if (alive.current) setShown(entry);
    } catch (cause) {
      if (!alive.current) return;
      if (isCancellation(cause)) setError('Generation stopped.');
      else report(cause);
    } finally {
      if (alive.current) { setRunning(false); setCancelling(false); setActive(null); }
    }
  }

  async function cancel() {
    setCancelling(true);
    try {
      await api.cancel();
    } catch (cause) {
      report(cause);
    } finally {
      if (!running) setCancelling(false);
    }
  }

  function remove(entry: HistoryEntry) {
    store(history.filter(({ id }) => id !== entry.id));
    if (shown?.id === entry.id) setShown(null);
  }

  function reuse(entry: HistoryEntry) {
    setForm({
      modality: entry.modality, prompt: entry.prompt, negative: entry.negativePrompt, width: entry.width, height: entry.height,
      modelId: generationModels(models ?? [], entry.modality).some(({ id }) => id === entry.modelId)
        ? entry.modelId : generationModels(models ?? [], entry.modality)[0]?.id ?? '',
      steps: entry.steps != null ? String(entry.steps) : '', seed: entry.seed != null ? String(entry.seed) : '',
      frames: entry.frames != null ? String(entry.frames) : '',
    });
  }

  const field = (label: string, value: string, onChange: (text: string) => void, placeholder: string) =>
    <View style={[styles.flex, { gap: 4 }]}>
      <Text style={styles.label}>{label}</Text>
      <TextInput accessibilityLabel={label} style={styles.input} value={value} onChangeText={onChange}
        placeholder={placeholder} placeholderTextColor={palette.muted} keyboardType="number-pad" editable={!running} />
    </View>;

  return <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="‹" label="Back to chat" onPress={onClose} />
      <Text numberOfLines={1} style={styles.topTitle}>Generate</Text>
      <Pressable accessibilityRole="button" accessibilityLabel="Choose generation model" disabled={running}
        style={[styles.chip, running && styles.disabled]} onPress={() => setShowModels(true)}>
        <Text numberOfLines={1} style={styles.chipText}>{form.modelId ? modelLabel(form.modelId) : 'Choose model'} ▾</Text>
      </Pressable>
    </View>
    <ScrollView style={styles.flex} contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled">
      <View style={styles.segment}>
        {(['image', 'video'] as const).map((modality) => <Pressable key={modality} accessibilityRole="button"
          accessibilityState={{ selected: form.modality === modality }} disabled={running}
          style={[styles.segmentItem, form.modality === modality && styles.segmentOn]} onPress={() => switchModality(modality)}>
          <Text style={[styles.segmentText, form.modality === modality && styles.segmentTextOn]}>
            {modality === 'image' ? 'Image' : 'Video'}
          </Text>
        </Pressable>)}
      </View>
      {models && !choices.length ? <Text style={styles.emptyText}>
        No {form.modality} model is installed on {hostName}. Add one in the desktop app’s Models view.
      </Text> : null}
      <TextInput accessibilityLabel="Prompt" style={[styles.input, styles.invite]} multiline value={form.prompt}
        onChangeText={(prompt) => set({ prompt })} editable={!running} placeholderTextColor={palette.muted}
        placeholder={form.modality === 'image' ? 'Describe the image' : 'Describe the video'} />
      <TextInput accessibilityLabel="Negative prompt" style={styles.input} value={form.negative}
        onChangeText={(negative) => set({ negative })} editable={!running} placeholderTextColor={palette.muted}
        placeholder="Negative prompt (optional)" />
      <Text style={styles.label}>Size</Text>
      <View style={styles.row}>
        {sizePresets[form.modality].map(([width, height]) => {
          const on = form.width === width && form.height === height;
          return <Pressable key={`${width}x${height}`} accessibilityRole="button" accessibilityState={{ selected: on }}
            disabled={running} style={[styles.chip, on && { backgroundColor: palette.accentSoft, borderColor: palette.accent }]}
            onPress={() => set({ width, height })}>
            <Text style={[styles.chipText, on && { color: palette.accent }]}>{width}×{height}</Text>
          </Pressable>;
        })}
      </View>
      <View style={{ flexDirection: 'row', gap: 10 }}>
        {field('Steps', form.steps, (steps) => set({ steps }), 'Default')}
        {field('Seed', form.seed, (seed) => set({ seed }), 'Random')}
        {form.modality === 'video' ? field('Frames', form.frames, (frames) => set({ frames }), 'Default') : null}
      </View>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      {busy ? <View style={styles.card}>
        <Text accessibilityRole="progressbar" style={styles.listText}>
          {cancelling ? 'Stopping…' : progress.label}
        </Text>
        {active && !running ? <Text numberOfLines={2} style={styles.listMeta}>
          {active.origin === 'model' ? 'Started by a chat' : 'Started elsewhere'}: {active.prompt}
        </Text> : null}
        <View style={{ height: 4, borderRadius: 2, backgroundColor: palette.raised, overflow: 'hidden' }}>
          <View style={{ height: 4, width: `${Math.round((progress.fraction ?? 0) * 100)}%`, backgroundColor: palette.accent }} />
        </View>
        <Action label="Cancel" quiet disabled={cancelling} onPress={() => void cancel()} />
      </View> : <Action label={form.modality === 'image' ? 'Generate image' : 'Generate video'}
        disabled={!form.modelId || !form.prompt.trim()} onPress={() => void generate()} />}
      {shown ? <View style={{ gap: 8 }}>
        <MediaView credential={credential} blob={shown.blob} />
        <Text selectable style={styles.body}>{shown.prompt}</Text>
        <Text style={styles.listMeta}>{modelLabel(shown.modelId)} · {entryDetails(shown)}</Text>
        <View style={styles.row}>
          <Action label="Reuse settings" quiet disabled={running} onPress={() => reuse(shown)} />
          <Action label="Remove" danger onPress={() => remove(shown)} />
        </View>
      </View> : null}
      <Text style={styles.section}>History</Text>
      {history.length ? <View style={styles.row}>
        {history.map((entry) => <Pressable key={entry.id} accessibilityRole="button"
          accessibilityLabel={`${entry.modality}: ${entry.prompt}`} onPress={() => setShown(entry)}
          style={shown?.id === entry.id ? { borderRadius: 10, borderWidth: 2, borderColor: palette.accent } : { padding: 2 }}>
          <MediaThumb credential={credential} blob={entry.blob} size={74} />
        </Pressable>)}
      </View> : <Text style={styles.listMeta}>Images and videos you generate here are listed on this phone.</Text>}
    </ScrollView>
    <Sheet title={form.modality === 'image' ? 'Image model' : 'Video model'} visible={showModels}
      onClose={() => setShowModels(false)}>
      <ScrollView>
        {choices.map(({ id }) => <Pressable key={id} accessibilityRole="button"
          style={[styles.listItem, id === form.modelId && styles.listItemOn]}
          onPress={() => { set({ modelId: id }); setShowModels(false); }}>
          <Text style={styles.check}>{id === form.modelId ? '✓' : ''}</Text>
          <Text style={styles.listText}>{modelLabel(id)}</Text>
        </Pressable>)}
        {!choices.length ? <Text style={[styles.emptyText, { padding: 18 }]}>
          {models ? `No ${form.modality} model is installed on ${hostName}.` : 'Loading models…'}
        </Text> : null}
      </ScrollView>
    </Sheet>
  </KeyboardAvoidingView>;
}
