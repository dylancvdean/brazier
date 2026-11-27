import { useCallback, useEffect, useRef, useState } from 'react';
import { Keyboard, KeyboardAvoidingView, Platform, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import { DaemonError, type Model } from './api.ts';
import { Gallery } from './Gallery.tsx';
import { whenLabel } from './history.ts';
import { imageSizes } from './local/images.ts';
import {
  entryDetails, generateBody, generationModels, generationProgress, historyEntry, isCancellation,
  modelLabel, sizePresets, type ActiveGeneration, type GenerateForm, type Generator, type HistoryEntry, type Modality,
} from './media.ts';
import { MediaView } from './Media.tsx';
import { useTheme } from './theme.ts';
import { Action, dragDismiss, errorText, IconButton, Sheet, useStyles } from './ui.tsx';

const blank = (modality: Modality): GenerateForm => ({
  modality, modelId: '', prompt: '', negative: '', steps: '', seed: '', frames: '',
  width: sizePresets[modality][0]![0], height: sizePresets[modality][0]![1],
});

/**
 * Image and video generation on a paired host, the mobile form of the desktop's Generate mode, and image
 * generation on this device.
 */
export function Generate({ generator: api, hostName, visible, onClose, onUnauthorized, onOpenModels }: {
  generator: Generator; hostName: string; visible: boolean; onClose: () => void; onUnauthorized: (error: unknown) => void;
  /** On this device: where image models are added. */
  onOpenModels?: () => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const sizes = api.local ? imageSizes : null;
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
  const scroll = useRef<ScrollView>(null);
  // Android does not move the history search above the keyboard by itself: leave room under the list and bring
  // the field to the top while it is being typed in.
  const galleryTop = useRef(0);
  const searching = useRef(false);
  const [keyboard, setKeyboard] = useState(0);
  const showSearch = () => { if (searching.current) scroll.current?.scrollTo({ y: galleryTop.current, animated: true }); };
  useEffect(() => {
    if (Platform.OS !== 'android') return;
    const shown = Keyboard.addListener('keyboardDidShow', (event) => setKeyboard(event.endCoordinates.height));
    const hidden = Keyboard.addListener('keyboardDidHide', () => setKeyboard(0));
    return () => { shown.remove(); hidden.remove(); };
  }, []);
  useEffect(() => { if (keyboard) requestAnimationFrame(showSearch); }, [keyboard]);
  const set = (change: Partial<GenerateForm>) => setForm((previous) => ({ ...previous, ...change }));

  function report(cause: unknown) {
    if (cause instanceof DaemonError && cause.status === 401) onUnauthorized(cause);
    else setError(errorText(cause));
  }

  async function loadModels() {
    try {
      const found = await api.models();
      if (!alive.current) return;
      setModels(found);
      // A model on this device can have been deleted in Models since it was chosen.
      setForm((previous) => previous.modelId && !(api.local && !found.some(({ id }) => id === previous.modelId)) ? previous
        : { ...previous, modelId: generationModels(found, previous.modality)[0]?.id ?? '' });
    } catch (cause) {
      if (alive.current) { setModels((previous) => previous ?? []); report(cause); }
    }
  }

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, [api]);

  // Read again whenever the screen is shown, and when a job that was not started here ends: a host also lists
  // what its desktop app and other phones made.
  useEffect(() => {
    if (!visible || active) return;
    api.history().then((saved) => { if (alive.current) setHistory(saved); }, () => undefined);
  }, [api, visible, !!active]);

  // The screen stays mounted behind the chat, so models installed on the host
  // since it was last open are picked up when it is shown again.
  useEffect(() => {
    if (visible) void loadModels();
  }, [api, visible]);

  // The host runs one generation at a time, so this also shows one started
  // from a chat or another device, and lets it be cancelled here. This device
  // only runs what this screen started.
  useEffect(() => {
    if (api.local && !running) return undefined;
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
    Keyboard.dismiss();
    setRunning(true);
    try {
      const result = await api.generate(form.modality, body);
      const entry = historyEntry(`${Date.now()}-${result.blob.sha256.slice(0, 8)}`, form.modality, body, result,
        new Date().toISOString());
      // The history is a convenience; a result that could not be listed is still shown.
      const saved = await api.addHistory(entry).catch(() => null);
      if (!alive.current) return;
      if (saved) setHistory(saved);
      setShown(entry);
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
    // False when the host has not registered the job yet: Cancel must stay usable.
    let stopping = false;
    try {
      stopping = await api.cancel();
    } catch (cause) {
      report(cause);
    } finally {
      if (!running || !stopping) setCancelling(false);
    }
  }

  async function remove(entry: HistoryEntry) {
    try {
      const saved = await api.removeHistory(entry);
      if (!alive.current) return;
      setHistory(saved);
      setShown((current) => current?.id === entry.id ? null : current);
    } catch (cause) {
      if (cause instanceof DaemonError && cause.status === 401) onUnauthorized(cause);
      throw cause;
    }
  }

  function reuse(entry: HistoryEntry) {
    // An entry a host recorded without a size keeps the size already chosen, unless the kind changes.
    const [width, height] = entry.width > 0 && entry.height > 0 ? [entry.width, entry.height]
      : entry.modality === form.modality ? [form.width, form.height] : sizePresets[entry.modality][0]!;
    setError('');
    setShown(null);
    scroll.current?.scrollTo({ y: 0, animated: true });
    setForm({
      modality: entry.modality, prompt: entry.prompt, negative: entry.negativePrompt, width, height,
      modelId: generationModels(models ?? [], entry.modality).some(({ id }) => id === entry.modelId)
        ? entry.modelId : generationModels(models ?? [], entry.modality)[0]?.id ?? '',
      steps: entry.steps != null ? String(entry.steps) : '', seed: entry.seed != null ? String(entry.seed) : '',
      frames: entry.frames != null ? String(entry.frames) : '',
    });
  }

  // A host's models are named by their ids; this device's ids are directory names.
  const label = useCallback(
    (id: string) => (api.local ? models?.find((model) => model.id === id)?.name : undefined) ?? modelLabel(id),
    [api, models]);

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
        <Text numberOfLines={1} style={styles.chipText}>{form.modelId ? label(form.modelId) : 'Choose model'} ▾</Text>
      </Pressable>
    </View>
    <ScrollView ref={scroll} style={styles.flex} contentContainerStyle={[styles.panel, keyboard ? { paddingBottom: keyboard } : null]} keyboardShouldPersistTaps="handled" keyboardDismissMode={dragDismiss}>
      {api.local ? null : <View style={styles.segment}>
        {(['image', 'video'] as const).map((modality) => <Pressable key={modality} accessibilityRole="button"
          accessibilityState={{ selected: form.modality === modality }} disabled={running}
          style={[styles.segmentItem, form.modality === modality && styles.segmentOn]} onPress={() => switchModality(modality)}>
          <Text style={[styles.segmentText, form.modality === modality && styles.segmentTextOn]}>
            {modality === 'image' ? 'Image' : 'Video'}
          </Text>
        </Pressable>)}
      </View>}
      {models && !choices.length ? <Text style={styles.emptyText}>
        {api.local ? 'No image model is on this device. Add one in Models from a Hugging Face repository.'
          : `No ${form.modality} model is installed on ${hostName}. Add one in the desktop app’s Models view.`}
      </Text> : null}
      {models && !choices.length && onOpenModels ? <Action label="Add a model" onPress={onOpenModels} /> : null}
      <TextInput accessibilityLabel="Prompt" style={[styles.input, styles.invite]} multiline value={form.prompt}
        onChangeText={(prompt) => set({ prompt })} editable={!running} placeholderTextColor={palette.muted}
        placeholder={form.modality === 'image' ? 'Describe the image' : 'Describe the video'} />
      <TextInput accessibilityLabel="Negative prompt" style={styles.input} value={form.negative}
        onChangeText={(negative) => set({ negative })} editable={!running} placeholderTextColor={palette.muted}
        placeholder="Negative prompt (optional)" />
      <Text style={styles.label}>Size</Text>
      <View style={styles.row}>
        {(sizes ?? sizePresets[form.modality]).map(([width, height]) => {
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
        <MediaView credential={api} blob={shown.blob} />
        <Text selectable style={styles.body}>{shown.prompt}</Text>
        <Text style={styles.listMeta}>
          {[label(shown.modelId), entryDetails(shown), whenLabel(shown.createdAt)].filter(Boolean).join(' · ')}
        </Text>
      </View> : null}
      <Text style={styles.section} onLayout={(event) => { galleryTop.current = event.nativeEvent.layout.y; }}>History</Text>
      <Gallery source={api} history={history} label={label} reuseDisabled={running} onReuse={reuse} onRemove={remove}
        onSearchFocus={(focused) => { searching.current = focused; showSearch(); }}
        emptyText={api.local ? 'Images you generate here are kept on this phone.'
          : `Images and videos generated on ${hostName} are listed here, whichever device started them.`} />
    </ScrollView>
    <Sheet title={form.modality === 'image' ? 'Image model' : 'Video model'} visible={showModels}
      onClose={() => setShowModels(false)}>
      <ScrollView>
        {choices.map(({ id }) => <Pressable key={id} accessibilityRole="button"
          style={[styles.listItem, id === form.modelId && styles.listItemOn]}
          onPress={() => { set({ modelId: id }); setShowModels(false); }}>
          <Text style={styles.check}>{id === form.modelId ? '✓' : ''}</Text>
          <Text style={styles.listText}>{label(id)}</Text>
        </Pressable>)}
        {!choices.length ? <Text style={[styles.emptyText, { padding: 18 }]}>
          {models ? `No ${form.modality} model is installed on ${hostName}.` : 'Loading models…'}
        </Text> : null}
      </ScrollView>
    </Sheet>
  </KeyboardAvoidingView>;
}
