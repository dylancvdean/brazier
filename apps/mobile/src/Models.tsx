import { useEffect, useRef, useState } from 'react';
import { Alert, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import type { Model } from './api.ts';
import {
  catalogue, choiceContext, fitNote, memoryFit, quantisation, recommend, suggestedChoice, type CatalogueEntry, type Recommendation,
} from './local/catalogue.ts';
import {
  activeDownload, builtInModels, cancelDownload, deleteModel, DownloadCancelled, downloadModel, freeSpace, installedModels,
  contextMemory, deviceMemory, repositoryChoices, usableRuntimes, watchDownload,
} from './local/device.ts';
import { downloadCatalogueModel, searchHub } from './local/discover.ts';
import { formatBytes, type ModelChoice, type RuntimeId } from './local/huggingface.ts';
import { memoryBudget, memoryWarning, type DeviceMemory } from './local/memory.ts';
import { modelId, type InstalledModel } from './local/models.ts';
import {
  debouncedSearch, formatAge, formatCount, formatNames, mergeResults, namedRepository, searchFormats, sortNames,
  type SearchFormat, type SearchPage, type SearchRequest, type SearchResult, type SearchSort,
} from './local/search.ts';
import { useTheme } from './theme.ts';
import { Action, dragDismiss, errorText, IconButton, useStyles } from './ui.tsx';

const runtimeNames: Record<RuntimeId, string> = { 'llama.cpp': 'llama.cpp', mlx: 'MLX', apple: 'Apple Intelligence', 'sd.cpp': 'stable-diffusion.cpp · images',
};
const sorts = Object.keys(sortNames) as SearchSort[];
const kinds = [['chat', 'Chat'], ['image', 'Images']] as const;
// Recommendations shown for each kind before "Show all".
const SHORT_LIST = 3;
const SEARCH_PAUSE = 400;

type Opened = { loading: true } | { error: string } | { choices: ModelChoice[] };

/** Models for "This device": recommendations for it, Hugging Face search, what is stored, delete. */
export function Models({ onClose, onChanged }: { onClose: () => void; onChanged: () => void }) {
  const styles = useStyles();
  const palette = useTheme();
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [builtIn, setBuiltIn] = useState<Model[]>([]);
  const [usable, setUsable] = useState<RuntimeId[]>([]);
  const [free, setFree] = useState(0);
  const [memory, setMemory] = useState<DeviceMemory>({});
  const [contexts, setContexts] = useState<Record<string, number | null>>({});
  const [showAll, setShowAll] = useState<Record<string, boolean>>({});
  const [starting, setStarting] = useState('');
  const [query, setQuery] = useState('');
  const [chosenFormat, setFormat] = useState<SearchFormat | null>(null);
  const [sort, setSort] = useState<SearchSort>('downloads');
  const [found, setFound] = useState<{ key: string; results: SearchResult[]; next: string | null } | null>(null);
  const [searching, setSearching] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [opened, setOpened] = useState<Record<string, Opened>>({});
  const [active, setActive] = useState(activeDownload);
  const [error, setError] = useState('');

  const available = searchFormats(usable);
  const format = chosenFormat && available.includes(chosenFormat) ? chosenFormat : available[0] ?? null;
  const text = query.trim();
  const key = format && text ? `${format}\n${sort}\n${text}` : '';
  const current = useRef(key);
  current.current = key;

  const [search] = useState(() => debouncedSearch<SearchRequest & { key: string }, SearchPage>(
    (request) => searchHub(request),
    ({ request, result, error: cause }) => {
      setSearching(false);
      setSearchError(result ? '' : errorText(cause));
      setFound(result ? { key: request.key, ...result } : null);
    }, SEARCH_PAUSE));

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
    const unwatch = watchDownload(() => {
      const download = activeDownload();
      setActive(download);
      if (!download) { onChanged(); void refresh(); }
    });
    return () => { unwatch(); search.cancel(); };
  }, []);

  // Typing waits for a pause; a changed filter or sort searches at once.
  const typed = useRef(text);
  useEffect(() => {
    const immediately = typed.current === text;
    typed.current = text;
    setLoadingMore(false);
    setSearchError('');
    if (!format || !text) {
      search.cancel();
      setSearching(false);
      setFound(null);
      return;
    }
    setSearching(true);
    search.request({ query: text, format, sort, key }, immediately);
  }, [key]);

  async function more() {
    if (!found?.next || !format || loadingMore) return;
    const from = key;
    setLoadingMore(true);
    setSearchError('');
    try {
      const page = await searchHub({ query: text, format, sort }, found.next);
      if (current.current !== from) return;
      setFound({ key: from, results: mergeResults(found.results, page.results), next: page.next });
    } catch (cause) {
      if (current.current === from) setSearchError(errorText(cause));
    } finally {
      if (current.current === from) setLoadingMore(false);
    }
  }

  async function open(slug: string) {
    if (opened[slug] && !('error' in opened[slug])) {
      setOpened(({ [slug]: _, ...rest }) => rest);
      return;
    }
    setOpened((all) => ({ ...all, [slug]: { loading: true } }));
    let next: Opened;
    try {
      const { choices } = await repositoryChoices(slug);
      next = { choices: [...choices].sort((a, b) => a.size - b.size) };
    } catch (cause) {
      next = { error: errorText(cause) };
    }
    // Closed while its files were being listed: stay closed.
    setOpened((all) => all[slug] ? { ...all, [slug]: next } : all);
  }

  async function start(id: string, download: () => Promise<unknown>) {
    setError('');
    setStarting(id);
    try {
      await download();
    } catch (cause) {
      if (!(cause instanceof DownloadCancelled)) setError(errorText(cause));
    } finally {
      setStarting('');
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
  const busy = !!active || !!starting;
  const tooLarge = (size: number, context?: number | null) => {
    const warning = memoryWarning(size, memory, context);
    return warning ? <Text style={[styles.listMeta, { color: palette.danger }]}>{warning}</Text> : null;
  };
  const note = (found: ReturnType<typeof fitNote>) => found
    ? <Text style={[styles.listMeta, found.problem && { color: palette.danger }]}>{found.text}</Text> : null;
  const badge = (label: string) => <Text style={[styles.listMeta, {
    color: palette.accent, backgroundColor: palette.accentSoft, fontWeight: '700', borderRadius: 6, overflow: 'hidden',
    paddingHorizontal: 6, paddingVertical: 2,
  }]}>{label}</Text>;
  const chip = (label: string, on: boolean, onPress: () => void) => <Pressable key={label} accessibilityRole="button"
    accessibilityState={{ selected: on }} onPress={onPress}
    style={[styles.chip, on && { borderColor: palette.accent, backgroundColor: palette.accentSoft }]}>
    <Text style={[styles.chipText, on && { color: palette.accent }]}>{label}</Text>
  </Pressable>;
  /** The line under a download: its progress, that it is here already, or the button. */
  const state = (id: string, blocked: boolean, download: () => Promise<unknown>, label: string) =>
    active?.id === id || starting === id ? <Text style={styles.listMeta}>Downloading…</Text>
      : installed.some((model) => model.id === id) ? <Text style={styles.listMeta}>On this device</Text>
      : blocked ? null
      : <View style={styles.row}><Action label={label} disabled={busy} onPress={() => void start(id, download)} /></View>;

  const recommended = recommend(catalogue, { runtimes: usable, memory, free, installed: installed.map(({ id }) => id) });
  const budget = memoryBudget(memory);
  const suggestion = (item: Recommendation) => {
    const { entry } = item;
    const fit = item.installed ? null : fitNote(item.fit, item.space);
    return <View key={entry.id} style={[styles.card, item.best && { borderColor: palette.accent }]}>
      <View style={styles.row}>
        <Text style={styles.body}>{entry.name}</Text>
        {item.best ? badge(entry.kind === 'chat' ? 'Best for this device' : 'Best for images') : null}
      </View>
      <Text style={styles.listMeta}>{entry.description}</Text>
      <Text style={styles.listMeta}>{formatBytes(entry.size)}  ·  {entry.detail}  ·  {runtimeNames[entry.runtime]}</Text>
      {note(fit)}
      {state(item.modelId, !item.space, () => downloadCatalogueModel(entry as CatalogueEntry), `Download  ·  ${formatBytes(entry.size)}`)}
    </View>;
  };

  const named = namedRepository(text);
  const results = found?.key === key ? found.results : [];
  const listed = named && !results.some(({ slug }) => slug.toLowerCase() === named.toLowerCase())
    ? [{ slug: named, named: true as const }, ...results] : results;
  const repository = (result: SearchResult | { slug: string; named: true }) => {
    const files = opened[result.slug];
    const suggested = files && 'choices' in files ? suggestedChoice(files.choices, memory, free) : null;
    return <View key={result.slug} style={styles.card}>
      <Pressable accessibilityRole="button" accessibilityState={{ expanded: !!files }}
        accessibilityLabel={`${result.slug}, ${files ? 'hide' : 'show'} downloads`} onPress={() => void open(result.slug)} style={{ gap: 4 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <Text style={[styles.body, styles.flex]}>{result.slug}</Text>
          <Text style={styles.iconText}>{files ? '⌄' : '›'}</Text>
        </View>
        <Text style={styles.listMeta}>{'named' in result ? 'Open this repository' : [
          `${formatCount(result.downloads)} downloads`, `${formatCount(result.likes)} likes`, formatAge(result.updated, Date.now()),
          result.files === 1 ? '1 download' : `${result.files} sizes`,
        ].filter(Boolean).join('  ·  ')}</Text>
      </Pressable>
      {!files ? null : 'loading' in files ? <Text style={styles.listMeta}>Listing files…</Text>
        : 'error' in files ? <Text style={[styles.listMeta, { color: palette.danger }]}>{files.error} Tap to try again.</Text>
        : files.choices.length === 0 ? <Text style={styles.listMeta}>Nothing here that this device can run.</Text>
        : files.choices.map((choice) => {
          const id = modelId(result.slug, choice);
          const space = choice.size <= free;
          const quant = quantisation(choice.name);
          return <View key={id} style={{ borderTopWidth: 1, borderColor: palette.line, paddingTop: 8, gap: 4 }}>
            <View style={styles.row}>
              <Text style={styles.label}>{quant ?? choice.name}</Text>
              <Text style={styles.listMeta}>{formatBytes(choice.size)}</Text>
              {choice === suggested ? badge('Suggested') : null}
            </View>
            <Text style={styles.listMeta}>{quant ? `${choice.name}  ·  ` : ''}{runtimeNames[choice.runtime]}
              {choice.files.length > 1 ? `  ·  ${choice.files.length} files` : ''}</Text>
            {installed.some((model) => model.id === id) ? null
              : note(fitNote(memoryFit(choice.size, memory, choiceContext(choice.runtime)), space))}
            {state(id, !space, () => downloadModel(result.slug, choice), 'Download')}
          </View>;
        })}
    </View>;
  };

  return <View style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="‹" label="Back" onPress={onClose} />
      <Text style={styles.topTitle}>Models on this device</Text>
    </View>
    <ScrollView contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled" keyboardDismissMode={dragDismiss}>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      <Text style={styles.listMeta}>{formatBytes(free)} free on this device
        {budget ? `  ·  about ${formatBytes(budget)} of memory for a model` : ''}</Text>

      {active ? <View style={styles.card}>
        <Text style={styles.body}>Downloading {active.name}</Text>
        <View accessibilityRole="progressbar" style={{ height: 6, borderRadius: 3, backgroundColor: palette.raised, overflow: 'hidden' }}>
          <View style={{ height: 6, backgroundColor: palette.accent, width: `${percent}%` }} />
        </View>
        <Text style={styles.listMeta}>{Math.floor(percent)}%  ·  {formatBytes(active.bytes)} of {formatBytes(active.size)}</Text>
        <View style={styles.row}><Action label="Cancel download" quiet onPress={cancelDownload} /></View>
      </View> : null}

      {kinds.map(([kind, title]) => {
        const all = recommended.filter(({ entry }) => entry.kind === kind);
        if (all.length === 0) return null;
        const shown = showAll[kind] ? all : all.slice(0, SHORT_LIST);
        return <View key={kind} style={{ gap: 12 }}>
          <Text style={styles.section}>Recommended  ·  {title}</Text>
          {shown.map(suggestion)}
          {all.length > SHORT_LIST ? <Action quiet label={showAll[kind] ? 'Show fewer' : `Show all ${all.length}`}
            onPress={() => setShowAll((now) => ({ ...now, [kind]: !now[kind] }))} /> : null}
        </View>;
      })}

      {format ? <>
        <Text style={styles.section}>Search Hugging Face</Text>
        <TextInput accessibilityLabel="Search Hugging Face models" autoCapitalize="none" autoCorrect={false} style={styles.input}
          placeholder="Search, or paste owner/name" placeholderTextColor={palette.muted} value={query} onChangeText={setQuery}
          onSubmitEditing={() => { if (text && format) { setSearching(true); search.request({ query: text, format, sort, key }, true); } }}
          returnKeyType="search" clearButtonMode="while-editing" />
        {available.length > 1
          ? <View style={styles.row}>{available.map((item) => chip(formatNames[item], item === format, () => setFormat(item)))}</View> : null}
        <View style={styles.row}>{sorts.map((item) => chip(sortNames[item], item === sort, () => setSort(item)))}</View>
        {searchError ? <Text accessibilityRole="alert" style={styles.error}>{searchError}</Text> : null}
        {!text ? <Text style={styles.listMeta}>Only models this device can run are listed: {available.map((item) => formatNames[item]).join(', ')}.</Text>
          : searching && listed.length === 0 ? <Text style={styles.listMeta}>Searching…</Text>
          : listed.length === 0 && !searchError ? <Text style={styles.emptyText}>No models this device can run match “{text}”.</Text> : null}
        {listed.map(repository)}
        {found?.key === key && found.next
          ? <Action quiet label={loadingMore ? 'Loading…' : 'Show more results'} disabled={loadingMore} onPress={() => void more()} /> : null}
      </> : null}

      <Text style={styles.section}>On this device</Text>
      {builtIn.map((model) => <View key={model.id} style={styles.card}>
        <Text style={styles.body}>{model.name ?? model.id}</Text>
        <Text style={styles.listMeta}>Built in  ·  no download</Text>
      </View>)}
      {installed.length === 0 && builtIn.length === 0
        ? <Text style={styles.emptyText}>No models yet. Add one above to chat or make images on this device.</Text> : null}
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
