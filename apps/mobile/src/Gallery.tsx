import { useEffect, useMemo, useState } from 'react';
import { Alert, Modal, Pressable, ScrollView, Text, TextInput, useWindowDimensions, View } from 'react-native';
import { historyDays, searchHistory, whenLabel, type HistoryFilter } from './history.ts';
import { entryDetails, type HistoryEntry } from './media.ts';
import { MediaThumb, MediaView, type MediaSource } from './Media.tsx';
import { useTheme } from './theme.ts';
import { Action, errorText, IconButton, Screen, useStyles } from './ui.tsx';

const page = 24;
const columns = 3;
const gap = 6;
const filters: readonly (readonly [HistoryFilter, string])[] = [['all', 'All'], ['image', 'Images'], ['video', 'Videos']];

/**
 * Past generations as a grid by day, searchable by prompt and model. An entry opens with its media (save and
 * share), prompt, model, settings and date, and can be reused in the form or deleted.
 */
export function Gallery({ source, history, label, emptyText, reuseDisabled, onReuse, onRemove }: {
  source: MediaSource; history: readonly HistoryEntry[]; label: (modelId: string) => string; emptyText: string;
  reuseDisabled?: boolean; onReuse: (entry: HistoryEntry) => void;
  /** Rejects when the entry could not be deleted. */
  onRemove: (entry: HistoryEntry) => Promise<void>;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const { width } = useWindowDimensions();
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<HistoryFilter>('all');
  const [limit, setLimit] = useState(page);
  const [openId, setOpenId] = useState<string | null>(null);
  const [removing, setRemoving] = useState(false);
  const [error, setError] = useState('');

  const mixed = history.some(({ modality }) => modality !== history[0]?.modality);
  const found = useMemo(() => searchHistory(history, query, mixed ? filter : 'all', label), [history, query, filter, mixed, label]);
  const days = useMemo(() => historyDays(found.slice(0, limit)), [found, limit]);
  const index = openId ? found.findIndex(({ id }) => id === openId) : -1;
  const open = index >= 0 ? found[index]! : null;
  // 18 is the panel's padding on each side.
  const size = Math.floor((Math.min(width, 720) - 36 - gap * (columns - 1)) / columns);

  useEffect(() => { setLimit(page); }, [query, filter]);
  // The entry was deleted on another device, or a search no longer matches it.
  useEffect(() => { if (openId && index < 0) setOpenId(null); }, [openId, index]);

  function show(entry: HistoryEntry | undefined) {
    if (!entry) return;
    setError('');
    setOpenId(entry.id);
  }

  function remove(entry: HistoryEntry) {
    Alert.alert(`Delete this ${entry.modality}?`,
      entry.hostId ? 'It is removed from the host’s history and from this phone.' : 'It is removed from this phone.', [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete', style: 'destructive', onPress: () => {
            const next = found[index + 1] ?? found[index - 1];
            setRemoving(true);
            setError('');
            onRemove(entry).then(() => setOpenId(next?.id ?? null), (cause) => setError(errorText(cause)))
              .finally(() => setRemoving(false));
          },
        },
      ], { cancelable: true });
  }

  if (!history.length) return <Text style={styles.listMeta}>{emptyText}</Text>;

  return <View style={{ gap: 10 }}>
    <TextInput accessibilityLabel="Search history" style={styles.input} value={query} onChangeText={setQuery}
      placeholder="Search prompts and models" placeholderTextColor={palette.muted} autoCapitalize="none"
      autoCorrect={false} clearButtonMode="while-editing" returnKeyType="search" />
    {mixed ? <View style={styles.row}>
      {filters.map(([value, name]) => <Pressable key={value} accessibilityRole="button"
        accessibilityState={{ selected: filter === value }} onPress={() => setFilter(value)}
        style={[styles.chip, filter === value && { backgroundColor: palette.accentSoft, borderColor: palette.accent }]}>
        <Text style={[styles.chipText, filter === value && { color: palette.accent }]}>{name}</Text>
      </Pressable>)}
    </View> : null}
    <Text style={styles.listMeta}>
      {found.length === history.length ? `${history.length} saved` : `${found.length} of ${history.length}`}
    </Text>
    {found.length ? null : <Text style={styles.emptyText}>Nothing matches.</Text>}
    {days.map(({ day, entries }) => <View key={day} style={{ gap: 6 }}>
      <Text style={styles.label}>{day}</Text>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap }}>
        {entries.map((entry) => <Pressable key={entry.id} accessibilityRole="button"
          accessibilityLabel={`${entry.modality}: ${entry.prompt || 'no prompt'}`} onPress={() => show(entry)}>
          <MediaThumb credential={source} blob={entry.blob} size={size} />
        </Pressable>)}
      </View>
    </View>)}
    {found.length > limit
      ? <Action label={`Show more (${found.length - limit})`} quiet onPress={() => setLimit(limit + page)} /> : null}

    <Modal visible={!!open} animationType="slide" onRequestClose={() => setOpenId(null)}>
      <Screen>
        <View style={styles.topBar}>
          <IconButton glyph="✕" label="Close" onPress={() => setOpenId(null)} />
          <Text numberOfLines={1} style={styles.topTitle}>{index + 1} of {found.length}</Text>
          <IconButton glyph="‹" label="Newer" disabled={index <= 0} onPress={() => show(found[index - 1])} />
          <IconButton glyph="›" label="Older" disabled={index >= found.length - 1} onPress={() => show(found[index + 1])} />
        </View>
        {open ? <ScrollView contentContainerStyle={styles.panel}>
          <MediaView key={open.id} credential={source} blob={open.blob} />
          <Text style={styles.listMeta}>{whenLabel(open.createdAt)}</Text>
          <Text selectable style={styles.body}>{open.prompt || 'No prompt was recorded.'}</Text>
          {open.negativePrompt ? <>
            <Text style={styles.label}>Negative prompt</Text>
            <Text selectable style={styles.body}>{open.negativePrompt}</Text>
          </> : null}
          <Text style={styles.label}>Model</Text>
          <Text selectable style={styles.body}>{open.modelId ? label(open.modelId) : 'Unknown'}</Text>
          <Text style={styles.label}>Settings</Text>
          <Text selectable style={styles.body}>{entryDetails(open) || 'Not recorded'}</Text>
          {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
          <View style={styles.row}>
            <Action label="Reuse settings" quiet disabled={reuseDisabled || !open.prompt}
              onPress={() => { onReuse(open); setOpenId(null); }} />
            <Action label={removing ? 'Deleting…' : 'Delete'} danger disabled={removing} onPress={() => remove(open)} />
          </View>
        </ScrollView> : null}
      </Screen>
    </Modal>
  </View>;
}
