import { useState } from 'react';
import { Alert, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import { DEVICE_ID, type HostIndex } from './hosts.ts';
import type { SavedHosts } from './storage.ts';
import { useTheme, type ThemePreference } from './theme.ts';
import { Action, dragDismiss, IconButton, useStyles } from './ui.tsx';

const themes: { value: ThemePreference; label: string }[] = [
  { value: 'system', label: 'System' }, { value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' },
];

export function hostLabel(index: HostIndex, id: string, origin: string): string {
  return index.hosts.find((host) => host.id === id)?.name || new URL(origin).host;
}

export function Settings({ saved, theme, error, onTheme, onSelect, onRename, onRemove, onAdd, onModels, onClose }: {
  saved: SavedHosts; theme: ThemePreference; error: string;
  onTheme: (theme: ThemePreference) => void; onSelect: (id: string) => void;
  onRename: (id: string, name: string) => void; onRemove: (id: string) => void;
  onAdd: () => void; onModels: () => void; onClose?: () => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);

  return <View style={styles.container}>
    <View style={styles.topBar}>
      {onClose ? <IconButton glyph="‹" label="Back" onPress={onClose} /> : null}
      <Text style={[styles.topTitle, !onClose && { paddingLeft: 10 }]}>Settings</Text>
    </View>
    <ScrollView contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled" keyboardDismissMode={dragDismiss}>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      <Text style={styles.section}>Appearance</Text>
      <View style={styles.segment}>
        {themes.map(({ value, label }) => <Pressable key={value} accessibilityRole="button"
          accessibilityState={{ selected: theme === value }}
          style={[styles.segmentItem, theme === value && styles.segmentOn]} onPress={() => onTheme(value)}>
          <Text style={[styles.segmentText, theme === value && styles.segmentTextOn]}>{label}</Text>
        </Pressable>)}
      </View>

      <Text style={styles.section}>Hosts</Text>
      <View style={[styles.card, saved.index.active === DEVICE_ID && { borderColor: palette.accent }]}>
        <Text style={styles.body}>This device{saved.index.active === DEVICE_ID ? '  ·  In use' : ''}</Text>
        <Text style={styles.listMeta}>Models run on this device. Text chat only; no host needed.</Text>
        <View style={styles.row}>
          {saved.index.active !== DEVICE_ID ? <Action label="Use this device" onPress={() => onSelect(DEVICE_ID)} /> : null}
          <Action label="Models" quiet onPress={onModels} />
        </View>
      </View>
      {saved.index.hosts.map(({ id }) => {
        const credential = saved.credentials[id];
        if (!credential) return null;
        const active = saved.index.active === id;
        const label = hostLabel(saved.index, id, credential.origin);
        return <View key={id} style={[styles.card, active && { borderColor: palette.accent }]}>
          {renaming?.id === id ? <>
            <TextInput accessibilityLabel="Host name" autoFocus style={styles.input} value={renaming.name}
              placeholderTextColor={palette.muted} onChangeText={(name) => setRenaming({ id, name })} />
            <View style={styles.row}>
              <Action label="Save" disabled={!renaming.name.trim()} onPress={() => {
                onRename(id, renaming.name); setRenaming(null);
              }} />
              <Action label="Cancel" quiet onPress={() => setRenaming(null)} />
            </View>
          </> : <>
            <Text style={styles.body}>{label}{active ? '  ·  Connected' : ''}</Text>
            <Text style={styles.listMeta}>{credential.origin}</Text>
            <View style={styles.row}>
              {!active ? <Action label="Use this host" onPress={() => onSelect(id)} /> : null}
              <Action label="Rename" quiet onPress={() => setRenaming({ id, name: label })} />
              <Action label="Remove" danger onPress={() => Alert.alert(
                `Remove ${label}?`, 'This device forgets the host. Revoke the device on the host to end its access.',
                [{ text: 'Cancel', style: 'cancel' }, { text: 'Remove', style: 'destructive', onPress: () => onRemove(id) }],
              )} />
            </View>
          </>}
        </View>;
      })}
      <Action label={saved.index.hosts.length ? 'Pair another host' : 'Pair a host'} quiet onPress={onAdd} />
    </ScrollView>
  </View>;
}
