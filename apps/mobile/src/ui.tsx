import { useMemo, type ReactNode } from 'react';
import { Modal, Platform, Pressable, SafeAreaView, StatusBar, StyleSheet, Text, View } from 'react-native';
import { useTheme, type Palette } from './theme.ts';

export const errorText = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong.';

const androidTop = Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) : 0;

const makeStyles = (p: Palette) => StyleSheet.create({
  safe: { flex: 1, backgroundColor: p.bg, paddingTop: androidTop },
  container: { flex: 1, paddingBottom: Platform.OS === 'android' ? 26 : 0 },
  topBar: {
    flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 8, paddingVertical: 6,
    borderBottomWidth: StyleSheet.hairlineWidth, borderColor: p.line,
  },
  topTitle: { flex: 1, fontSize: 16, fontWeight: '600', color: p.text },
  iconButton: { minWidth: 40, height: 40, alignItems: 'center', justifyContent: 'center', borderRadius: 10, paddingHorizontal: 8 },
  iconText: { fontSize: 20, color: p.soft },
  chip: {
    maxWidth: 190, flexDirection: 'row', alignItems: 'center', borderRadius: 999, borderWidth: 1, borderColor: p.line,
    backgroundColor: p.panel, paddingHorizontal: 12, paddingVertical: 7,
  },
  chipText: { fontSize: 13, fontWeight: '600', color: p.soft },
  brand: { fontSize: 27, fontWeight: '800', color: p.accent },
  subtitle: { fontSize: 14, color: p.muted, marginTop: 2 },
  header: { paddingHorizontal: 22, paddingVertical: 16 },
  panel: { padding: 18, gap: 12 },
  heading: { fontSize: 22, fontWeight: '700', color: p.text },
  body: { fontSize: 16, lineHeight: 23, color: p.text },
  label: { fontSize: 13, fontWeight: '600', color: p.muted },
  section: { fontSize: 12, fontWeight: '700', color: p.muted, textTransform: 'uppercase', letterSpacing: 0.6, marginTop: 10 },
  error: { padding: 10, color: p.danger, backgroundColor: p.dangerSoft, borderRadius: 8, overflow: 'hidden' },
  row: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 8 },
  button: { borderRadius: 10, backgroundColor: p.accent, paddingHorizontal: 14, paddingVertical: 11, alignItems: 'center' },
  buttonText: { color: p.onAccent, fontWeight: '700', fontSize: 15 },
  quiet: { backgroundColor: p.raised, borderWidth: 1, borderColor: p.line },
  quietText: { color: p.text, fontWeight: '600', fontSize: 15 },
  dangerText: { color: p.danger, fontWeight: '600', fontSize: 15 },
  disabled: { opacity: 0.45 },
  input: {
    borderWidth: 1, borderColor: p.line, borderRadius: 10, padding: 11, color: p.text,
    backgroundColor: p.panel, fontSize: 16,
  },
  invite: { minHeight: 66, textAlignVertical: 'top' },
  camera: { height: 330, overflow: 'hidden', borderRadius: 12 },
  flex: { flex: 1 },
  messages: { paddingHorizontal: 14, paddingVertical: 12, gap: 14 },
  userBubble: {
    alignSelf: 'flex-end', maxWidth: '88%', borderRadius: 16, paddingHorizontal: 13, paddingVertical: 9,
    backgroundColor: p.raised, borderWidth: 1, borderColor: p.line,
  },
  assistant: { gap: 6 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8, padding: 30 },
  emptyText: { fontSize: 15, color: p.muted, textAlign: 'center' },
  composer: {
    flexDirection: 'row', alignItems: 'flex-end', gap: 8, margin: 10, padding: 6, borderRadius: 18,
    borderWidth: 1, borderColor: p.line, backgroundColor: p.panel,
  },
  composeInput: { flex: 1, maxHeight: 130, fontSize: 16, color: p.text, paddingHorizontal: 8, paddingVertical: 8 },
  send: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: p.accent },
  sendText: { color: p.onAccent, fontSize: 18, fontWeight: '800' },
  status: { fontSize: 12, color: p.muted, paddingHorizontal: 16, paddingBottom: 2 },
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', flexDirection: 'row' },
  drawer: { width: '82%', maxWidth: 340, backgroundColor: p.panel, borderRightWidth: 1, borderColor: p.line, paddingTop: androidTop },
  sheetBackdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'flex-end' },
  sheet: {
    maxHeight: '80%', backgroundColor: p.panel, borderTopLeftRadius: 18, borderTopRightRadius: 18,
    borderWidth: 1, borderColor: p.line, paddingBottom: 18,
  },
  sheetHead: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 18, paddingTop: 14, paddingBottom: 6 },
  listItem: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 18, paddingVertical: 12 },
  listItemOn: { backgroundColor: p.accentSoft },
  listText: { flex: 1, fontSize: 15, color: p.text },
  listMeta: { fontSize: 12, color: p.muted },
  check: { width: 18, fontSize: 15, color: p.accent, fontWeight: '800' },
  segment: { flexDirection: 'row', borderRadius: 10, borderWidth: 1, borderColor: p.line, overflow: 'hidden' },
  segmentItem: { flex: 1, paddingVertical: 10, alignItems: 'center', backgroundColor: p.panel },
  segmentOn: { backgroundColor: p.accentSoft },
  segmentText: { fontSize: 14, fontWeight: '600', color: p.soft },
  segmentTextOn: { color: p.accent },
  card: { borderRadius: 12, borderWidth: 1, borderColor: p.line, backgroundColor: p.panel, padding: 12, gap: 8 },
});

export type Styles = ReturnType<typeof makeStyles>;

export function useStyles(): Styles {
  const palette = useTheme();
  return useMemo(() => makeStyles(palette), [palette]);
}

export function Action({ label, onPress, disabled, quiet = false, danger = false }: {
  label: string; onPress: () => void; disabled?: boolean; quiet?: boolean; danger?: boolean;
}) {
  const styles = useStyles();
  return <Pressable accessibilityRole="button" disabled={disabled} onPress={onPress}
    style={[styles.button, (quiet || danger) && styles.quiet, disabled && styles.disabled]}>
    <Text style={danger ? styles.dangerText : quiet ? styles.quietText : styles.buttonText}>{label}</Text>
  </Pressable>;
}

export function IconButton({ glyph, label, onPress, disabled }: {
  glyph: string; label: string; onPress: () => void; disabled?: boolean;
}) {
  const styles = useStyles();
  return <Pressable accessibilityRole="button" accessibilityLabel={label} disabled={disabled} onPress={onPress}
    hitSlop={6} style={[styles.iconButton, disabled && styles.disabled]}>
    <Text style={styles.iconText}>{glyph}</Text>
  </Pressable>;
}

/** Bottom sheet for pickers, the mobile stand-in for the desktop's popovers. */
export function Sheet({ title, visible, onClose, children }: {
  title: string; visible: boolean; onClose: () => void; children: ReactNode;
}) {
  const styles = useStyles();
  return <Modal transparent visible={visible} animationType="slide" onRequestClose={onClose}>
    <Pressable style={styles.sheetBackdrop} onPress={onClose} accessibilityLabel={`Close ${title}`}>
      <Pressable style={styles.sheet} onPress={() => undefined}>
        <View style={styles.sheetHead}>
          <Text style={[styles.heading, styles.flex]}>{title}</Text>
          <IconButton glyph="✕" label="Close" onPress={onClose} />
        </View>
        {children}
      </Pressable>
    </Pressable>
  </Modal>;
}

export function Screen({ children }: { children: ReactNode }) {
  const styles = useStyles();
  const palette = useTheme();
  return <SafeAreaView style={styles.safe}>
    <StatusBar barStyle={palette.dark ? 'light-content' : 'dark-content'} backgroundColor={palette.bg} />
    {children}
  </SafeAreaView>;
}
