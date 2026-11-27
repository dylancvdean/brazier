import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import type { Conversation } from './api.ts';
import { conversationDay } from './conversationDates.ts';
import type { ChatBackend } from './local/engine.ts';
import { useTheme } from './theme.ts';
import { dragDismiss, errorText, IconButton, useStyles } from './ui.tsx';

/**
 * The drawer's conversations: search by title or message text, open, rename and delete. The same for a host's
 * conversations and this device's.
 */
export function Conversations({ api, conversations, selectedId, disabled, loading, onOpen, onRenamed, onDeleted, onUnauthorized }: {
  api: Pick<ChatBackend, 'conversations' | 'renameConversation' | 'deleteConversation'>;
  /** Everything saved, newest first; what is listed when nothing is searched for. */
  conversations: readonly Conversation[]; selectedId?: string; disabled?: boolean; loading?: boolean;
  onOpen: (conversation: Conversation) => void; onRenamed: (conversation: Conversation) => void;
  onDeleted: (id: string) => void;
  /** Returns true when it handled the error (the host no longer accepts this phone). */
  onUnauthorized: (error: unknown) => boolean;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const [query, setQuery] = useState('');
  const [found, setFound] = useState<Conversation[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [menu, setMenu] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const search = useRef(0);
  const words = query.trim();

  const fail = (cause: unknown) => { if (!onUnauthorized(cause)) setError(errorText(cause)); };

  // Searched after a pause in typing; a host is asked over the network.
  useEffect(() => {
    const run = search.current += 1;
    if (!words) { setFound(null); setSearching(false); return undefined; }
    setSearching(true);
    const timer = setTimeout(() => {
      api.conversations(words).then((result) => { if (search.current === run) { setFound(result); setError(''); } },
        (cause) => { if (search.current === run) fail(cause); })
        .finally(() => { if (search.current === run) setSearching(false); });
    }, 250);
    return () => clearTimeout(timer);
  }, [api, words]);

  const known = new Map(conversations.map((chat) => [chat.id, chat]));
  // Results show the titles as they are now, and nothing deleted since the search ran.
  const listed = found && words ? found.flatMap((chat) => known.get(chat.id) ?? []) : conversations;

  function closeMenu() { setMenu(null); setRenaming(null); setError(''); }

  async function rename(chat: Conversation) {
    if (!title.trim() || title.trim() === chat.title) { closeMenu(); return; }
    setWorking(true);
    try {
      onRenamed(await api.renameConversation(chat.id, title));
      closeMenu();
    } catch (cause) {
      fail(cause);
    } finally {
      setWorking(false);
    }
  }

  function remove(chat: Conversation) {
    Alert.alert('Delete this conversation?', `“${chat.title}” and its messages are deleted. This cannot be undone.`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive', onPress: () => {
          setWorking(true);
          api.deleteConversation(chat.id).then(() => { closeMenu(); onDeleted(chat.id); }, fail)
            .finally(() => setWorking(false));
        },
      },
    ], { cancelable: true });
  }

  const link = (text: string, onPress: () => void, danger = false) => <Pressable accessibilityRole="button"
    disabled={working || disabled} onPress={onPress} hitSlop={8} style={(working || disabled) && styles.disabled}>
    <Text style={[styles.chipText, { color: danger ? palette.danger : palette.accent }]}>{text}</Text>
  </Pressable>;

  return <View style={styles.flex}>
    {conversations.length ? <View style={{ paddingHorizontal: 18, paddingTop: 8 }}>
      <TextInput accessibilityLabel="Search conversations" style={styles.input} value={query} onChangeText={setQuery}
        placeholder="Search conversations" placeholderTextColor={palette.muted} autoCapitalize="none"
        autoCorrect={false} clearButtonMode="while-editing" returnKeyType="search" />
    </View> : null}
    <ScrollView style={styles.flex} keyboardShouldPersistTaps="handled" keyboardDismissMode={dragDismiss}>
      {loading || searching ? <ActivityIndicator style={{ padding: 8 }} color={palette.accent} /> : null}
      {error && !menu ? <Text accessibilityRole="alert" style={[styles.error, { margin: 18 }]}>{error}</Text> : null}
      {!listed.length && !loading && !searching ? <Text style={[styles.listMeta, { padding: 18 }]}>
        {words ? 'No conversation matches.' : 'No saved conversations yet.'}</Text> : null}
      {listed.map((chat) => {
        const day = conversationDay(chat.updated_at);
        return <View key={chat.id} style={chat.id === selectedId && styles.listItemOn}>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Pressable accessibilityRole="button" disabled={disabled} onPress={() => onOpen(chat)}
              style={[styles.flex, { paddingLeft: 18, paddingVertical: 10 }]}>
              <Text numberOfLines={1} style={{ fontSize: 15, color: palette.text }}>{chat.title}</Text>
              {day ? <Text style={styles.listMeta}>{day}</Text> : null}
            </Pressable>
            <IconButton glyph="⋯" label={`Rename or delete ${chat.title}`} disabled={disabled}
              onPress={() => { if (menu === chat.id) closeMenu(); else { closeMenu(); setMenu(chat.id); } }} />
          </View>
          {menu === chat.id ? <View style={{ paddingHorizontal: 18, paddingBottom: 12, gap: 8 }}>
            {renaming === chat.id ? <TextInput accessibilityLabel="Conversation title" style={styles.input} value={title}
              onChangeText={setTitle} autoFocus selectTextOnFocus maxLength={72} returnKeyType="done"
              editable={!working} onSubmitEditing={() => void rename(chat)} /> : null}
            <View style={[styles.row, { gap: 20 }]}>
              {renaming === chat.id ? <>
                {link('Save', () => void rename(chat))}
                {link('Cancel', closeMenu)}
              </> : <>
                {link('Rename', () => { setTitle(chat.title); setRenaming(chat.id); })}
                {link('Delete', () => remove(chat), true)}
              </>}
              {working ? <ActivityIndicator color={palette.muted} /> : null}
            </View>
            {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
          </View> : null}
        </View>;
      })}
    </ScrollView>
  </View>;
}
