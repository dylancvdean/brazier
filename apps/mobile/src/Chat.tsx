import { fetch } from 'expo/fetch';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Modal, Platform, Pressable, SafeAreaView, ScrollView, Text, TextInput, View,
} from 'react-native';
import {
  chatBranch, chatContext, daemonClient, DaemonError, messageText,
  type Conversation, type Credential, type Message, type Model,
} from './api.ts';
import { Markdown } from './Markdown.tsx';
import { useTheme } from './theme.ts';
import { Action, errorText, IconButton, Sheet, useStyles } from './ui.tsx';

export function Chat({ credential, hostName, onUnauthorized, onOpenSettings }: {
  credential: Credential; hostName: string; onUnauthorized: (error: unknown) => void; onOpenSettings: () => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const api = useMemo(() => daemonClient(fetch, credential), [credential]);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState('');
  const [showModels, setShowModels] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [draft, setDraft] = useState('');
  const [streamed, setStreamed] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const currentTurn = useRef<AbortController | null>(null);
  const scroll = useRef<ScrollView>(null);
  const generating = busy === 'Generating…';

  function report(cause: unknown) {
    if (cause instanceof DaemonError && cause.status === 401) onUnauthorized(cause);
    else setError(errorText(cause));
  }

  async function refresh() {
    setBusy('Refreshing…');
    setError('');
    try {
      const [available, history] = await Promise.all([api.models(), api.conversations()]);
      setModels(available);
      setConversations(history);
      if (selected && !history.some(({ id }) => id === selected.id)) {
        setSelected(null);
        setMessages([]);
      }
      if (model && !available.some(({ id }) => id === model)) setModel('');
    } catch (cause) {
      report(cause);
    } finally {
      setBusy('');
    }
  }

  useEffect(() => {
    void refresh();
    return () => currentTurn.current?.abort();
  }, []);

  async function chooseModel(id: string) {
    setShowModels(false);
    setBusy('Preparing chat model…');
    setError('');
    try {
      await api.prepare(id);
      setModel(id);
    } catch (cause) {
      report(cause);
    } finally {
      setBusy('');
    }
  }

  async function openConversation(conversation: Conversation) {
    setShowDrawer(false);
    setBusy('Loading conversation…');
    setError('');
    try {
      setMessages(chatBranch(await api.messages(conversation.id)));
      setSelected(conversation);
    } catch (cause) {
      report(cause);
    } finally {
      setBusy('');
    }
  }

  function newChat() {
    setSelected(null); setMessages([]); setDraft(''); setError(''); setShowDrawer(false);
  }

  async function send() {
    const prompt = draft.trim();
    if (!prompt || !model || busy || currentTurn.current) return;
    const turn = new AbortController();
    currentTurn.current = turn;
    setBusy('Generating…');
    setError('');
    setDraft('');
    let stored = false;
    try {
      const chat = selected ?? await api.createConversation(prompt);
      if (!selected) {
        setSelected(chat);
        setConversations((previous) => [chat, ...previous]);
      }
      if (turn.signal.aborted) return;
      const { user, history } = await api.appendUserMessage(chat.id, selected ? messages : [], prompt);
      stored = true;
      setMessages(history);
      if (turn.signal.aborted) return;
      const answer = await api.chat(model, chatContext(history), turn.signal, (token) => setStreamed((text) => text + token));
      if (turn.signal.aborted) return;
      if (answer.trim()) {
        const reply = await api.createMessage(chat.id, { parent_id: user.id, role: 'assistant', content: answer, model });
        setMessages([...history, reply]);
      }
    } catch (cause) {
      if (!turn.signal.aborted) report(cause);
    } finally {
      if (!stored) setDraft(prompt);
      if (turn.signal.aborted) setError('Generation stopped. Unfinished replies are not saved.');
      currentTurn.current = null;
      setStreamed('');
      setBusy('');
    }
  }

  const visible = messages.filter(({ role }) => role === 'user' || role === 'assistant');

  return <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="☰" label="Conversations" onPress={() => setShowDrawer(true)} />
      <Text numberOfLines={1} style={styles.topTitle}>{selected?.title ?? 'New chat'}</Text>
      <Pressable accessibilityRole="button" accessibilityLabel="Choose model" disabled={!!busy}
        style={[styles.chip, !!busy && styles.disabled]} onPress={() => setShowModels(true)}>
        <Text numberOfLines={1} style={styles.chipText}>{model || 'Choose model'} ▾</Text>
      </Pressable>
      <IconButton glyph="＋" label="New chat" disabled={!!busy} onPress={newChat} />
    </View>
    {error ? <Text accessibilityRole="alert" style={[styles.error, { margin: 10 }]}>{error}</Text> : null}
    {visible.length === 0 && !streamed ? <View style={styles.empty}>
      <Text style={styles.brand}>Brazier</Text>
      <Text style={styles.emptyText}>{model ? `Connected to ${hostName}.` : 'Choose a chat model to start.'}</Text>
      {!model && !busy ? <Action label="Choose model" onPress={() => setShowModels(true)} /> : null}
    </View> : <ScrollView style={styles.flex} contentContainerStyle={styles.messages} ref={scroll}
      onContentSizeChange={() => scroll.current?.scrollToEnd({ animated: false })} keyboardShouldPersistTaps="handled">
      {visible.map((message) => {
        const text = messageText(message.content) || 'Non-text message';
        return message.role === 'user'
          ? <View key={message.id} style={styles.userBubble}><Text selectable style={styles.body}>{text}</Text></View>
          : <View key={message.id} style={styles.assistant}><Markdown text={text} /></View>;
      })}
      {streamed ? <View style={styles.assistant}><Markdown text={streamed} /></View> : null}
    </ScrollView>}
    {busy ? <Text accessibilityRole="progressbar" style={styles.status}>{busy}</Text> : null}
    <View style={styles.composer}>
      <TextInput accessibilityLabel="Message" placeholder={`Message ${hostName}`} placeholderTextColor={palette.muted}
        style={styles.composeInput} value={draft} onChangeText={setDraft} editable={!busy} multiline />
      {generating
        ? <Pressable accessibilityRole="button" accessibilityLabel="Stop" style={styles.send}
          onPress={() => currentTurn.current?.abort()}><Text style={styles.sendText}>■</Text></Pressable>
        : <Pressable accessibilityRole="button" accessibilityLabel="Send" disabled={!draft.trim() || !model || !!busy}
          style={[styles.send, (!draft.trim() || !model || !!busy) && styles.disabled]} onPress={() => void send()}>
          <Text style={styles.sendText}>↑</Text></Pressable>}
    </View>

    <Sheet title="Chat models" visible={showModels} onClose={() => setShowModels(false)}>
      <ScrollView>
        {models.length === 0 ? <Text style={[styles.emptyText, { padding: 18 }]}>
          No chat models on this host. Add one on the host, then refresh.</Text> : null}
        {models.map(({ id }) => <Pressable key={id} accessibilityRole="button"
          style={[styles.listItem, id === model && styles.listItemOn]} onPress={() => void chooseModel(id)}>
          <Text style={styles.check}>{id === model ? '✓' : ''}</Text>
          <Text style={styles.listText}>{id}</Text>
        </Pressable>)}
        <View style={{ padding: 18 }}><Action label="Refresh" quiet disabled={!!busy} onPress={() => void refresh()} /></View>
      </ScrollView>
    </Sheet>

    <Modal transparent visible={showDrawer} animationType="fade" onRequestClose={() => setShowDrawer(false)}>
      <View style={styles.backdrop}>
        <SafeAreaView style={styles.drawer}>
          <View style={styles.header}>
            <Text style={styles.brand}>Brazier</Text>
            <Text numberOfLines={1} style={styles.subtitle}>{hostName}</Text>
          </View>
          <View style={{ paddingHorizontal: 18 }}><Action label="New chat" disabled={!!busy} onPress={newChat} /></View>
          <Text style={[styles.section, { paddingHorizontal: 18 }]}>Conversations</Text>
          <ScrollView style={styles.flex}>
            {busy === 'Refreshing…' ? <ActivityIndicator color={palette.accent} /> : null}
            {conversations.length === 0 && !busy ? <Text style={[styles.listMeta, { padding: 18 }]}>
              No saved conversations yet.</Text> : null}
            {conversations.map((chat) => <Pressable key={chat.id} accessibilityRole="button" disabled={!!busy}
              style={[styles.listItem, chat.id === selected?.id && styles.listItemOn]}
              onPress={() => void openConversation(chat)}>
              <Text numberOfLines={1} style={styles.listText}>{chat.title}</Text>
            </Pressable>)}
          </ScrollView>
          <Pressable accessibilityRole="button" style={styles.listItem} onPress={() => {
            setShowDrawer(false); onOpenSettings();
          }}>
            <Text style={styles.iconText}>⚙</Text><Text style={styles.listText}>Settings</Text>
          </Pressable>
        </SafeAreaView>
        <Pressable style={styles.flex} accessibilityLabel="Close conversations" onPress={() => setShowDrawer(false)} />
      </View>
    </Modal>
  </KeyboardAvoidingView>;
}
