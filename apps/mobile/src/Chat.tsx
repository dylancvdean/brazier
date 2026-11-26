import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Modal, Platform, Pressable, SafeAreaView, ScrollView, Text, TextInput, View,
} from 'react-native';
import {
  chatBranch, chatContext, DaemonError, messageText, toolCallsOf,
  type Conversation, type Credential, type Message, type Model, type Tool, type ToolCall, type TranscriptMessage,
} from './api.ts';
import type { ChatBackend } from './local/engine.ts';
import { Markdown } from './Markdown.tsx';
import { generatedMediaMessage, mediaByMessage, toolCallMedia } from './media.ts';
import { MediaList } from './Media.tsx';
import { useTheme } from './theme.ts';
import { Action, errorText, IconButton, Sheet, useStyles } from './ui.tsx';

function ToolCalls({ calls }: { calls: readonly ToolCall[] }) {
  const styles = useStyles();
  const [open, setOpen] = useState<string | null>(null);
  return <>{calls.map((call) => <Pressable key={call.call_id} accessibilityRole="button" style={styles.card}
    onPress={() => setOpen(open === call.call_id ? null : call.call_id)}>
    <Text style={call.is_error ? styles.error : styles.listMeta}>
      {call.is_error ? '✕' : '⚙'} {call.name}{call.is_error ? ' failed' : ''}</Text>
    {open === call.call_id ? <>
      <Text selectable style={styles.mono}>{call.arguments}</Text>
      {call.output ? <Text selectable style={styles.mono}>{call.output.slice(0, 4000)}</Text> : null}
    </> : null}
  </Pressable>)}</>;
}

/**
 * `backend` is a paired host or this device. With `onOpenModels` set the chat is on this device: text only,
 * so the host-only entry points (tools, Generate, Agent) are not shown.
 */
export function Chat({
  backend: api, credential, hostName, modelsRevision = 0, disabledTools, onDisabledTools, onUnauthorized, onOpenSettings,
  onOpenGenerate, onOpenAgent, onOpenModels,
}: {
  backend: ChatBackend; credential: Credential | null; hostName: string; modelsRevision?: number;
  disabledTools: readonly string[];
  onDisabledTools: (names: string[]) => void; onUnauthorized: (error: unknown) => void; onOpenSettings: () => void;
  onOpenGenerate: () => void; onOpenAgent: () => void; onOpenModels?: () => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const local = !!onOpenModels;
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState('');
  const [showModels, setShowModels] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [draft, setDraft] = useState('');
  const [streamed, setStreamed] = useState('');
  const [tools, setTools] = useState<Tool[]>([]);
  const [showTools, setShowTools] = useState(false);
  const [running, setRunning] = useState<ToolCall[]>([]);
  const enabledTools = tools.filter(({ name }) => !disabledTools.includes(name));
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
      // A host too old to list tools still chats; it just offers none.
      const [available, history, offered] = await Promise.all([
        api.models(), api.conversations(), api.tools().catch((cause) => {
          if (cause instanceof DaemonError && cause.status === 401) throw cause;
          return [];
        }),
      ]);
      setTools(offered);
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

  // Models added or deleted on the Models screen while this chat stays mounted behind it.
  useEffect(() => {
    if (!modelsRevision) return;
    void api.models().then((available) => {
      setModels(available);
      setModel((current) => available.some(({ id }) => id === current) ? current : '');
    }).catch(report);
  }, [modelsRevision]);

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
      const transcript: TranscriptMessage[] = [];
      const calls: ToolCall[] = [];
      const answer = await api.chat(model, chatContext(history), turn.signal, {
        onToken: (token) => setStreamed((text) => text + token),
        onToolCall: (call) => { calls.push(call); setRunning([...calls]); },
        onTranscript: (message) => {
          transcript.push(message);
          if (message.role === 'assistant') setStreamed('');
        },
      }, enabledTools.map(({ name }) => name));
      if (turn.signal.aborted) return;
      let parent = user.id;
      const saved = [...history];
      for (const entry of transcript) {
        const step = await api.createMessage(chat.id, {
          parent_id: parent, role: entry.role, content: entry.content ?? '',
          tool_calls: entry.tool_calls ?? null, tool_call_id: entry.tool_call_id ?? null,
          ...(entry.role === 'assistant' ? { model } : {}),
        });
        saved.push(step);
        parent = step.id;
      }
      if (answer.trim()) {
        saved.push(await api.createMessage(chat.id, { parent_id: parent, role: 'assistant', content: answer, model }));
      }
      const generated = generatedMediaMessage(calls);
      if (generated) saved.push(await api.createMessage(chat.id, { parent_id: saved.at(-1)!.id, ...generated, model }));
      setMessages(saved);
    } catch (cause) {
      if (!turn.signal.aborted) report(cause);
    } finally {
      if (!stored) setDraft(prompt);
      if (turn.signal.aborted) setError('Generation stopped. Unfinished replies are not saved.');
      currentTurn.current = null;
      setStreamed('');
      setRunning([]);
      setBusy('');
    }
  }

  const modelName = (id: string) => models.find((entry) => entry.id === id)?.name ?? id;
  const visible = messages.filter(({ role }) => role === 'user' || role === 'assistant');
  const savedCalls = useMemo(() => toolCallsOf(messages), [messages]);
  const savedMedia = useMemo(() => mediaByMessage(messages), [messages]);
  const toggleTool = (name: string) => onDisabledTools(disabledTools.includes(name)
    ? disabledTools.filter((other) => other !== name) : [...disabledTools, name]);
  const toolGroups = [...new Set(tools.map((tool) => tool.server_name ?? 'Built in'))];

  return <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="☰" label="Conversations" onPress={() => setShowDrawer(true)} />
      <Text numberOfLines={1} style={styles.topTitle}>{selected?.title ?? 'New chat'}</Text>
      <Pressable accessibilityRole="button" accessibilityLabel="Choose model" disabled={!!busy}
        style={[styles.chip, !!busy && styles.disabled]} onPress={() => setShowModels(true)}>
        <Text numberOfLines={1} style={styles.chipText}>{modelName(model) || 'Choose model'} ▾</Text>
      </Pressable>
      <IconButton glyph="＋" label="New chat" disabled={!!busy} onPress={newChat} />
    </View>
    {error ? <Text accessibilityRole="alert" style={[styles.error, { margin: 10 }]}>{error}</Text> : null}
    {visible.length === 0 && !streamed && !running.length ? <View style={styles.empty}>
      <Text style={styles.brand}>Brazier</Text>
      <Text style={styles.emptyText}>{model ? (local ? 'Replies are generated on this device.' : `Connected to ${hostName}.`)
        : local && !models.length && !busy ? 'Add a model to chat on this device.' : 'Choose a chat model to start.'}</Text>
      {!model && !busy ? (local && !models.length
        ? <Action label="Add a model" onPress={onOpenModels!} />
        : <Action label="Choose model" onPress={() => setShowModels(true)} />) : null}
    </View> : <ScrollView style={styles.flex} contentContainerStyle={styles.messages} ref={scroll}
      onContentSizeChange={() => scroll.current?.scrollToEnd({ animated: false })} keyboardShouldPersistTaps="handled">
      {visible.map((message) => {
        const calls = savedCalls.get(message.id);
        const media = savedMedia.media.get(message.id);
        if (!media && savedMedia.moved.has(message.id)) return null;
        const text = messageText(message.content) || (calls || media ? '' : 'Non-text message');
        if (message.role === 'user') {
          return <View key={message.id} style={{ gap: 6 }}>
            {text ? <View style={styles.userBubble}><Text selectable style={styles.body}>{text}</Text></View> : null}
            {media && credential ? <MediaList credential={credential} blobs={media} /> : null}
          </View>;
        }
        return <View key={message.id} style={styles.assistant}>
          {text ? <Markdown text={text} /> : null}
          {calls ? <ToolCalls calls={calls} /> : null}
          {media && credential ? <MediaList credential={credential} blobs={media} /> : null}
        </View>;
      })}
      {running.length ? <View style={styles.assistant}>
        <ToolCalls calls={running} />
        {credential ? <MediaList credential={credential} blobs={toolCallMedia(running)} /> : null}
      </View> : null}
      {streamed ? <View style={styles.assistant}><Markdown text={streamed} /></View> : null}
    </ScrollView>}
    {busy ? <Text accessibilityRole="progressbar" style={styles.status}>{busy}</Text> : null}
    <View style={styles.composer}>
      {tools.length ? <IconButton glyph={enabledTools.length ? '⚙' : '⊘'}
        label={`Tools, ${enabledTools.length} of ${tools.length} on`} onPress={() => setShowTools(true)} /> : null}
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
          {local ? 'No models on this device yet.' : 'No chat models on this host. Add one on the host, then refresh.'}</Text> : null}
        {models.map(({ id }) => <Pressable key={id} accessibilityRole="button"
          style={[styles.listItem, id === model && styles.listItemOn]} onPress={() => void chooseModel(id)}>
          <Text style={styles.check}>{id === model ? '✓' : ''}</Text>
          <Text style={styles.listText}>{modelName(id)}</Text>
        </Pressable>)}
        <View style={{ padding: 18 }}>{local
          ? <Action label="Manage models" quiet onPress={() => { setShowModels(false); onOpenModels!(); }} />
          : <Action label="Refresh" quiet disabled={!!busy} onPress={() => void refresh()} />}</View>
      </ScrollView>
    </Sheet>

    <Sheet title={`Tools · ${enabledTools.length} of ${tools.length} on`} visible={showTools} onClose={() => setShowTools(false)}>
      <ScrollView>
        <View style={[styles.row, { paddingHorizontal: 18, paddingBottom: 8 }]}>
          <Action label="All" quiet onPress={() => onDisabledTools([])} />
          <Action label="None" quiet onPress={() => onDisabledTools(tools.map(({ name }) => name))} />
        </View>
        {toolGroups.map((group) => <View key={group}>
          <Text style={[styles.section, { paddingHorizontal: 18 }]}>{group}</Text>
          {tools.filter((tool) => (tool.server_name ?? 'Built in') === group).map((tool) => {
            const on = !disabledTools.includes(tool.name);
            return <Pressable key={tool.name} accessibilityRole="switch" accessibilityState={{ checked: on }}
              style={styles.listItem} onPress={() => toggleTool(tool.name)}>
              <Text style={styles.check}>{on ? '✓' : ''}</Text>
              <View style={styles.flex}>
                <Text style={styles.listText}>{tool.title || tool.name}{tool.network ? '  🌐' : ''}</Text>
                {tool.description ? <Text numberOfLines={2} style={styles.listMeta}>{tool.description}</Text> : null}
              </View>
            </Pressable>;
          })}
        </View>)}
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
          {local ? <Pressable accessibilityRole="button" style={styles.listItem} onPress={() => {
            setShowDrawer(false); onOpenModels!();
          }}>
            <Text style={styles.iconText}>▤</Text><Text style={styles.listText}>Models</Text>
          </Pressable> : <>
            <Pressable accessibilityRole="button" style={styles.listItem} onPress={() => {
              setShowDrawer(false); onOpenGenerate();
            }}>
              <Text style={styles.iconText}>✦</Text><Text style={styles.listText}>Generate</Text>
            </Pressable>
            <Pressable accessibilityRole="button" style={styles.listItem} onPress={() => {
              setShowDrawer(false); onOpenAgent();
            }}>
              <Text style={styles.iconText}>🛠</Text><Text style={styles.listText}>Agent</Text>
            </Pressable>
          </>}
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
