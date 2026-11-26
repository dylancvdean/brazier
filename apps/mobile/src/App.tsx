import { CameraView, useCameraPermissions } from 'expo-camera';
import * as SecureStore from 'expo-secure-store';
import { fetch } from 'expo/fetch';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, AppState, KeyboardAvoidingView, Platform, Pressable, SafeAreaView,
  ScrollView, StatusBar, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { parsePairingInvite } from '../../../packages/pairing/src/invite.ts';
import { checkLocalNetwork } from './localNetwork';
import { pinCertificate } from './tlsPinning';
import {
  chatBranch, chatContext, claimInvite, daemonClient, DaemonError, messageText, pairingOrigin,
  parseCredential, previewInvite, type Conversation, type Credential, type Host, type InviteInput,
  type Message, type Model, type Preview, UnreachableError,
} from './api.ts';

const credentialKey = 'brazier.host.v1';
const androidEmulator = __DEV__ && Platform.OS === 'android';
const options: SecureStore.SecureStoreOptions = { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY };
const emptyInvite = { origin: '', pairingId: '', code: '' };
const errorText = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong.';

function Action({ label, onPress, disabled, quiet = false }: {
  label: string; onPress: () => void; disabled?: boolean; quiet?: boolean;
}) {
  return <Pressable accessibilityRole="button" disabled={disabled} onPress={onPress}
    style={[styles.button, quiet && styles.quiet, disabled && styles.disabled]}>
    <Text style={quiet ? styles.quietText : styles.buttonText}>{label}</Text>
  </Pressable>;
}

export function App() {
  const [credential, setCredential] = useState<Credential | null>(null);
  const [starting, setStarting] = useState(true);
  const [access, setAccess] = useState<{ message: string; pairAgain: boolean } | null>(null);
  const [revision, setRevision] = useState(0);
  const [host, setHost] = useState<Host | null>(null);

  useEffect(() => {
    let active = true;
    void SecureStore.getItemAsync(credentialKey, options).then((saved) => {
      if (!active || !saved) return;
      const stored = parseCredential(saved, androidEmulator);
      return pinCertificate(stored.origin, stored.certificateSha256).then(() => { if (active) setCredential(stored); });
    }).catch(() => {
      if (active) setAccess({ message: 'Could not read the saved host credential.', pairAgain: true });
    }).finally(() => { if (active) setStarting(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!credential) return;
    let active = true;
    let checking = false;
    const verify = async () => {
      if (checking) return;
      checking = true;
      try {
        const result = await daemonClient(fetch, credential).verify();
        if (active) setHost(result);
      } catch (error) {
        const reason = error instanceof UnreachableError
          ? await checkLocalNetwork(credential.origin, 5_000).then(() => error, (cause: unknown) => cause)
          : error;
        if (active) reportAccess(reason);
      } finally {
        checking = false;
      }
    };
    void verify();
    const listener = AppState.addEventListener('change', (state) => { if (state === 'active') void verify(); });
    return () => { active = false; listener.remove(); };
  }, [credential, revision]);

  function reportAccess(error: unknown) {
    setAccess({
      message: errorText(error),
      pairAgain: error instanceof DaemonError && [401, 409].includes(error.status ?? 0),
    });
    setHost(null);
  }

  async function forget() {
    try {
      await SecureStore.deleteItemAsync(credentialKey, options);
      setCredential(null);
      setHost(null);
      setAccess(null);
    } catch {
      setAccess({ message: 'Could not remove the saved host credential. Check secure storage and try again.', pairAgain: true });
    }
  }

  return <SafeAreaView style={styles.safe}>
    <StatusBar barStyle="dark-content" />
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.container}>
      <View style={styles.header}><Text style={styles.brand}>Brazier</Text>
        <Text style={styles.subtitle}>{host?.displayName ?? 'Your own AI, on the go'}</Text></View>
      {starting ? <ActivityIndicator /> : access ? <View style={styles.panel}>
        <Text style={styles.heading}>{access.pairAgain ? 'Pair again' : 'Host unavailable'}</Text>
        <Text style={styles.body}>{access.message}</Text>
        {!access.pairAgain && <Action label="Retry connection" onPress={() => {
          setAccess(null);
          setRevision((value) => value + 1);
        }} />}
        <Action label={access.pairAgain ? 'Pair again' : 'Forget this host'} onPress={() => void forget()} quiet={!access.pairAgain} />
      </View> : credential ? host ?
        <Workspace credential={credential} onUnauthorized={reportAccess} onForget={() => void forget()} />
        : <View style={styles.panel}><ActivityIndicator /><Text style={styles.body}>Verifying host identity…</Text></View>
        : <Pairing onPaired={setCredential} />}
    </KeyboardAvoidingView>
  </SafeAreaView>;
}

function inviteCertificate(invite: object): { certificateSha256?: string } {
  return 'certificateSha256' in invite && typeof invite.certificateSha256 === 'string'
    ? { certificateSha256: invite.certificateSha256 } : {};
}

function Pairing({ onPaired }: { onPaired: (credential: Credential) => void }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [scanning, setScanning] = useState(false);
  const scanLock = useRef(false);
  const [inviteText, setInviteText] = useState('');
  const [input, setInput] = useState<InviteInput>(emptyInvite);
  const [review, setReview] = useState<{ input: InviteInput; preview: Preview } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function parse(text: string) {
    const invite = parsePairingInvite(text);
    if (!invite) throw new Error('This is not a supported Brazier invite. Use the host’s QR or Copy invite.');
    if (invite.version === 1 && invite.expiresAt <= Date.now() / 1000) {
      throw new Error('This invite expired. Ask the host for a fresh invite.');
    }
    return invite;
  }

  async function inspect(candidate: InviteInput) {
    setBusy(true);
    setError('');
    try {
      const target = { ...candidate, origin: pairingOrigin(candidate.origin, androidEmulator),
        pairingId: candidate.pairingId.trim(), code: candidate.code.trim() };
      if (!target.pairingId || !target.code) throw new Error('Enter the invitation ID and code.');
      // The check connects directly, so a host behind the phone's proxy can
      // fail it and still be reachable: only report it if the request fails too.
      const unreachable = await checkLocalNetwork(target.origin).then(() => null, (cause: unknown) => cause);
      await pinCertificate(target.origin, target.certificateSha256);
      const preview = await previewInvite(fetch, target).catch((cause: unknown) => {
        throw unreachable && cause instanceof UnreachableError ? unreachable : cause;
      });
      setReview({ input: target, preview });
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setBusy(false);
    }
  }

  async function scan() {
    setError('');
    if (!permission?.granted && !(await requestPermission()).granted) {
      setError('Camera access is off. You can still paste an invite below.');
      return;
    }
    scanLock.current = false;
    setScanning(true);
  }

  async function claim() {
    if (!review) return;
    setBusy(true);
    setError('');
    try {
      const issued = await claimInvite(fetch, review.input, review.preview);
      try {
        await SecureStore.setItemAsync(credentialKey, JSON.stringify(issued), options);
      } catch {
        throw new Error('Invite claimed, but the credential was not saved. Ask the host to revoke this device and issue a fresh invite.');
      }
      onPaired(issued);
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setBusy(false);
      setReview(null);
      setInviteText('');
      setInput(emptyInvite);
    }
  }

  return <ScrollView contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled">
    <Text style={styles.heading}>{review ? 'Review pairing' : 'Pair with your host'}</Text>
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {review ? <>
      <Text style={styles.body}>Host: {review.preview.daemon.display_name}</Text>
      <Text style={styles.body}>Origin: {review.input.origin}</Text>
      <Text style={styles.body}>Device: {review.preview.pairing.client_name}</Text>
      <Text style={styles.body}>Access: {review.preview.pairing.scopes.join(', ')}</Text>
      <Action label="Confirm and pair" disabled={busy} onPress={() => void claim()} />
      <Action label="Cancel" disabled={busy} quiet onPress={() => {
        setReview(null); setInput(emptyInvite); setInviteText('');
      }} />
    </> : scanning ? <>
      <CameraView barcodeScannerSettings={{ barcodeTypes: ['qr'] }} style={styles.camera}
        onBarcodeScanned={({ data }) => {
          if (scanLock.current) return;
          scanLock.current = true;
          setScanning(false);
          try {
            const invite = parse(data);
            const candidate = { origin: invite.origin ?? '', pairingId: invite.pairingId, code: invite.code,
              ...inviteCertificate(invite) };
            setInput(candidate);
            if (candidate.origin) void inspect(candidate);
          } catch (cause) {
            setError(errorText(cause));
          }
        }} />
      <Action label="Cancel scan" quiet onPress={() => setScanning(false)} />
    </> : <>
      <Text style={styles.body}>Open Manage → Connections on your Brazier host to create an inference invite.</Text>
      <Action label="Scan QR code" disabled={busy} onPress={() => void scan()} />
      <TextInput accessibilityLabel="Pairing invite" placeholder="Paste pairing invite" value={inviteText}
        style={[styles.input, styles.invite]} editable={!busy} onChangeText={setInviteText} multiline
        autoCapitalize="none" autoCorrect={false} />
      <Action label="Review invite" disabled={busy || !inviteText.trim()} onPress={() => {
        try {
          const invite = parse(inviteText);
          const candidate = { origin: invite.origin ?? input.origin, pairingId: invite.pairingId, code: invite.code,
            ...inviteCertificate(invite) };
          setInput(candidate);
          if (candidate.origin) void inspect(candidate);
          else setError('Enter the host address below to review this invite.');
        } catch (cause) {
          setError(errorText(cause));
        }
      }} />
      <Text style={styles.label}>Or enter the invite details</Text>
      {(['origin', 'pairingId', 'code'] as const).map((field) => {
        const label = { origin: 'Host address', pairingId: 'Pairing ID', code: 'Pairing code' }[field];
        return <TextInput key={field} accessibilityLabel={label} placeholder={label} style={styles.input}
          editable={!busy} value={input[field]} onChangeText={(value) => setInput({ ...input, [field]: value })}
          autoCapitalize="none" autoCorrect={false} secureTextEntry={field === 'code'}
          keyboardType={field === 'origin' ? 'url' : 'default'} />;
      })}
      <Action label="Review details" disabled={busy || !input.origin || !input.pairingId || !input.code}
        quiet onPress={() => void inspect(input)} />
    </>}
    {busy ? <ActivityIndicator /> : null}
  </ScrollView>;
}

function Workspace({ credential, onUnauthorized, onForget }: {
  credential: Credential; onUnauthorized: (error: unknown) => void; onForget: () => void;
}) {
  const api = daemonClient(fetch, credential);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState('');
  const [showModels, setShowModels] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [draft, setDraft] = useState('');
  const [streamed, setStreamed] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const currentTurn = useRef<AbortController | null>(null);
  const scroll = useRef<ScrollView>(null);

  function report(error: unknown) {
    if (error instanceof DaemonError && error.status === 401) onUnauthorized(error);
    else setError(errorText(error));
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
      if (model && !available.some(({ id }) => id === model)) {
        setModel('');
        setShowModels(true);
      }
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
    setBusy('Preparing chat model…');
    setError('');
    try {
      await api.prepare(id);
      setModel(id);
      setShowModels(false);
    } catch (cause) {
      report(cause);
    } finally {
      setBusy('');
    }
  }

  async function openConversation(conversation: Conversation) {
    setBusy('Loading conversation…');
    setError('');
    try {
      setMessages(chatBranch(await api.messages(conversation.id)));
      setSelected(conversation);
      setShowHistory(false);
    } catch (cause) {
      report(cause);
    } finally {
      setBusy('');
    }
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

  return <View style={styles.workspace}>
    <View style={styles.row}>
      <Action label="Models" quiet disabled={!!busy} onPress={() => {
        setShowModels(!showModels); setShowHistory(false);
      }} />
      <Action label="History" quiet disabled={!!busy} onPress={() => {
        setShowHistory(!showHistory); setShowModels(false);
      }} />
      <Action label="New chat" quiet disabled={!!busy} onPress={() => {
        setSelected(null); setMessages([]); setDraft(''); setError(''); setShowHistory(false);
      }} />
    </View>
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {busy ? <Text accessibilityRole="progressbar" style={styles.label}>{busy}</Text> : null}
    <ScrollView style={styles.content} contentContainerStyle={styles.panel} ref={scroll}
      onContentSizeChange={() => { if (!showModels && !showHistory) scroll.current?.scrollToEnd({ animated: false }); }}
      keyboardShouldPersistTaps="handled">
      {showModels ? <>
        <Text style={styles.heading}>Chat models</Text>
        {models.length === 0 && !busy && <Text style={styles.body}>No chat models on this host. Add one on the host, then refresh.</Text>}
        {models.map(({ id }) => <Action key={id} label={id} quiet={id !== model}
          disabled={!!busy} onPress={() => void chooseModel(id)} />)}
      </> : showHistory ? <>
        <Text style={styles.heading}>Conversation history</Text>
        {conversations.length === 0 && <Text style={styles.body}>No saved conversations yet.</Text>}
        {conversations.map((chat) => <Action key={chat.id} label={chat.title} quiet
          disabled={!!busy} onPress={() => void openConversation(chat)} />)}
      </> : <>
        <Text style={styles.heading}>{selected?.title ?? 'New conversation'}</Text>
        <Text style={styles.label}>{model || 'Choose a chat model to send a message.'}</Text>
        {messages.map((message) => {
          const text = messageText(message.content);
          if (!['user', 'assistant'].includes(message.role)) return null;
          return <View key={message.id} style={[styles.message, message.role === 'user' && styles.userMessage]}>
            <Text style={styles.label}>{message.role === 'user' ? 'You' : 'Assistant'}</Text>
            <Text selectable style={styles.body}>{text || 'Non-text message'}</Text>
          </View>;
        })}
        {streamed ? <View style={styles.message}><Text style={styles.label}>Assistant</Text>
          <Text selectable style={styles.body}>{streamed}</Text></View> : null}
      </>}
      {(showModels || showHistory) && <View style={styles.row}>
        <Action label="Refresh" quiet disabled={!!busy} onPress={() => void refresh()} />
        <Action label="Forget host" quiet disabled={!!busy} onPress={onForget} />
      </View>}
    </ScrollView>
    {!showModels && !showHistory && <View style={styles.composer}>
      <TextInput accessibilityLabel="Message" placeholder="Message Brazier" style={[styles.input, styles.composeInput]}
        value={draft} onChangeText={setDraft} editable={!busy} multiline />
      {busy === 'Generating…' ? <Action label="Stop" onPress={() => currentTurn.current?.abort()} />
        : <Action label="Send" disabled={!draft.trim() || !model || !!busy} onPress={() => void send()} />}
    </View>}
  </View>;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#f8f6f1', paddingTop: Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) : 0 },
  container: { flex: 1, paddingBottom: Platform.OS === 'android' ? 26 : 0 },
  header: { paddingHorizontal: 22, paddingVertical: 16, borderBottomWidth: 1, borderColor: '#e6dfd3' },
  brand: { fontSize: 27, fontWeight: '800', color: '#692c1a' },
  subtitle: { fontSize: 14, color: '#685b51', marginTop: 2 },
  panel: { padding: 18, gap: 12 },
  heading: { fontSize: 23, fontWeight: '700', color: '#2e201b' },
  body: { fontSize: 16, lineHeight: 23, color: '#2e201b' },
  label: { fontSize: 14, fontWeight: '600', color: '#685b51' },
  error: { padding: 10, color: '#831c0d', backgroundColor: '#fae7df', borderRadius: 8 },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, padding: 8 },
  button: { borderRadius: 9, backgroundColor: '#81391f', paddingHorizontal: 12, paddingVertical: 11 },
  buttonText: { color: '#fff', fontWeight: '700' },
  quiet: { backgroundColor: '#eee6db' },
  quietText: { color: '#572d20', fontWeight: '700' },
  disabled: { opacity: 0.45 },
  input: { borderWidth: 1, borderColor: '#cbbdb0', borderRadius: 9, padding: 11, color: '#2e201b', backgroundColor: '#fff', fontSize: 16 },
  invite: { minHeight: 66, textAlignVertical: 'top' },
  camera: { height: 330, overflow: 'hidden', borderRadius: 12 },
  workspace: { flex: 1, paddingHorizontal: 12, gap: 8 },
  content: { flex: 1 },
  composer: { flexDirection: 'row', alignItems: 'flex-end', gap: 8, padding: 5 },
  composeInput: { flex: 1, maxHeight: 120 },
  message: { borderRadius: 10, padding: 11, backgroundColor: '#fff', gap: 4 },
  userMessage: { backgroundColor: '#f0e5d6' },
});
