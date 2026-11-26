import { fetch } from 'expo/fetch';
import { Agent, AgentUnavailable } from './Agent.tsx';
import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, AppState, KeyboardAvoidingView, Platform, Text, useColorScheme, View } from 'react-native';
import { daemonClient, DaemonError, UnreachableError, type Credential, type Host } from './api.ts';
import { Chat } from './Chat.tsx';
import { Generate } from './Generate.tsx';
import { defaultPreferences, DEVICE_ID, hostId, renameHost, selectHost, type Preferences } from './hosts.ts';
import { deviceBackend, deviceEngine } from './local/device.ts';
import { Models } from './Models.tsx';
import { checkLocalNetwork } from './localNetwork';
import { Pairing } from './Pairing.tsx';
import { hostLabel, Settings } from './Settings.tsx';
import {
  deleteHost, emptyIndex, loadHosts, loadPreferences, saveHost, saveIndex, savePreferences, type SavedHosts,
} from './storage.ts';
import { resolvePalette, ThemeContext, useTheme } from './theme.ts';
import { pinCertificate } from './tlsPinning';
import { Action, errorText, Screen, useStyles } from './ui.tsx';

type Access = { message: string; pairAgain: boolean };

export function App() {
  const system = useColorScheme();
  const [preferences, setPreferences] = useState<Preferences>(defaultPreferences);
  useEffect(() => { void loadPreferences().then(setPreferences); }, []);
  const palette = useMemo(() => resolvePalette(preferences.theme, system), [preferences.theme, system]);

  return <ThemeContext.Provider value={palette}>
    <Screen>
      <Shell preferences={preferences} onPreferences={(change) => {
        const next = { ...preferences, ...change };
        setPreferences(next);
        void savePreferences(next).catch(() => undefined);
      }} />
    </Screen>
  </ThemeContext.Provider>;
}

function Shell({ preferences, onPreferences }: {
  preferences: Preferences; onPreferences: (change: Partial<Preferences>) => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const [saved, setSaved] = useState<SavedHosts>({ index: emptyIndex, credentials: {} });
  const [starting, setStarting] = useState(true);
  const [view, setView] = useState<'chat' | 'settings' | 'pairing' | 'generate' | 'models'>('chat');
  const [generateOpened, setGenerateOpened] = useState(false);
  // Agent is mounted on first use and then kept, so a running task survives a visit to Chat.
  const [agent, setAgent] = useState<'unopened' | 'shown' | 'hidden'>('unopened');
  const [access, setAccess] = useState<Access | null>(null);
  const [storageError, setStorageError] = useState('');
  const [revision, setRevision] = useState(0);
  // Tagged with the credential it was verified for, so a host switch never renders Chat
  // (and sends the new host's key) before that host's identity is checked.
  const [verified, setVerified] = useState<{ credential: Credential; host: Host } | null>(null);
  const activeId = saved.index.active;
  const onDevice = activeId === DEVICE_ID;
  const credential: Credential | null = activeId ? saved.credentials[activeId] ?? null : null;
  const hostBackend = useMemo(() => credential ? daemonClient(fetch, credential) : null, [credential]);
  const [modelsRevision, setModelsRevision] = useState(0);

  // A model loaded for "This device" is not kept in memory while a host is in use.
  useEffect(() => {
    if (!onDevice && !starting) void deviceEngine.unload().catch(() => undefined);
  }, [onDevice, starting]);

  useEffect(() => {
    let active = true;
    void loadHosts().then(async (loaded) => {
      await Promise.all(Object.values(loaded.credentials)
        .map((stored) => pinCertificate(stored.origin, stored.certificateSha256)));
      if (active) setSaved(loaded);
    }).catch(() => {
      if (active) setStorageError('Could not read the saved hosts. Pair again.');
    }).finally(() => { if (active) setStarting(false); });
    return () => { active = false; };
  }, []);

  function reportAccess(error: unknown) {
    setAccess({
      message: errorText(error),
      pairAgain: error instanceof DaemonError && [401, 409].includes(error.status ?? 0),
    });
    setVerified(null);
  }

  useEffect(() => {
    setVerified(null);
    setAccess(null);
    if (!credential) return;
    let active = true;
    let checking = false;
    const verify = async () => {
      if (checking) return;
      checking = true;
      try {
        const result = await daemonClient(fetch, credential).verify();
        if (!active) return;
        setVerified({ credential, host: result });
        // A host migrated from a single-host build has no name until it is first reached.
        setSaved((current) => {
          const id = hostId(credential);
          if (current.index.hosts.find((entry) => entry.id === id)?.name !== '') return current;
          const index = renameHost(current.index, id, result.displayName);
          void saveIndex(index).catch(() => undefined);
          return { ...current, index };
        });
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

  async function update(change: () => Promise<SavedHosts>, failure: string) {
    setStorageError('');
    try {
      setSaved(await change());
    } catch {
      setStorageError(failure);
    }
  }

  const remove = (id: string) => update(async () => {
    const index = await deleteHost(saved.index, id);
    const { [id]: _removed, ...credentials } = saved.credentials;
    return { index, credentials };
  }, 'Could not remove the saved host. Check secure storage and try again.');

  const applyIndex = (index: SavedHosts['index']) => update(async () => {
    await saveIndex(index);
    return { ...saved, index };
  }, 'Could not save the host list.');

  if (starting) return <View style={styles.empty}><ActivityIndicator color={palette.accent} /></View>;

  if (view === 'pairing' || (!credential && !onDevice)) {
    return <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.container}>
      <View style={styles.header}><Text style={styles.brand}>Brazier</Text>
        <Text style={styles.subtitle}>Your own AI, on the go</Text></View>
      {storageError ? <Text accessibilityRole="alert" style={[styles.error, { marginHorizontal: 18 }]}>{storageError}</Text> : null}
      <Pairing onCancel={credential || onDevice ? () => setView('settings') : undefined} onPaired={async (issued, name) => {
        const index = await saveHost(saved.index, issued, name);
        setSaved({ index, credentials: { ...saved.credentials, [hostId(issued)]: issued } });
        setStorageError('');
        setView('chat');
      }} />
      {!credential && !onDevice ? <View style={{ paddingHorizontal: 18, paddingBottom: 12 }}>
        <Action label="Use models on this device instead" quiet
          onPress={() => { void applyIndex(selectHost(saved.index, DEVICE_ID)); setView('chat'); }} />
      </View> : null}
    </KeyboardAvoidingView>;
  }

  const host = credential && verified?.credential === credential ? verified.host : null;
  const name = credential ? hostLabel(saved.index, hostId(credential), credential.origin) : 'This device';

  const settings = <Settings saved={saved} theme={preferences.theme} error={storageError}
      onTheme={(theme) => onPreferences({ theme })}
      onSelect={(id) => { void applyIndex(selectHost(saved.index, id)); setView('chat'); }}
      onRename={(id, next) => void applyIndex(renameHost(saved.index, id, next))}
      onRemove={(id) => void remove(id)} onAdd={() => setView('pairing')} onModels={() => setView('models')}
      onClose={() => setView('chat')} />;
  const models = <Models onClose={() => setView(onDevice ? 'chat' : 'settings')}
    onChanged={() => setModelsRevision((value) => value + 1)} />;

  if (view === 'models' && !onDevice) return models;

  // Chat stays mounted behind Settings and Models so a draft or a running reply survives the visit.
  if (onDevice || !credential || !hostBackend) {
    return <>
      {view === 'settings' ? settings : view === 'models' ? models : null}
      <View style={view === 'chat' ? styles.flex : { display: 'none' }}>
        <Chat key={DEVICE_ID} backend={deviceBackend} credential={null} hostName="this device" modelsRevision={modelsRevision}
          disabledTools={[]} onDisabledTools={() => undefined} onUnauthorized={() => undefined}
          onOpenSettings={() => setView('settings')} onOpenModels={() => setView('models')}
          onOpenGenerate={() => undefined} onOpenAgent={() => undefined} />
      </View>
    </>;
  }

  if (view === 'settings' && (access || !host)) return settings;

  if (access) {
    return <View style={styles.panel}>
      <Text style={styles.brand}>Brazier</Text>
      <Text style={styles.heading}>{access.pairAgain ? 'Pair again' : `${name} is unavailable`}</Text>
      <Text style={styles.body}>{access.message}</Text>
      {storageError ? <Text accessibilityRole="alert" style={styles.error}>{storageError}</Text> : null}
      {access.pairAgain
        ? <Action label="Pair again" onPress={() => setView('pairing')} />
        : <Action label="Retry connection" onPress={() => setRevision((value) => value + 1)} />}
      <Action label="Hosts and settings" quiet onPress={() => setView('settings')} />
      <Action label="Forget this host" danger onPress={() => void remove(hostId(credential))} />
    </View>;
  }

  if (!host) {
    return <View style={styles.empty}><ActivityIndicator color={palette.accent} />
      <Text style={styles.emptyText}>Verifying {name}…</Text>
      <Action label="Hosts and settings" quiet onPress={() => setView('settings')} /></View>;
  }

  // Chat stays mounted behind Settings and Agent so a draft or a running reply survives the visit.
  const main = view !== 'chat' ? null : agent === 'shown' ? 'agent' : 'chat';
  return <>
    {view === 'settings' ? settings : null}
    {/* Stays mounted once opened so a running generation is not lost on returning to the chat. */}
    {generateOpened ? <View style={view === 'generate' ? styles.flex : { display: 'none' }}>
      <Generate key={hostId(credential)} credential={credential} hostName={name} visible={view === 'generate'}
        onUnauthorized={reportAccess} onClose={() => setView('chat')} />
    </View> : null}
    <View style={main === 'chat' ? styles.flex : { display: 'none' }}>
      <Chat key={hostId(credential)} backend={hostBackend} credential={credential} hostName={name}
        disabledTools={preferences.disabledTools} onDisabledTools={(disabledTools) => onPreferences({ disabledTools })}
        onUnauthorized={reportAccess} onOpenSettings={() => setView('settings')} onOpenAgent={() => setAgent('shown')}
        onOpenGenerate={() => { setGenerateOpened(true); setView('generate'); }} />
    </View>
    {agent === 'unopened' ? null : <View style={main === 'agent' ? styles.flex : { display: 'none' }}>
      {host.scopes.includes('agent')
        ? <Agent key={hostId(credential)} credential={credential} hostName={name} onUnauthorized={reportAccess}
          onOpenChat={() => setAgent('hidden')} onOpenSettings={() => setView('settings')} />
        : <AgentUnavailable hostName={name} onBack={() => setAgent('hidden')} />}
    </View>}
  </>;
}
