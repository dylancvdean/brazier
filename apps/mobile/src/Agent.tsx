import { fetch } from 'expo/fetch';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator, Alert, Keyboard, KeyboardAvoidingView, Modal, Platform, Pressable, SafeAreaView, ScrollView, Text, TextInput, View,
} from 'react-native';
import { AgentController } from './agent/controller.ts';
import { agentApi } from './agent/rest.ts';
import {
  agentErrorText, applyEvent, argsPreview, emptyView, locationLine, PERMISSION_LABELS, permissionModesFor,
  selectableTools, summaryLine, transcriptRows, viewFromDetail,
  type AgentApproval, type AgentCapabilities, type AgentPermissionMode, type AgentSessionSummary, type AgentTool,
  type AgentView, type TimelineEntry,
} from './agent/state.ts';
import { daemonClient, DaemonError, type Credential, type Model } from './api.ts';
import { Markdown } from './Markdown.tsx';
import { useTheme } from './theme.ts';
import { Action, dragDismiss, errorText, IconButton, Sheet, useStyles } from './ui.tsx';

type Draft = { model: string; workspace: string; mode: AgentPermissionMode; tools: string[] | null };

const folderName = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path;

/** Shown in place of Agent when the paired key cannot use it. */
export function AgentUnavailable({ hostName, onBack }: { hostName: string; onBack: () => void }) {
  const styles = useStyles();
  return <View style={styles.panel}>
    <Text style={styles.brand}>Agent</Text>
    <Text style={styles.heading}>This device cannot run agent tasks on {hostName}</Text>
    <Text style={styles.body}>
      Agent tasks read and change files and run commands on the host, so the host has to allow them for each device.
      This device was paired without that permission.</Text>
    <Text style={styles.body}>
      On the host, open Manage → Connections, create a new invite with “Agent” turned on, and pair this device again.</Text>
    <Action label="Back to chat" onPress={onBack} />
  </View>;
}

function ToolRow({ entry }: { entry: TimelineEntry }) {
  const styles = useStyles();
  const palette = useTheme();
  const [open, setOpen] = useState(false);
  const bad = entry.status === 'failed' || entry.status === 'denied';
  const glyph = { running: '…', reviewing: '…', 'awaiting-approval': '?', completed: '✓', failed: '✕', denied: '⊘' }[entry.status];
  const state = {
    running: 'running', reviewing: 'checking', 'awaiting-approval': 'needs approval', completed: '', failed: 'failed',
    denied: 'denied',
  }[entry.status];
  return <Pressable accessibilityRole="button" accessibilityLabel={`${entry.tool} ${state}`} style={styles.card}
    onPress={() => setOpen(!open)}>
    <View style={[styles.row, { flexWrap: 'nowrap' }]}>
      <Text style={[styles.check, bad && { color: palette.danger }]}>{glyph}</Text>
      <Text numberOfLines={open ? 6 : 1} style={[styles.mono, styles.flex]}>
        <Text style={{ color: palette.text, fontWeight: '700' }}>{entry.tool}</Text>
        {'  '}{argsPreview(entry.tool, entry.args)}</Text>
      <Text style={[styles.listMeta, bad && { color: palette.danger }]}>
        {entry.environment === 'host' ? 'host' : 'sandbox'}{state ? ` · ${state}` : ''}</Text>
    </View>
    {entry.reviewNote ? <Text style={styles.listMeta}>{entry.reviewNote}</Text> : null}
    {open ? <>
      {entry.error ? <Text selectable style={[styles.mono, { color: palette.danger }]}>{entry.error}</Text> : null}
      {entry.output ? <Text selectable style={styles.mono}>{entry.output.slice(-4000)}</Text> : null}
      {entry.changedPaths?.length ? <Text style={styles.listMeta}>Changed: {entry.changedPaths.join(', ')}</Text> : null}
      {entry.sandbox && !entry.sandbox.isolated ? <Text style={styles.listMeta}>{entry.sandbox.detail}</Text> : null}
    </> : null}
  </Pressable>;
}

function ApprovalCard({ approval, busy, onDecide }: {
  approval: AgentApproval; busy: boolean;
  onDecide: (approval: AgentApproval, decision: 'approve' | 'deny', scope?: 'once' | 'session') => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const elevation = approval.elevation;
  const host = approval.environment === 'host' || !!elevation.requested_host_execution;
  const paths = elevation.requested_filesystem_paths ?? [];
  const facts: [string, string, boolean][] = [
    ['Tool', `${approval.tool} · ${approval.risk}`, false],
    ['Environment', host ? 'Host — no sandbox, full user privileges' : `Sandbox · ${approval.sandbox.backend}`, host],
    ['Runs on', locationLine(approval.execution_location), false],
    ...(elevation.proposed_command ? [['Command', elevation.proposed_command, false] as [string, string, boolean]] : []),
    ['Filesystem', paths.length ? paths.map((entry) => `${entry.path}${entry.write ? ' (write)' : ' (read)'}`).join(', ')
      : 'Inside the workspace only', paths.length > 0],
    ['Network', elevation.requested_network_access ? 'Outbound access requested' : 'Blocked', !!elevation.requested_network_access],
  ];
  return <View accessibilityRole="alert" style={[styles.card, { borderColor: host ? palette.danger : palette.accent }]}>
    <Text style={[styles.body, { fontWeight: '700' }]}>{host ? '⚠ ' : '🔒 '}{approval.summary}</Text>
    {elevation.reason ? <Text style={styles.listMeta}>{elevation.reason}</Text> : null}
    {facts.map(([label, value, warn]) => <View key={label} style={{ flexDirection: 'row', gap: 8 }}>
      <Text style={[styles.listMeta, { width: 86 }]}>{label}</Text>
      <Text selectable style={[styles.mono, styles.flex, warn && { color: palette.danger }]}>{value}</Text>
    </View>)}
    {!approval.sandbox.isolated ? <Text style={[styles.listMeta, { color: palette.danger }]}>{approval.sandbox.detail}</Text> : null}
    <View style={styles.row}>
      <Action label="Deny" danger disabled={busy} onPress={() => onDecide(approval, 'deny')} />
      {approval.allow_session_scope
        ? <Action label="Allow for session" quiet disabled={busy} onPress={() => onDecide(approval, 'approve', 'session')} /> : null}
      <Action label="Allow once" disabled={busy} onPress={() => onDecide(approval, 'approve', 'once')} />
    </View>
  </View>;
}

export function Agent({ credential, hostName, onUnauthorized, onOpenChat, onOpenSettings }: {
  credential: Credential; hostName: string; onUnauthorized: (error: unknown) => void;
  onOpenChat: () => void; onOpenSettings: () => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const api = useMemo(() => agentApi(fetch, credential), [credential]);
  const [view, setView] = useState<AgentView>(emptyView);
  const [sessions, setSessions] = useState<AgentSessionSummary[]>([]);
  const [active, setActive] = useState<AgentSessionSummary | null>(null);
  const [draft, setDraft] = useState<Draft>({ model: '', workspace: '', mode: 'ask', tools: null });
  const [models, setModels] = useState<Model[]>([]);
  const [catalog, setCatalog] = useState<AgentTool[]>([]);
  const [capabilities, setCapabilities] = useState<AgentCapabilities | null>(null);
  const [sheet, setSheet] = useState<'models' | 'workspace' | 'mode' | 'tools' | null>(null);
  const [showDrawer, setShowDrawer] = useState(false);
  const [text, setText] = useState('');
  const [folder, setFolder] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [deciding, setDeciding] = useState(false);
  const activeId = useRef<string | null>(null);
  const scroll = useRef<ScrollView>(null);
  const controller = useMemo(() => new AgentController(fetch, credential, (sessionId, event) => {
    if (sessionId === activeId.current) setView((current) => applyEvent(current, event));
  }), [credential]);
  useEffect(() => () => controller.dispose(), [controller]);

  const model = active?.model ?? draft.model;
  const workspace = active?.workspace_path ?? draft.workspace;
  const mode = active?.permission_mode ?? draft.mode;
  const offered = selectableTools(catalog, active?.enabled_tools);
  const enabled = active ? active.enabled_tools ?? null : draft.tools;
  const isOn = (name: string) => enabled ? enabled.includes(name) : true;
  const enabledCount = offered.filter((tool) => isOn(tool.name)).length;
  const locked = view.running || !!busy;

  function report(cause: unknown) {
    if (cause instanceof DaemonError && cause.status === 401) onUnauthorized(cause);
    else setError(agentErrorText(errorText(cause)));
  }

  /** Run one host request with a status line; returns whether it succeeded. */
  async function attempt(label: string, task: () => Promise<void>): Promise<boolean> {
    setBusy(label);
    setError('');
    try {
      await task();
      return true;
    } catch (cause) {
      report(cause);
      return false;
    } finally {
      setBusy('');
    }
  }

  const select = (session: AgentSessionSummary | null) => {
    activeId.current = session?.id ?? null;
    setActive(session);
  };
  const replace = (session: AgentSessionSummary) => {
    setActive(session);
    setSessions((current) => current.map((entry) => entry.id === session.id ? session : entry));
  };
  /** After the core changed a session on the host, show what the host now holds. */
  const reload = async (id: string) => replace((await api.session(id)).session);

  const refresh = () => attempt('Refreshing…', async () => {
    const [available, stored, tools, limits] = await Promise.all([
      daemonClient(fetch, credential).models(), api.sessions(), api.tools(), api.capabilities(),
    ]);
    setModels(available);
    setSessions(stored);
    setCatalog(tools);
    setCapabilities(limits);
    // A new task starts where the last one was, like the desktop's remembered folder.
    setDraft((current) => ({
      ...current, model: current.model || stored[0]?.model || '',
      workspace: current.workspace || stored[0]?.workspace_path || '',
    }));
  });

  useEffect(() => { void refresh(); }, []);

  function newTask() {
    setShowDrawer(false);
    select(null);
    setView(emptyView);
    setError('');
  }

  async function openSession(session: AgentSessionSummary) {
    setShowDrawer(false);
    await attempt('Loading task…', async () => {
      const detail = await api.session(session.id);
      select(detail.session);
      setView(viewFromDetail(detail));
    });
  }

  function removeSession(session: AgentSessionSummary) {
    Alert.alert('Delete this task?', `“${session.title}” and its history are removed from ${hostName}.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => void attempt('Deleting…', async () => {
        await controller.close(session.id).catch(() => undefined);
        await api.deleteSession(session.id);
        setSessions((current) => current.filter((entry) => entry.id !== session.id));
        if (activeId.current === session.id) { select(null); setView(emptyView); }
      }) },
    ]);
  }

  async function chooseModel(id: string) {
    setSheet(null);
    if (!active) return setDraft((current) => ({ ...current, model: id }));
    await attempt('Changing model…', async () => {
      await controller.setModel(active.id, id);
      await reload(active.id);
    });
  }

  async function chooseMode(next: AgentPermissionMode) {
    setSheet(null);
    if (!active) return setDraft((current) => ({ ...current, mode: next }));
    await attempt('Changing permissions…', async () => {
      await controller.setPermissionMode(active.id, next);
      await reload(active.id);
    });
  }

  // The host checks the folder when a session takes it: only the owner may probe paths, so a
  // paired device learns whether a folder is allowed by using it.
  async function chooseWorkspace(path: string) {
    const wanted = path.trim();
    if (!wanted) return;
    if (!active) {
      setDraft((current) => ({ ...current, workspace: wanted }));
      return setSheet(null);
    }
    const applied = await attempt('Checking folder…', async () => {
      replace(await api.setWorkspace(active.id, wanted));
      // A loaded session keeps the folder it was opened with and would start delegated tasks
      // there; drop it so the next run loads the session with its new folder.
      await controller.close(active.id);
    });
    if (applied) setSheet(null);
  }

  async function setTools(names: string[]) {
    if (!active) return setDraft((current) => ({ ...current, tools: names }));
    await attempt('Changing tools…', async () => {
      await controller.setTools(active.id, names);
      await reload(active.id);
    });
  }
  const toggleTool = (name: string) => {
    const current = offered.filter((tool) => isOn(tool.name)).map((tool) => tool.name);
    void setTools(current.includes(name) ? current.filter((other) => other !== name) : [...current, name]);
  };

  async function send() {
    const prompt = text.trim();
    if (!prompt || locked) return;
    if (!model) return setError('Choose a model before starting a task.');
    if (!workspace) return setError('Choose a folder on the host first. The agent works inside it.');
    Keyboard.dismiss();
    let session = active;
    if (!session) {
      const created = await attempt('Starting task…', async () => {
        session = await api.createSession({
          title: Array.from(prompt).slice(0, 60).join(''), workspace_path: workspace, model, permission_mode: mode,
          ...(draft.tools ? { enabled_tools: draft.tools } : {}),
        });
        select(session);
        setSessions((current) => [session!, ...current]);
      });
      if (!created || !session) return;
    }
    const id = session.id;
    setText('');
    setError('');
    setView((current) => ({
      ...current, running: true, summary: null, error: '',
      messages: [...current.messages, { role: 'user', text: prompt, timestamp: new Date().toISOString() }],
    }));
    try {
      await controller.run(id, prompt);
    } catch (cause) {
      if (activeId.current === id) {
        setView((current) => ({ ...current, running: false }));
        report(cause);
      }
    } finally {
      void api.session(id).then(({ session: stored }) => {
        setSessions((current) => current.map((entry) => entry.id === id ? stored : entry));
        if (activeId.current === id) setActive(stored);
      }).catch(() => undefined);
    }
  }

  async function stop() {
    if (!active) return;
    try {
      await controller.cancel(active.id);
    } catch (cause) {
      report(cause);
    }
  }

  async function decide(approval: AgentApproval, decision: 'approve' | 'deny', scope?: 'once' | 'session') {
    setDeciding(true);
    try {
      await api.decide(approval, decision, scope);
      setView((current) => ({ ...current, approvals: current.approvals.filter((entry) => entry.id !== approval.id) }));
    } catch (cause) {
      report(cause);
    } finally {
      setDeciding(false);
    }
  }

  const rows = useMemo(() => transcriptRows(view), [view.messages, view.timeline]);
  const modes = permissionModesFor(capabilities);
  const folders = [...new Set(sessions.flatMap((session) => session.workspace_path ? [session.workspace_path] : []))];
  const status = busy || view.progress || (view.running
    ? view.approvals.length ? 'Waiting for your approval…' : 'Working…' : '');
  const blank = rows.length === 0 && !view.running;
  const chip = (label: string, hint: string, target: NonNullable<typeof sheet>) =>
    <Pressable accessibilityRole="button" accessibilityLabel={hint} disabled={locked}
      style={[styles.chip, locked && styles.disabled]} onPress={() => { setFolder(workspace); setSheet(target); }}>
      <Text numberOfLines={1} style={styles.chipText}>{label} ▾</Text>
    </Pressable>;

  return <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.container}>
    <View style={styles.topBar}>
      <IconButton glyph="☰" label="Tasks" onPress={() => {
        Keyboard.dismiss();
        setShowDrawer(true);
        void api.sessions().then(setSessions).catch(() => undefined);
      }} />
      <Text numberOfLines={1} style={styles.topTitle}>{active?.title ?? 'New task'}</Text>
      {chip(model || 'Choose model', 'Choose model', 'models')}
      <IconButton glyph="＋" label="New task" disabled={locked} onPress={newTask} />
    </View>
    <View style={[styles.row, { paddingHorizontal: 10, paddingVertical: 8 }]}>
      {chip(workspace ? `▤ ${folderName(workspace)}` : '▤ Choose folder', 'Choose folder', 'workspace')}
      {chip(PERMISSION_LABELS[mode].title, 'Permissions', 'mode')}
      {chip(`⚙ ${enabledCount}`, `Tools, ${enabledCount} of ${offered.length} on`, 'tools')}
    </View>
    {error || view.error ? <Text accessibilityRole="alert" style={[styles.error, { marginHorizontal: 10 }]}>
      {error || view.error}</Text> : null}
    {blank ? <Pressable accessible={false} style={styles.empty} onPress={Keyboard.dismiss}>
      <Text style={styles.brand}>Agent</Text>
      <Text style={styles.emptyText}>{!model ? 'Choose a model to start.' : !workspace
        ? `Choose a folder on ${hostName} for the agent to work in.`
        : `Describe a task. It runs on ${hostName}, in ${folderName(workspace)}.`}</Text>
      {!model && !busy ? <Action label="Choose model" onPress={() => setSheet('models')} /> : null}
      {model && !workspace && !busy ? <Action label="Choose folder" onPress={() => { setFolder(''); setSheet('workspace'); }} /> : null}
    </Pressable> : <ScrollView style={styles.flex} contentContainerStyle={styles.messages} ref={scroll}
      onContentSizeChange={() => scroll.current?.scrollToEnd({ animated: false })} keyboardShouldPersistTaps="handled" keyboardDismissMode={dragDismiss}>
      {rows.map((row) => {
        if (row.kind === 'tool') return <ToolRow key={row.key} entry={row.entry} />;
        const message = row.message;
        if (message.role === 'user') {
          return <View key={row.key} style={styles.userBubble}><Text selectable style={styles.body}>{message.text}</Text></View>;
        }
        if (message.role === 'system') return <Text key={row.key} style={styles.listMeta}>{message.text}</Text>;
        if (message.role !== 'assistant') return null;
        return <View key={row.key} style={styles.assistant}>
          {message.reasoning?.trim() ? <Text style={[styles.listMeta, { fontStyle: 'italic' }]}>{message.reasoning.trim()}</Text> : null}
          {message.text.trim() ? <Markdown text={message.text} /> : null}
          {message.error ? <Text style={styles.error}>{message.error}</Text> : null}
        </View>;
      })}
      {view.reasoning ? <Text style={[styles.listMeta, { fontStyle: 'italic' }]}>{view.reasoning}</Text> : null}
      {view.streaming ? <View style={styles.assistant}><Markdown text={view.streaming} /></View> : null}
      {view.approvals.map((approval) =>
        <ApprovalCard key={approval.id} approval={approval} busy={deciding} onDecide={(...args) => void decide(...args)} />)}
      {view.summary && !view.running ? <Text style={styles.listMeta}>Done · {summaryLine(view.summary)}</Text> : null}
    </ScrollView>}
    {status ? <Text accessibilityRole="progressbar" style={styles.status}>{status}</Text> : null}
    <View style={styles.composer}>
      <TextInput accessibilityLabel="Task" placeholder={`Give ${hostName} a task`} placeholderTextColor={palette.muted}
        style={styles.composeInput} value={text} onChangeText={setText} editable={!locked} multiline />
      {view.running
        ? <Pressable accessibilityRole="button" accessibilityLabel="Stop" style={styles.send} onPress={() => void stop()}>
          <Text style={styles.sendText}>■</Text></Pressable>
        : <Pressable accessibilityRole="button" accessibilityLabel="Send" disabled={!text.trim() || locked}
          style={[styles.send, (!text.trim() || locked) && styles.disabled]} onPress={() => void send()}>
          <Text style={styles.sendText}>↑</Text></Pressable>}
    </View>

    <Sheet title="Agent model" visible={sheet === 'models'} onClose={() => setSheet(null)}>
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

    <Sheet title="Folder on the host" visible={sheet === 'workspace'} onClose={() => setSheet(null)}>
      <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={{ paddingHorizontal: 18, gap: 10 }}>
        <Text style={styles.listMeta}>
          The agent reads, writes and runs commands inside this folder on {hostName}. A paired device can only use a
          folder that was already opened in Agent mode on the host.</Text>
        <TextInput accessibilityLabel="Folder path" placeholder="/Users/you/project" placeholderTextColor={palette.muted}
          autoCapitalize="none" autoCorrect={false} style={[styles.input, styles.mono, { fontSize: 14 }]}
          value={folder} onChangeText={setFolder} onSubmitEditing={() => void chooseWorkspace(folder)} />
        {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
        <Action label={busy || 'Use this folder'} disabled={!folder.trim() || !!busy} onPress={() => void chooseWorkspace(folder)} />
        {folders.length ? <Text style={styles.section}>Used before</Text> : null}
        {folders.map((path) => <Pressable key={path} accessibilityRole="button" style={[styles.listItem, { paddingHorizontal: 0 }]}
          onPress={() => void chooseWorkspace(path)}>
          <Text style={styles.check}>{path === workspace ? '✓' : ''}</Text>
          <Text numberOfLines={2} style={[styles.mono, styles.flex]}>{path}</Text>
        </Pressable>)}
      </ScrollView>
    </Sheet>

    <Sheet title="Permissions" visible={sheet === 'mode'} onClose={() => setSheet(null)}>
      <ScrollView>
        {modes.map((option) => <Pressable key={option} accessibilityRole="button"
          style={[styles.listItem, option === mode && styles.listItemOn]} onPress={() => void chooseMode(option)}>
          <Text style={styles.check}>{option === mode ? '✓' : ''}</Text>
          <View style={styles.flex}>
            <Text style={styles.listText}>{PERMISSION_LABELS[option].title}</Text>
            <Text style={styles.listMeta}>{PERMISSION_LABELS[option].detail}</Text>
          </View>
        </Pressable>)}
        <Text style={[styles.listMeta, { paddingHorizontal: 18, paddingTop: 8 }]}>
          Skipping permissions and unprompted host actions can only be turned on from the host itself.</Text>
      </ScrollView>
    </Sheet>

    <Sheet title={`Tools · ${enabledCount} of ${offered.length} on`} visible={sheet === 'tools'} onClose={() => setSheet(null)}>
      <ScrollView>
        <View style={[styles.row, { paddingHorizontal: 18, paddingBottom: 8 }]}>
          <Action label="All" quiet disabled={!!busy} onPress={() => void setTools(offered.map((tool) => tool.name))} />
          <Action label="None" quiet disabled={!!busy} onPress={() => void setTools([])} />
        </View>
        {offered.map((tool) => <Pressable key={tool.name} accessibilityRole="switch" disabled={!!busy}
          accessibilityState={{ checked: isOn(tool.name) }} style={styles.listItem} onPress={() => toggleTool(tool.name)}>
          <Text style={styles.check}>{isOn(tool.name) ? '✓' : ''}</Text>
          <View style={styles.flex}>
            <Text style={styles.listText}>{tool.label || tool.name}  <Text style={styles.listMeta}>{tool.risk}</Text></Text>
            {tool.description ? <Text numberOfLines={2} style={styles.listMeta}>{tool.description}</Text> : null}
          </View>
        </Pressable>)}
      </ScrollView>
    </Sheet>

    <Modal transparent visible={showDrawer} animationType="fade" onRequestClose={() => setShowDrawer(false)}>
      <View style={styles.backdrop}>
        <SafeAreaView style={styles.drawer}>
          <View style={styles.header}>
            <Text style={styles.brand}>Brazier</Text>
            <Text numberOfLines={1} style={styles.subtitle}>{hostName} · Agent</Text>
          </View>
          <View style={{ paddingHorizontal: 18 }}><Action label="New task" disabled={locked} onPress={newTask} /></View>
          <Text style={[styles.section, { paddingHorizontal: 18 }]}>Tasks</Text>
          <ScrollView style={styles.flex}>
            {busy === 'Refreshing…' ? <ActivityIndicator color={palette.accent} /> : null}
            {sessions.length === 0 && !busy ? <Text style={[styles.listMeta, { padding: 18 }]}>
              No tasks from this device yet.</Text> : null}
            {sessions.map((session) => <Pressable key={session.id} accessibilityRole="button" disabled={locked}
              style={[styles.listItem, session.id === active?.id && styles.listItemOn]}
              onPress={() => void openSession(session)}>
              <View style={styles.flex}>
                <Text numberOfLines={1} style={styles.listText}>{session.title}</Text>
                <Text numberOfLines={1} style={styles.listMeta}>
                  {session.workspace_path ? folderName(session.workspace_path) : 'No folder'} · {session.id === active?.id && !view.running && view.summary ? 'completed' : session.last_run_status}</Text>
              </View>
              <IconButton glyph="✕" label={`Delete ${session.title}`} disabled={locked} onPress={() => removeSession(session)} />
            </Pressable>)}
          </ScrollView>
          <Pressable accessibilityRole="button" style={styles.listItem} onPress={() => { setShowDrawer(false); onOpenChat(); }}>
            <Text style={styles.iconText}>💬</Text><Text style={styles.listText}>Chat</Text>
          </Pressable>
          <Pressable accessibilityRole="button" style={styles.listItem} onPress={() => { setShowDrawer(false); onOpenSettings(); }}>
            <Text style={styles.iconText}>⚙</Text><Text style={styles.listText}>Settings</Text>
          </Pressable>
        </SafeAreaView>
        <Pressable style={styles.flex} accessibilityLabel="Close tasks" onPress={() => setShowDrawer(false)} />
      </View>
    </Modal>
  </KeyboardAvoidingView>;
}
