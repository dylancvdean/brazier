import { CameraView, useCameraPermissions } from 'expo-camera';
import { fetch } from 'expo/fetch';
import { useRef, useState } from 'react';
import { ActivityIndicator, Platform, ScrollView, Text, TextInput } from 'react-native';
import { parsePairingInvite } from '../../../packages/pairing/src/invite.ts';
import {
  claimInvite, pairingOrigin, previewInvite, UnreachableError,
  type Credential, type InviteInput, type Preview,
} from './api.ts';
import { checkLocalNetwork } from './localNetwork';
import { pinCertificate } from './tlsPinning';
import { useTheme } from './theme.ts';
import { Action, errorText, useStyles } from './ui.tsx';

const androidEmulator = __DEV__ && Platform.OS === 'android';
const emptyInvite = { origin: '', pairingId: '', code: '' };

function inviteCertificate(invite: object): { certificateSha256?: string } {
  return 'certificateSha256' in invite && typeof invite.certificateSha256 === 'string'
    ? { certificateSha256: invite.certificateSha256 } : {};
}

/** `onPaired` must persist the credential and throw if it cannot. */
export function Pairing({ onPaired, onCancel }: {
  onPaired: (credential: Credential, hostName: string) => Promise<void>; onCancel?: () => void;
}) {
  const styles = useStyles();
  const palette = useTheme();
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
        await onPaired(issued, review.preview.daemon.display_name);
      } catch {
        throw new Error('Invite claimed, but the credential was not saved. Ask the host to revoke this device and issue a fresh invite.');
      }
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
      <TextInput placeholderTextColor={palette.muted} accessibilityLabel="Pairing invite" placeholder="Paste pairing invite" value={inviteText}
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
        return <TextInput placeholderTextColor={palette.muted} key={field} accessibilityLabel={label} placeholder={label} style={styles.input}
          editable={!busy} value={input[field]} onChangeText={(value) => setInput({ ...input, [field]: value })}
          autoCapitalize="none" autoCorrect={false} secureTextEntry={field === 'code'}
          keyboardType={field === 'origin' ? 'url' : 'default'} />;
      })}
      <Action label="Review details" disabled={busy || !input.origin || !input.pairingId || !input.code}
        quiet onPress={() => void inspect(input)} />
    </>}
    {busy ? <ActivityIndicator color={palette.accent} /> : null}
    {onCancel && !review && !scanning ? <Action label="Back" quiet disabled={busy} onPress={onCancel} /> : null}
  </ScrollView>;
}
