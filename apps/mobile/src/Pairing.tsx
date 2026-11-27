import { CameraView, useCameraPermissions } from 'expo-camera';
import { fetch } from 'expo/fetch';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { BackHandler, Platform, ScrollView, Text, TextInput, View } from 'react-native';
import { parsePairingInvite } from '../../../packages/pairing/src/invite.ts';
import {
  claimInvite, pairingOrigin, previewInvite, UnreachableError,
  type Credential, type InviteInput, type Preview,
} from './api.ts';
import { checkLocalNetwork } from './localNetwork';
import { pinCertificate } from './tlsPinning';
import { useTheme } from './theme.ts';
import { describeScopes } from './navigation.ts';
import { Action, Busy, dragDismiss, ErrorText, errorText, useStyles } from './ui.tsx';

const androidEmulator = __DEV__ && Platform.OS === 'android';
const emptyInvite = { origin: '', pairingId: '', code: '' };

function inviteCertificate(invite: object): { certificateSha256?: string } {
  return 'certificateSha256' in invite && typeof invite.certificateSha256 === 'string'
    ? { certificateSha256: invite.certificateSha256 } : {};
}

const fields = [
  { field: 'origin', label: 'Host address', placeholder: 'https://192.168.1.20:7614' },
  { field: 'pairingId', label: 'Pairing ID', placeholder: 'From the invite' },
  { field: 'code', label: 'Pairing code', placeholder: 'From the invite' },
] as const;

/** `onPaired` must persist the credential and throw if it cannot. `header` and `footer` scroll with the form. */
export function Pairing({ onPaired, onCancel, header, footer }: {
  onPaired: (credential: Credential, hostName: string) => Promise<void>; onCancel?: () => void;
  header?: ReactNode; footer?: ReactNode;
}) {
  const styles = useStyles();
  const palette = useTheme();
  const [permission, requestPermission] = useCameraPermissions();
  const [scanning, setScanning] = useState(false);
  const scanLock = useRef(false);
  const [inviteText, setInviteText] = useState('');
  const [input, setInput] = useState<InviteInput>(emptyInvite);
  const [review, setReview] = useState<{ input: InviteInput; preview: Preview } | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const inputs = useRef<Partial<Record<typeof fields[number]['field'], TextInput | null>>>({});

  function cancelReview() {
    setReview(null); setInput(emptyInvite); setInviteText('');
  }

  // Back steps out of the review or the scanner before it leaves pairing.
  const nested = !!review || scanning;
  useEffect(() => {
    if (!nested) return undefined;
    const listener = BackHandler.addEventListener('hardwareBackPress', () => {
      if (busy) return true;
      if (review) cancelReview();
      else setScanning(false);
      return true;
    });
    return () => listener.remove();
  }, [nested, review, busy]);

  function parse(text: string) {
    const invite = parsePairingInvite(text);
    if (!invite) throw new Error('This is not a supported Brazier invite. Use the host’s QR or Copy invite.');
    if (invite.version === 1 && invite.expiresAt <= Date.now() / 1000) {
      throw new Error('This invite expired. Ask the host for a fresh invite.');
    }
    return invite;
  }

  async function inspect(candidate: InviteInput) {
    setBusy('Contacting the host…');
    setError('');
    try {
      const target = { ...candidate, origin: pairingOrigin(candidate.origin, androidEmulator),
        pairingId: candidate.pairingId.trim(), code: candidate.code.trim() };
      if (!target.pairingId || !target.code) throw new Error('Enter the pairing ID and code.');
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
      setBusy('');
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
    setBusy('Pairing…');
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
      setBusy('');
      setReview(null);
      setInviteText('');
      setInput(emptyInvite);
    }
  }

  return <ScrollView contentContainerStyle={styles.panel} keyboardShouldPersistTaps="handled" keyboardDismissMode={dragDismiss}>
    {header}
    <Text accessibilityRole="header" style={styles.heading}>{review ? 'Review pairing' : 'Pair with your host'}</Text>
    {error ? <ErrorText message={error} /> : null}
    {review ? <>
      <Text style={styles.body}>Check that these match what the host shows, then confirm.</Text>
      <View style={styles.card}>
        {([
          ['Host', review.preview.daemon.display_name], ['Address', review.input.origin],
          ['This device', review.preview.pairing.client_name], ['Access', describeScopes(review.preview.pairing.scopes)],
        ] as const).map(([label, value]) => <View key={label} accessible>
          <Text style={styles.label}>{label}</Text>
          <Text style={styles.body}>{value}</Text>
        </View>)}
      </View>
      {busy ? <Busy label={busy} /> : null}
      <Action label="Confirm and pair" disabled={!!busy} onPress={() => void claim()} />
      <Action label="Cancel" disabled={!!busy} quiet onPress={cancelReview} />
    </> : scanning ? <>
      <Text style={styles.body}>Point the camera at the QR code shown on the host.</Text>
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
      <Text style={styles.body}>On your Brazier host, open Manage → Connections and create an invite. Then scan its QR code or paste it here.</Text>
      <Action label="Scan QR code" disabled={!!busy} onPress={() => void scan()} />
      <TextInput placeholderTextColor={palette.muted} accessibilityLabel="Pairing invite" placeholder="Paste pairing invite" value={inviteText}
        keyboardAppearance={palette.dark ? 'dark' : 'light'}
        style={[styles.input, styles.invite]} editable={!busy} onChangeText={setInviteText} multiline
        autoCapitalize="none" autoCorrect={false} />
      <Action label="Review invite" disabled={!!busy || !inviteText.trim()} onPress={() => {
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
      <Text accessibilityRole="header" style={styles.section}>Or enter the invite details</Text>
      {fields.map(({ field, label, placeholder }, index) => {
        const next = fields[index + 1]?.field;
        const complete = !!input.origin && !!input.pairingId && !!input.code;
        return <View key={field} style={styles.field}>
          <Text style={styles.label}>{label}</Text>
          <TextInput ref={(node) => { inputs.current[field] = node; }} placeholderTextColor={palette.muted}
            accessibilityLabel={label} placeholder={placeholder} style={styles.input}
            editable={!busy} value={input[field]} onChangeText={(value) => setInput({ ...input, [field]: value })}
            autoCapitalize="none" autoCorrect={false} secureTextEntry={field === 'code'}
            keyboardType={field === 'origin' ? 'url' : 'default'} keyboardAppearance={palette.dark ? 'dark' : 'light'}
            returnKeyType={next ? 'next' : 'done'} submitBehavior={next ? 'submit' : 'blurAndSubmit'}
            onSubmitEditing={() => {
              if (next) inputs.current[next]?.focus();
              else if (complete && !busy) void inspect(input);
            }} />
        </View>;
      })}
      <Action label="Review details" disabled={!!busy || !input.origin || !input.pairingId || !input.code}
        quiet onPress={() => void inspect(input)} />
      {busy ? <Busy label={busy} /> : null}
    </>}
    {onCancel && !review && !scanning ? <Action label="Back" quiet disabled={!!busy} onPress={onCancel} /> : null}
    {!review && !scanning ? footer : null}
  </ScrollView>;
}
