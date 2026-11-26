import { NativeModules } from 'react-native';

type PinningModule = { pin(hostAndPort: string, sha256: string): Promise<void> };

/** Accept exactly this certificate for the origin's host and port. */
export async function pinCertificate(origin: string, certificateSha256?: string): Promise<void> {
  if (!certificateSha256) return;
  const native = NativeModules.BrazierTLSPinning as PinningModule | undefined;
  if (!native?.pin) throw new Error('This version of Brazier cannot verify the host’s certificate. Update the app.');
  const url = new URL(origin);
  await native.pin(`${url.hostname.replace(/^\[|\]$/g, '')}:${url.port || '443'}`, certificateSha256);
}
