import { NativeModules, Platform } from 'react-native';
import { explainProbe, probeTarget, type Probe } from './reachability.ts';

type LocalNetworkModule = { probe(host: string, port: number, timeoutMs: number): Promise<Probe> };

/**
 * Connect to a local network host before talking to it, so iOS shows the
 * Local Network alert first and an unreachable host gets a specific reason.
 */
export async function checkLocalNetwork(origin: string, timeoutMs = 20_000): Promise<void> {
  if (Platform.OS !== 'ios') return;
  const target = probeTarget(origin);
  const native = NativeModules.BrazierLocalNetwork as LocalNetworkModule | undefined;
  if (!target || !native?.probe) return;
  const reason = explainProbe(target, await native.probe(target.host, target.port, timeoutMs));
  if (reason) throw new Error(reason);
}
