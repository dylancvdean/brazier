export type ProbeStatus = 'ready' | 'denied' | 'refused' | 'failed' | 'timeout';
export type ProbeInterface = 'wifi' | 'wired' | 'cellular' | 'loopback' | 'other' | 'none';
export type Probe = {
  status: ProbeStatus;
  interface: ProbeInterface;
  error?: string | null;
  /** IPv4 subnets of the phone's Wi-Fi and Ethernet interfaces. */
  networks: { address: string; netmask: string }[];
};
export type ProbeTarget = { host: string; port: number };

function ipv4(value: string): number | null {
  const parts = value.split('.');
  if (parts.length !== 4 || !parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255)) return null;
  return parts.reduce((total, part) => total * 256 + Number(part), 0);
}

function privateIpv4(address: number): boolean {
  const [a, b] = [Math.floor(address / 2 ** 24), Math.floor(address / 2 ** 16) % 256];
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254);
}

/** The host and port to probe, for origins that iOS treats as local network hosts. */
export function probeTarget(origin: string): ProbeTarget | null {
  let url: URL;
  try {
    url = new URL(origin);
  } catch {
    return null;
  }
  const host = url.hostname.replace(/^\[|\]$/g, '').replace(/\.$/, '');
  const address = ipv4(host);
  const local = address === null
    ? host.endsWith('.local') || (host.includes(':') && /^(fe80|fc|fd)/i.test(host))
    : privateIpv4(address);
  if (!local) return null;
  return { host, port: Number(url.port || (url.protocol === 'http:' ? 80 : 443)) };
}

function prefix(netmask: string): number {
  let bits = ipv4(netmask) ?? 0;
  let count = 0;
  while (bits > 0) {
    count += bits % 2;
    bits = Math.floor(bits / 2);
  }
  return count;
}

function onSubnet(host: number, network: Probe['networks'][number]): boolean {
  const address = ipv4(network.address);
  const mask = ipv4(network.netmask);
  if (address === null || mask === null) return false;
  return ((host & mask) >>> 0) === ((address & mask) >>> 0);
}

const firewallHelp = 'On a Mac host, open System Settings → Network → Firewall and allow incoming connections for Brazier.';

/** Why the phone cannot reach the host, or null when it can. */
export function explainProbe(target: ProbeTarget, probe: Probe): string | null {
  const { host, port } = target;
  if (probe.status === 'ready') return null;
  if (probe.status === 'denied') {
    return 'Brazier needs Local Network access to reach your host. Turn it on in Settings → Privacy & Security → Local Network, then try again.';
  }
  if (probe.status === 'refused') {
    return `${host} refused the connection on port ${port}. Make sure Brazier is running on the host, then create a fresh invite there.`;
  }
  const address = ipv4(host);
  const viaVpn = probe.interface === 'other';
  if (!probe.networks.length) {
    return `This phone is not on Wi-Fi, so it cannot reach ${host}. Join the same Wi-Fi network as the host and try again.`;
  }
  if (address !== null && !probe.networks.some((network) => onSubnet(address, network))) {
    const networks = probe.networks.map((network) => `${network.address}/${prefix(network.netmask)}`).join(', ');
    return `${host} is not on this phone's network (${networks}). ` +
      (viaVpn ? 'A VPN is routing the connection; turn it off or allow local network access in it. ' : '') +
      'Join the same Wi-Fi network as the host, or check the address in the invite.';
  }
  if (viaVpn) {
    return `A VPN on this phone is routing the connection to ${host}. Turn it off or allow local network access in it, then try again.`;
  }
  return `${host} did not answer on port ${port}. Check that Brazier is running there and that its firewall allows incoming connections. ` +
    `${firewallHelp} Guest Wi-Fi and networks with client isolation also block devices from reaching each other.`;
}
