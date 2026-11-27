import assert from 'node:assert/strict';
import { test } from 'node:test';
import { explainProbe, probeTarget, type Probe } from './reachability.ts';

const home = [{ address: '192.168.1.20', netmask: '255.255.255.0' }];
const target = { host: '192.168.1.51', port: 7614 };
const probe = (status: Probe['status'], overrides: Partial<Probe> = {}): Probe =>
  ({ status, interface: 'wifi', networks: home, ...overrides });

test('probes only hosts that iOS treats as local network hosts', () => {
  assert.deepEqual(probeTarget('https://192.168.1.51:7614'), target);
  assert.deepEqual(probeTarget('https://10.0.0.2'), { host: '10.0.0.2', port: 443 });
  assert.deepEqual(probeTarget('https://studio.local:7614'), { host: 'studio.local', port: 7614 });
  assert.deepEqual(probeTarget('https://[fe80::1]:7614'), { host: 'fe80::1', port: 7614 });
  assert.equal(probeTarget('https://100.101.102.103:7614'), null);
  assert.equal(probeTarget('https://brazier.example'), null);
  assert.deepEqual(probeTarget('https://[fd12:3456::1]:7614'), { host: 'fd12:3456::1', port: 7614 });
  // Public names that merely start like a local IPv6 prefix are not local hosts.
  assert.equal(probeTarget('https://fdroid.example.com'), null);
  assert.equal(probeTarget('https://fc.example.com:7614'), null);
  assert.equal(probeTarget('https://fe80.example.com'), null);
  assert.equal(probeTarget('not a url'), null);
});

test('explains each reason the phone cannot reach the host', () => {
  assert.equal(explainProbe(target, probe('ready')), null);
  assert.match(explainProbe(target, probe('denied')) ?? '', /Local Network/);
  assert.match(explainProbe(target, probe('refused')) ?? '', /refused the connection on port 7614/);
  assert.match(explainProbe(target, probe('timeout', { networks: [] })) ?? '', /not on Wi-Fi/);
  assert.match(
    explainProbe(target, probe('timeout', { networks: [{ address: '192.168.0.23', netmask: '255.255.255.0' }] })) ?? '',
    /not on this phone's network \(192\.168\.0\.23\/24\)/,
  );
  assert.match(
    explainProbe(target, probe('timeout', { interface: 'other', networks: [{ address: '10.0.0.5', netmask: '255.0.0.0' }] })) ?? '',
    /VPN is routing the connection/,
  );
  assert.match(explainProbe(target, probe('timeout', { interface: 'other' })) ?? '', /A VPN on this phone/);
  assert.match(explainProbe(target, probe('timeout')) ?? '', /did not answer on port 7614.*Firewall/);
  assert.match(
    explainProbe({ host: '10.1.2.3', port: 7614 }, probe('failed', { networks: [{ address: '10.1.9.9', netmask: '255.255.0.0' }] })) ?? '',
    /did not answer/,
  );
});
