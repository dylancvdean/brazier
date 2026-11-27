import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEVICE_ID, emptyIndex, hostId, parseIndex, parsePreferences, removeHost, renameHost, selectHost, upsertHost,
} from './hosts.ts';

test('pairing a host adds it and makes it active; pairing it again replaces it', () => {
  const one = upsertHost(emptyIndex, { id: 'a', name: 'Desk' });
  const two = upsertHost(one, { id: 'b', name: 'Laptop' });
  assert.deepEqual(two, { active: 'b', hosts: [{ id: 'a', name: 'Desk' }, { id: 'b', name: 'Laptop' }] });
  assert.deepEqual(upsertHost(two, { id: 'a', name: 'Desk 2' }),
    { active: 'a', hosts: [{ id: 'a', name: 'Desk 2' }, { id: 'b', name: 'Laptop' }] });
});

test('removing the active host falls back to another, and to none when it was the last', () => {
  const index = upsertHost(upsertHost(emptyIndex, { id: 'a', name: 'Desk' }), { id: 'b', name: 'Laptop' });
  assert.equal(removeHost(index, 'b').active, 'a');
  assert.equal(removeHost(index, 'a').active, 'b');
  assert.deepEqual(removeHost(removeHost(index, 'a'), 'b'), emptyIndex);
});

test('rename ignores blank names and select ignores unknown hosts', () => {
  const index = upsertHost(emptyIndex, { id: 'a', name: 'Desk' });
  assert.equal(renameHost(index, 'a', '  ').hosts[0].name, 'Desk');
  assert.equal(renameHost(index, 'a', ' Studio ').hosts[0].name, 'Studio');
  assert.equal(selectHost(index, 'missing').active, 'a');
});

test('a stored index drops malformed and duplicate entries and repairs the active host', () => {
  const raw = JSON.stringify({ active: 'gone', hosts: [{ id: 'a', name: 'Desk' }, { id: 'a', name: 'Dup' }, { id: 3 }, null] });
  assert.deepEqual(parseIndex(raw), { active: 'a', hosts: [{ id: 'a', name: 'Desk' }] });
  assert.deepEqual(parseIndex(null), emptyIndex);
  assert.throws(() => parseIndex('{"hosts":1}'));
});

test('host ids are safe storage keys and preferences fall back to the system theme', () => {
  assert.equal(hostId({ instanceId: 'ab:cd/ef-1' }), 'ab_cd_ef-1');
  assert.deepEqual(parsePreferences('{"theme":"light"}'), { theme: 'light', disabledTools: [], disabledLocalTools: [] });
  assert.deepEqual(parsePreferences('{"theme":"neon","disabledTools":["web_search",3],"disabledLocalTools":["fetch_url",null]}'),
    { theme: 'system', disabledTools: ['web_search'], disabledLocalTools: ['fetch_url'] });
  assert.deepEqual(parsePreferences('not json'), { theme: 'system', disabledTools: [], disabledLocalTools: [] });
});

test('"This device" can be the active entry with or without paired hosts', () => {
  assert.deepEqual(parseIndex(JSON.stringify({ active: DEVICE_ID, hosts: [] })), { active: DEVICE_ID, hosts: [] });
  assert.equal(selectHost(emptyIndex, DEVICE_ID).active, DEVICE_ID);
  const paired = upsertHost(selectHost(emptyIndex, DEVICE_ID), { id: 'a', name: 'A' });
  assert.equal(paired.active, 'a');
  assert.equal(removeHost(selectHost(paired, DEVICE_ID), 'a').active, DEVICE_ID);
  assert.equal(removeHost(paired, 'a').active, null);
});
