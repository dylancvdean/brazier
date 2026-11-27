import assert from 'node:assert/strict';
import test from 'node:test';
import { backStep, describeScopes, keyboardInset } from './navigation.ts';

const paired = { agentShown: false, onDevice: false, hasHost: true };

test('back from a secondary screen returns to the screen that opened it', () => {
  assert.deepEqual(backStep({ ...paired, view: 'settings' }), { view: 'chat' });
  assert.deepEqual(backStep({ ...paired, view: 'generate' }), { view: 'chat' });
  assert.deepEqual(backStep({ ...paired, view: 'models' }), { view: 'settings' });
  assert.deepEqual(backStep({ ...paired, view: 'models', onDevice: true }), { view: 'chat' });
  assert.deepEqual(backStep({ ...paired, view: 'pairing' }), { view: 'settings' });
  assert.deepEqual(backStep({ ...paired, hasHost: false, onDevice: true, view: 'pairing' }), { view: 'settings' });
});

test('back closes the agent before it leaves the app', () => {
  assert.deepEqual(backStep({ ...paired, view: 'chat', agentShown: true }), { agent: 'hidden' });
  assert.equal(backStep({ ...paired, view: 'chat' }), null);
  // Settings opened over the agent returns to the agent, which stays shown.
  assert.deepEqual(backStep({ ...paired, view: 'settings', agentShown: true }), { view: 'chat' });
});

test('back leaves the app from the root screens', () => {
  assert.equal(backStep({ view: 'chat', agentShown: false, onDevice: true, hasHost: false }), null);
  assert.equal(backStep({ view: 'chat', agentShown: true, onDevice: true, hasHost: false }), null);
  assert.equal(backStep({ view: 'chat', agentShown: false, onDevice: false, hasHost: false }), null);
  assert.equal(backStep({ view: 'pairing', agentShown: false, onDevice: false, hasHost: false }), null);
  assert.equal(backStep({ view: 'settings', agentShown: false, onDevice: false, hasHost: false }), null);
});

test('invite access is described in the app’s words', () => {
  assert.equal(describeScopes(['inference', 'agent']), 'Chat and generation, Agent tasks');
  assert.equal(describeScopes(['inference', 'future']), 'Chat and generation, future');
  assert.equal(describeScopes([]), 'None');
});

test('the keyboard inset covers the keyboard and never drops below the resting inset', () => {
  assert.equal(keyboardInset(914, null, 26), 26);
  assert.equal(keyboardInset(914, 577.9, 26), 336);
  assert.equal(keyboardInset(914, 910, 26), 26);
});
