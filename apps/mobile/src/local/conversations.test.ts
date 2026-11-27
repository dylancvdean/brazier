import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chatBranch } from '../api.ts';
import { conversationStore, parseConversations, parseMessages } from './conversations.ts';

function memory() {
  const data = new Map<string, string>();
  let next = 0;
  let tick = 0;
  return {
    data,
    store: () => conversationStore({
      read: async (name) => data.get(name) ?? null,
      write: async (name, text) => { data.set(name, text); },
      remove: async (name) => { data.delete(name); },
    }, () => `id-${next += 1}`, () => `2026-03-10T00:00:${String(tick += 1).padStart(2, '0')}.000Z`),
  };
}

test('a conversation and its messages survive a restart', async () => {
  const { store } = memory();
  const first = store();
  const chat = await first.createConversation('Hello there');
  const { user, history } = await first.appendUserMessage(chat.id, [], 'Hello there');
  const answer = await first.createMessage(chat.id, { parent_id: user.id, role: 'assistant', content: 'Hi.', model: 'm' });
  assert.deepEqual(history, [user]);

  const reopened = store();
  assert.deepEqual(await reopened.conversations(), [{ ...chat, updated_at: '2026-03-10T00:00:03.000Z' }]);
  const saved = await reopened.messages(chat.id);
  assert.deepEqual(saved.map(({ role, content, parent_id }) => [role, content, parent_id]),
    [['user', 'Hello there', null], ['assistant', 'Hi.', user.id]]);
  assert.deepEqual(chatBranch(saved).at(-1), answer);
});

test('new conversations are listed first and titles are cut to 72 characters', async () => {
  const { store } = memory();
  const local = store();
  await local.createConversation('first');
  const second = await local.createConversation('é'.repeat(100));
  assert.equal(second.title, 'é'.repeat(72));
  assert.deepEqual((await local.conversations()).map(({ title }) => title.slice(0, 5)), ['ééééé', 'first']);
});

test('messages written at the same time are all kept', async () => {
  const { store } = memory();
  const local = store();
  const chat = await local.createConversation('c');
  await Promise.all(['a', 'b', 'c'].map((content) => local.createMessage(chat.id, { parent_id: null, role: 'user', content })));
  assert.deepEqual((await local.messages(chat.id)).map(({ content }) => content), ['a', 'b', 'c']);
});

test('deleting a conversation removes its messages and refuses later writes', async () => {
  const { store, data } = memory();
  const local = store();
  const chat = await local.createConversation('c');
  await local.appendUserMessage(chat.id, [], 'hi');
  await local.deleteConversation(chat.id);
  assert.deepEqual(await local.conversations(), []);
  assert.deepEqual([...data.keys()], ['conversations.json']);
  await assert.rejects(local.createMessage(chat.id, { parent_id: null, role: 'user', content: 'x' }), /no longer on this device/);
});

test('damaged files read as empty instead of failing', async () => {
  assert.deepEqual(parseConversations('{'), []);
  assert.deepEqual(parseConversations('[{"id":"../x","title":"t"},{"id":"ok","title":"t"},{"id":"ok","title":"again"}]'),
    [{ id: 'ok', title: 't' }]);
  assert.deepEqual(parseMessages('[{"id":"m","role":"user"},{"role":"user"}]'),
    [{ id: 'm', role: 'user', content: '', parent_id: null }]);
  const { store } = memory();
  assert.deepEqual(await store().messages('../../etc'), []);
});

test('the conversation written to last is listed first', async () => {
  const local = memory().store();
  const first = await local.createConversation('First');
  const second = await local.createConversation('Second');
  assert.deepEqual((await local.conversations()).map(({ title }) => title), ['Second', 'First']);
  await local.appendUserMessage(first.id, [], 'more');
  const listed = await local.conversations();
  assert.deepEqual(listed.map(({ title }) => title), ['First', 'Second']);
  assert.ok(listed[0]!.updated_at! > second.updated_at!);
});

test('a conversation is renamed in place and keeps its messages', async () => {
  const { store } = memory();
  const local = store();
  const first = await local.createConversation('First');
  await local.appendUserMessage(first.id, [], 'hello');
  await local.createConversation('Second');
  const renamed = await local.renameConversation(first.id, `  Trip\nplans ${'x'.repeat(100)}`);
  assert.equal(renamed.title, `Trip plans ${'x'.repeat(61)}`);
  assert.equal(renamed.id, first.id);
  // Renaming is not writing to it: the order stays.
  assert.deepEqual((await store().conversations()).map(({ title }) => title), ['Second', renamed.title]);
  assert.equal((await local.messages(first.id)).length, 1);
  await assert.rejects(local.renameConversation(first.id, '   '), /Enter a title/);
  await assert.rejects(local.renameConversation('missing', 'x'), /no longer on this device/);
});

test('conversations are searched by title and by what was said in them', async () => {
  const local = memory().store();
  const trip = await local.createConversation('Trip plans');
  const { user } = await local.appendUserMessage(trip.id, [], 'Where should we stay in Lisbon?');
  await local.createMessage(trip.id, { parent_id: user.id, role: 'assistant', content: [{ type: 'text', text: 'Try Alfama.' }] });
  const code = await local.createConversation('Rust lifetimes');
  await local.appendUserMessage(code.id, [], 'Explain the borrow checker');
  const titles = async (query: string) => (await local.conversations(query)).map(({ title }) => title);
  assert.deepEqual(await titles('trip'), ['Trip plans']);
  assert.deepEqual(await titles('LISBON stay'), ['Trip plans']);
  assert.deepEqual(await titles('alfama'), ['Trip plans']);
  assert.deepEqual(await titles('borrow'), ['Rust lifetimes']);
  // Every word must be in the title or in one message, and the structure around a message is not text.
  assert.deepEqual(await titles('lisbon borrow'), []);
  assert.deepEqual(await titles('text'), []);
  assert.deepEqual(await titles('   '), ['Rust lifetimes', 'Trip plans']);
});

test('a message is saved even when its conversation cannot be moved to the top', async () => {
  const data = new Map<string, string>();
  let failIndex = false;
  let next = 0;
  const local = conversationStore({
    read: async (name) => data.get(name) ?? null,
    write: async (name, text) => {
      if (failIndex && name === 'conversations.json') throw new Error('disk full');
      data.set(name, text);
    },
    remove: async (name) => { data.delete(name); },
  }, () => `id-${next += 1}`);
  const chat = await local.createConversation('Chat');
  failIndex = true;
  const { user } = await local.appendUserMessage(chat.id, [], 'hello');
  assert.deepEqual((await local.messages(chat.id)).map(({ id }) => id), [user.id]);
});
