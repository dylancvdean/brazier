import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chatBranch, toolCallsOf } from '../api.ts';
import { conversationStore, parseConversations, parseMessages } from './conversations.ts';

function memory() {
  const data = new Map<string, string>();
  let next = 0;
  return {
    data,
    store: () => conversationStore({
      read: async (name) => data.get(name) ?? null,
      write: async (name, text) => { data.set(name, text); },
      remove: async (name) => { data.delete(name); },
    }, () => `id-${next += 1}`),
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
  assert.deepEqual(await reopened.conversations(), [chat]);
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

test('tool rounds are saved with their calls and results', async () => {
  const { store } = memory();
  const chat = await store().createConversation('Search');
  const calls = [{ id: 'c1', type: 'function', function: { name: 'web_search', arguments: '{"query":"x"}' } }];
  const round = await store().createMessage(chat.id, { parent_id: null, role: 'assistant', content: '', tool_calls: calls });
  await store().createMessage(chat.id, { parent_id: round.id, role: 'tool', content: 'hits', tool_call_id: 'c1' });
  await store().createMessage(chat.id, { parent_id: null, role: 'assistant', content: 'Plain', tool_calls: null, tool_call_id: null });
  const saved = await store().messages(chat.id);
  assert.deepEqual(saved.map(({ id, parent_id, ...rest }) => rest), [
    { role: 'assistant', content: '', tool_calls: calls },
    { role: 'tool', content: 'hits', tool_call_id: 'c1' },
    { role: 'assistant', content: 'Plain' },
  ]);
  assert.equal(toolCallsOf(saved).get(round.id)?.[0].output, 'hits');
});
