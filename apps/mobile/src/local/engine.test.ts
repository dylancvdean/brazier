import assert from 'node:assert/strict';
import { test } from 'node:test';
import { answerDelta, chatTurns, localEngine, toolTurns, type Runtime } from './engine.ts';
import { addInstalled, installedFrom, modelId, parseInstalled, removeInstalled, type InstalledModel } from './models.ts';

const model = (id: string, runtime: InstalledModel['runtime']): InstalledModel =>
  ({ id, slug: 'o/n', runtime, name: id, entry: 'f', files: ['f'], size: 1 });

function fake(id: Runtime['id'], log: string[], options: { available?: boolean; builtIn?: string } = {}): Runtime {
  return {
    id,
    available: async () => options.available ?? true,
    ...(options.builtIn ? { builtIn: async () => [{ id: options.builtIn!, name: 'System model' }] } : {}),
    load: async (loaded) => { log.push(`${id} load ${loaded?.id ?? 'built-in'}`); },
    generate: async (messages, _signal, onToken) => {
      log.push(`${id} generate ${messages.map(({ role }) => role).join(',')}`);
      onToken('Hi');
      return 'Hi';
    },
    unload: async () => { log.push(`${id} unload`); },
  };
}

test('models are offered only for runtimes this device can run', async () => {
  const log: string[] = [];
  const installed = [model('gguf', 'llama.cpp'), model('mlx', 'mlx')];
  const engine = localEngine([
    fake('apple', log, { builtIn: 'apple-intelligence' }), fake('llama.cpp', log), fake('mlx', log, { available: false }),
  ], async () => installed);
  assert.deepEqual((await engine.models()).map(({ id }) => id), ['apple-intelligence', 'gguf']);
  await assert.rejects(engine.prepare('mlx'), /cannot run mlx/);
  await assert.rejects(engine.prepare('missing'), /not on this device/);

  const without = localEngine([fake('apple', log, { builtIn: 'apple-intelligence', available: false }), fake('llama.cpp', log)],
    async () => installed);
  assert.deepEqual((await without.models()).map(({ id }) => id), ['gguf']);
  await assert.rejects(without.prepare('apple-intelligence'), /not on this device/);
});

test('each model goes to its runtime, with one model loaded at a time', async () => {
  const log: string[] = [];
  const engine = localEngine([fake('apple', log, { builtIn: 'apple-intelligence' }), fake('llama.cpp', log), fake('mlx', log)],
    async () => [model('gguf', 'llama.cpp'), model('mlx', 'mlx')]);
  const signal = new AbortController().signal;
  const tokens: string[] = [];
  await engine.prepare('gguf');
  await engine.prepare('gguf');
  assert.equal(await engine.chat('gguf', [{ role: 'user', content: 'hi' }], signal, (token) => tokens.push(token)), 'Hi');
  await engine.chat('mlx', [{ role: 'user', content: 'hi' }], signal, { onToken: () => undefined });
  await engine.chat('apple-intelligence', [{ role: 'user', content: 'hi' }], signal, () => undefined);
  await engine.unload('gguf');
  await engine.unload('apple-intelligence');
  await engine.unload();
  assert.deepEqual(tokens, ['Hi']);
  assert.deepEqual(log, [
    'llama.cpp load gguf', 'llama.cpp generate user',
    'llama.cpp unload', 'mlx load mlx', 'mlx generate user',
    'mlx unload', 'apple load built-in', 'apple generate user', 'apple unload',
  ]);
});

test('a stopped reply is not returned', async () => {
  const turn = new AbortController();
  const runtime: Runtime = {
    ...fake('llama.cpp', []),
    generate: async () => { turn.abort(); return 'partial'; },
  };
  const engine = localEngine([runtime], async () => [model('gguf', 'llama.cpp')]);
  await assert.rejects(engine.chat('gguf', [{ role: 'user', content: 'hi' }], turn.signal, () => undefined), /Chat stopped/);
});

test('only text turns reach an on-device model', () => {
  assert.deepEqual(chatTurns([
    { role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hi' },
    { role: 'assistant', content: '', tool_calls: [{}] }, { role: 'tool', content: '{}', tool_call_id: 'c' },
    { role: 'assistant', content: '  ' }, { role: 'assistant', content: 'Hello.' },
  ]), [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hi' }, { role: 'assistant', content: 'Hello.' }]);
});

test('a prompt left unanswered by Stop is sent with the next one, keeping the sides alternating', () => {
  assert.deepEqual(chatTurns([
    { role: 'user', content: 'first' }, { role: 'user', content: 'second' }, { role: 'assistant', content: 'ok' },
    { role: 'assistant', content: '', tool_calls: [{}] }, { role: 'user', content: 'third' },
  ]), [{ role: 'user', content: 'first\n\nsecond' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'third' }]);
});

test('streamed text leaves out the reasoning of a thinking model', () => {
  assert.equal(answerDelta('', { token: 'Hel' }), 'Hel');
  assert.equal(answerDelta('', { token: '<think>', accumulated_text: '<think>' }), '');
  assert.equal(answerDelta('', { token: 'hmm', reasoning_content: 'hmm', accumulated_text: '<think>hmm' }), '');
  assert.equal(answerDelta('', { token: 'Hel', content: 'Hel', reasoning_content: 'hmm' }), 'Hel');
  assert.equal(answerDelta('Hel', { token: 'lo', content: 'Hello' }), 'lo');
  assert.equal(answerDelta('Hello', { token: '', content: 'Hello' }), '');
});

test('the installed list keeps one entry per model and drops damaged ones', () => {
  const choice = { runtime: 'llama.cpp' as const, name: 'm-Q4.gguf', entry: 'dir/m-Q4.gguf', files: [{ path: 'dir/m-Q4.gguf', size: 9 }], size: 9 };
  const installed = installedFrom('Owner/Repo', choice);
  assert.equal(installed.id, 'gguf--Owner--Repo--dir_m-Q4.gguf');
  assert.equal(modelId('o/n', { runtime: 'mlx', entry: '' }), 'mlx--o--n');
  assert.equal(modelId('o/n', { runtime: 'sd.cpp', entry: 'sd.gguf' }), 'sd--o--n--sd.gguf');
  assert.equal(installed.name, 'm-Q4');
  assert.equal(installedFrom('mlx-community/Tiny-4bit', { ...choice, runtime: 'mlx', entry: '' }).name, 'Tiny-4bit');
  const list = addInstalled(addInstalled([], installed), { ...installed, size: 10 });
  assert.deepEqual(list.map(({ size }) => size), [10]);
  assert.deepEqual(parseInstalled(JSON.stringify(list)), list);
  assert.deepEqual(removeInstalled(list, installed.id), []);
  assert.deepEqual(parseInstalled('nonsense'), []);
  assert.deepEqual(parseInstalled(JSON.stringify([
    { ...installed, id: '../x' }, { ...installed, runtime: 'apple' }, { ...installed, files: ['../../x'] },
  ])), []);
});

function toolRuntime(rounds: { content: string; calls?: { name: string; arguments: string }[] }[], seen: { roles: string; tools: number }[]): Runtime {
  return {
    ...fake('llama.cpp', []),
    generateWithTools: async (messages, tools, _signal, onToken) => {
      seen.push({ roles: messages.map(({ role }) => role).join(','), tools: tools.length });
      const round = rounds.shift()!;
      if (round.content) onToken(round.content);
      return { content: round.content, calls: round.calls ?? [] };
    },
  };
}
const localTools = (ran: string[]) => ({
  definitions: (names: readonly string[]) => names.map((name) => ({ type: 'function', function: { name } })),
  run: async (name: string, args: string) => {
    ran.push(`${name} ${args}`);
    return name === 'fetch_url' ? { output: 'Error: HTTP 500', is_error: true } : { output: 'x'.repeat(5000), is_error: false };
  },
});

test('a reply runs the tools the model calls and reports each round as a host does', async () => {
  const seen: { roles: string; tools: number }[] = [];
  const ran: string[] = [];
  const engine = localEngine([toolRuntime([
    { content: 'Looking.', calls: [{ name: 'web_search', arguments: '{"query":"tea"}' }, { name: 'fetch_url', arguments: '' }] },
    { content: '', calls: [{ name: 'get_current_time', arguments: '{}' }] },
    { content: 'Done.' },
  ], seen)], async () => [model('gguf', 'llama.cpp')], localTools(ran));
  assert.equal(await engine.supportsTools('gguf'), true);
  const events: string[] = [];
  const ids: string[] = [];
  const answer = await engine.chat('gguf', [{ role: 'user', content: 'Hello' }], new AbortController().signal, {
    onToken: (token) => events.push(`token ${token}`),
    onToolCall: (call) => { ids.push(call.call_id); events.push(`call ${call.name} ${call.is_error} ${call.output.length}`); },
    onTranscript: (message) => events.push(`${message.role} ${message.tool_call_id ? 'result' : (message.tool_calls ?? []).length}`),
  }, ['web_search', 'fetch_url']);
  assert.equal(answer, 'Done.');
  assert.deepEqual(ran, ['web_search {"query":"tea"}', 'fetch_url {}']);
  assert.equal(ids.length, 3);
  assert.equal(new Set(ids).size, 3);
  // Mistral's chat template accepts nothing but nine letters and digits as a call id.
  for (const id of ids) assert.match(id, /^[a-z0-9]{9}$/);
  assert.deepEqual(seen, [
    { roles: 'user', tools: 2 }, { roles: 'user,assistant,tool,tool', tools: 2 },
    { roles: 'user,assistant,tool,tool,assistant,tool', tools: 2 },
  ]);
  assert.deepEqual(events, [
    'token Looking.', 'assistant 2', 'call web_search false 4013', 'tool result', 'call fetch_url true 15', 'tool result',
    // A tool that is turned off is refused without running.
    'assistant 1', 'call get_current_time true 50', 'tool result', 'token Done.',
  ]);
});

test('the last round is asked without tools, and no tools means a plain chat', async () => {
  const seen: { roles: string; tools: number }[] = [];
  const call = { content: '', calls: [{ name: 'web_search', arguments: '{}' }] };
  const engine = localEngine([toolRuntime([call, call, call, { content: 'Answer.', calls: [call.calls[0]] }], seen)],
    async () => [model('gguf', 'llama.cpp')], localTools([]));
  assert.equal(await engine.chat('gguf', [{ role: 'user', content: 'Hi' }], new AbortController().signal, () => undefined, ['web_search']), 'Answer.');
  assert.deepEqual(seen.map(({ tools }) => tools), [1, 1, 1, 0]);
  assert.equal(await engine.chat('gguf', [{ role: 'user', content: 'Hi' }], new AbortController().signal, () => undefined), 'Hi');
  // A runtime that cannot call tools chats without them.
  const log: string[] = [];
  const plain = localEngine([fake('mlx', log)], async () => [model('m', 'mlx')], localTools([]));
  assert.equal(await plain.supportsTools('m'), false);
  assert.equal(await plain.chat('m', [{ role: 'user', content: 'Hi' }], new AbortController().signal, () => undefined, ['web_search']), 'Hi');
});

test('saved tool rounds are sent back whole, and broken ones are left out', () => {
  const call = (id: string) => ({ id, type: 'function', function: { name: 'web_search', arguments: { query: 'x' } } });
  assert.deepEqual(toolTurns([
    { role: 'user', content: 'One' },
    { role: 'assistant', content: '', tool_calls: [call('a')] },
    { role: 'tool', content: 'result', tool_call_id: 'a' },
    { role: 'assistant', content: 'Answer' },
    { role: 'user', content: 'Two' },
    { role: 'assistant', content: 'Partly', tool_calls: [call('b')] },
    { role: 'user', content: 'Three' },
  ]), [
    { role: 'user', content: 'One' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'a', type: 'function', function: { name: 'web_search', arguments: '{"query":"x"}' } }] },
    { role: 'tool', content: 'result', tool_call_id: 'a' },
    { role: 'assistant', content: 'Answer' },
    { role: 'user', content: 'Two' },
    { role: 'assistant', content: 'Partly' },
    { role: 'user', content: 'Three' },
  ]);
  // A page read in an earlier turn is cut short; one read for the question being answered is whole.
  const page = 'p'.repeat(2000);
  const [, , old, , , , current] = toolTurns([
    { role: 'user', content: 'One' },
    { role: 'assistant', content: '', tool_calls: [call('a')] },
    { role: 'tool', content: page, tool_call_id: 'a' },
    { role: 'assistant', content: 'Answer' },
    { role: 'user', content: 'Two' },
    { role: 'assistant', content: '', tool_calls: [call('b')] },
    { role: 'tool', content: page, tool_call_id: 'b' },
  ]);
  assert.equal(old.content, `${'p'.repeat(500)}\n[Cut short.]`);
  assert.equal(current.content, page);
});
