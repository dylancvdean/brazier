import assert from 'node:assert/strict';
import test from 'node:test';
import { parseInline, parseMarkdown } from './markdown.ts';

test('blocks: headings, lists, quotes, rules and paragraphs', () => {
  assert.deepEqual(parseMarkdown('# Title\n\nfirst\nsecond\n\n- one\n  2) two\n> quoted\n---'), [
    { type: 'heading', level: 1, children: [{ type: 'text', text: 'Title' }] },
    { type: 'paragraph', children: [{ type: 'text', text: 'first\nsecond' }] },
    { type: 'item', marker: '•', depth: 0, children: [{ type: 'text', text: 'one' }] },
    { type: 'item', marker: '2.', depth: 1, children: [{ type: 'text', text: 'two' }] },
    { type: 'quote', children: [{ type: 'text', text: 'quoted' }] },
    { type: 'rule' },
  ]);
});

test('code fences keep their text verbatim, and an unclosed fence runs to the end', () => {
  assert.deepEqual(parseMarkdown('```ts\nconst a = **1**;\n```\nafter'), [
    { type: 'code', language: 'ts', text: 'const a = **1**;' },
    { type: 'paragraph', children: [{ type: 'text', text: 'after' }] },
  ]);
  assert.deepEqual(parseMarkdown('```\n# not a heading'), [{ type: 'code', language: '', text: '# not a heading' }]);
});

test('inline: code, emphasis and links; unsafe link schemes stay text', () => {
  assert.deepEqual(parseInline('a `b*c` **bold _it_** [x](https://e.com) [y](javascript:1)'), [
    { type: 'text', text: 'a ' },
    { type: 'code', text: 'b*c' },
    { type: 'text', text: ' ' },
    { type: 'strong', children: [{ type: 'text', text: 'bold ' }, { type: 'em', children: [{ type: 'text', text: 'it' }] }] },
    { type: 'text', text: ' ' },
    { type: 'link', href: 'https://e.com', children: [{ type: 'text', text: 'x' }] },
    { type: 'text', text: ' ' },
    { type: 'text', text: '[y](javascript:1)' },
  ]);
  assert.deepEqual(parseInline('snake_case_name and 2 * 3 * 4'), [{ type: 'text', text: 'snake_case_name and 2 * 3 * 4' }]);
});

test('links keep balanced parentheses, fences ignore inner openers, and headings keep a trailing #', () => {
  assert.deepEqual(parseInline('[w](https://e.com/a_(b))'), [
    { type: 'link', href: 'https://e.com/a_(b)', children: [{ type: 'text', text: 'w' }] },
  ]);
  assert.deepEqual(parseMarkdown('````md\n```js\nx\n```\n````'), [{ type: 'code', language: 'md', text: '```js\nx\n```' }]);
  assert.deepEqual(parseMarkdown('# C#\n## Closed ##'), [
    { type: 'heading', level: 1, children: [{ type: 'text', text: 'C#' }] },
    { type: 'heading', level: 2, children: [{ type: 'text', text: 'Closed' }] },
  ]);
});
