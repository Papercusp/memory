import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, expect, it } from 'vitest';
import { resolveLocalEmbedEngine } from './local-embedder-worker';

// WI-6554700010050711: the embed engine's availability is answered by module
// resolution, never by evaluating the package. The fixture's entry points THROW
// when evaluated, so a resolvable verdict proves neither package ran.
const root = realpathSync(mkdtempSync(join(tmpdir(), 'wi0711-embed-')));
afterAll(() => rmSync(root, { recursive: true, force: true }));
for (const name of ['@huggingface/transformers', 'onnxruntime-node']) {
  const dir = join(root, 'node_modules', ...name.split('/'));
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name, version: '0.0.0', main: 'index.js' }));
  writeFileSync(join(dir, 'index.js'), `throw new Error(${JSON.stringify(`${name} was evaluated`)});\n`);
}

it('reports an installed engine resolvable from its base without evaluating either package', () => {
  const base = join(root, 'probe.mjs');
  expect(resolveLocalEmbedEngine(base)).toEqual({
    status: 'resolvable',
    base,
    transformersEntry: join(root, 'node_modules', '@huggingface', 'transformers', 'index.js'),
    onnxRuntimeEntry: join(root, 'node_modules', 'onnxruntime-node', 'index.js'),
  });
});

it('reports a missing install unavailable, naming the base it resolved from', () => {
  const base = '/wi0711-no-engine/x.mjs';
  expect(resolveLocalEmbedEngine(base)).toEqual({
    status: 'unavailable',
    base,
    reason: `@huggingface/transformers is not resolvable from ${base}`,
  });
});

it('resolves from the embed worker script by default, where this repo installs the engine', () => {
  expect(resolveLocalEmbedEngine().status).toBe('resolvable');
});
