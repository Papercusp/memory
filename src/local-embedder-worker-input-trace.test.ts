import { afterEach, describe, expect, it, vi } from 'vitest';
import { Worker } from 'node:worker_threads';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { embedViaWorker, getWorkerState, shutdownLocalEmbedder, _resetBeforeExitHookForTest,
  type WorkerInputTrace } from './local-embedder-worker';

const trace: WorkerInputTrace = { model: '/private/model', device: 'cpu', observedAt: '2026-10-02T00:00:00.000Z',
  inputShape: [1, 2], inputIds: [1, 2], attentionMask: [1, 1] };
afterEach(async () => { vi.restoreAllMocks(); await shutdownLocalEmbedder(); _resetBeforeExitHookForTest(); });

describe('file worker boot from a stdin parent', () => {
  it.each([['--input-type=module'], ['--input-type', 'module']])('does not inherit stdin-only flags %j', (...inputArgs) => {
    const moduleUrl = new URL('./local-embedder-worker.ts', import.meta.url).href;
    const scriptUrl = new URL('./local-embedder-worker.script.mjs', import.meta.url).href;
    const program = `
import { Worker } from 'node:worker_threads';
const control = new Worker(new URL(${JSON.stringify(scriptUrl)}), { workerData: { device: 'cpu' } });
const controlCode = await new Promise(resolve => control.once('error', error => resolve(error.code)));
await control.terminate();
if (controlCode !== 'ERR_INPUT_TYPE_NOT_ALLOWED') throw new Error('stdin flag control did not distinguish file-worker boot');
const imported = await import(${JSON.stringify(moduleUrl)}), bridge = imported.default ?? imported;
Worker.prototype.postMessage = function (msg) {
  setImmediate(() => this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1, 0] }));
};
try {
  const vector = await bridge.embedViaWorker('worker boot only; no model request is forwarded');
  if (!bridge.getWorkerState().alive) throw new Error('shipped file worker did not become ready');
  console.log(JSON.stringify({ controlCode, vector, ready: true, modelRequestForwarded: false }));
} finally { await bridge.shutdownLocalEmbedder(); bridge._resetBeforeExitHookForTest(); }
`;
    const child = spawnSync(process.execPath, ['--import', import.meta.resolve('tsx/esm'), ...inputArgs], {
      input: program, encoding: 'utf8', timeout: 20000,
      env: { ...process.env, PAPERCUSP_EMBED_DEVICE: 'cpu', PAPERCUSP_FORBID_REAL_PG: '1' },
    });
    expect(child.status, child.stderr).toBe(0);
    expect(JSON.parse(child.stdout.trim())).toEqual({ controlCode: 'ERR_INPUT_TYPE_NOT_ALLOWED', vector: [1, 0], ready: true, modelRequestForwarded: false });
  });
});

describe('worker input evidence protocol', () => {
  it('delivers evidence without settling or draining the in-flight vector request', async () => {
    const observed: WorkerInputTrace[] = [];
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
      expect(msg.traceInput).toBe(true);
      setImmediate(() => {
        this.emit('message', { kind: 'embed_input', id: msg.id, trace });
        expect(getWorkerState().pendingCount).toBe(1);
        this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [0.25, 0.5] });
      });
    });
    expect(await embedViaWorker('q', { tokenizerBackend: 'rust', onInputTrace: (row) => observed.push(row) })).toEqual([0.25, 0.5]);
    expect(observed).toEqual([trace]); expect(getWorkerState().pendingCount).toBe(0);
  });
  it.each(['missing', 'shape', 'mask', 'ids', 'callback'])('refuses a successful vector with %s requested evidence', async (failure) => {
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
      setImmediate(() => {
        const row = structuredClone(trace);
        if (failure === 'shape') row.inputShape = [2, 2];
        if (failure === 'mask') row.attentionMask = [1];
        if (failure === 'ids') row.inputIds[0] = -1;
        if (failure !== 'missing') this.emit('message', { kind: 'embed_input', id: msg.id, trace: row });
        this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1] });
      });
    });
    await expect(embedViaWorker('q', { onInputTrace: () => {
      if (failure === 'callback') throw new Error('evidence persistence failed');
    } })).rejects.toThrow(failure === 'missing' ? 'without requested input evidence' : failure === 'callback' ? 'persistence failed' : 'invalid worker input trace');
    expect(getWorkerState().pendingCount).toBe(0);
  });
  it('retains pre-inference evidence when the graph fails and preserves that cause', async () => {
    const observed: WorkerInputTrace[] = [];
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
      setImmediate(() => {
        this.emit('message', { kind: 'embed_input', id: msg.id, trace });
        this.emit('message', { kind: 'embed_err', id: msg.id, error: 'graph allocation failed' });
      });
    });
    await expect(embedViaWorker('q', { onInputTrace: (row) => observed.push(row) })).rejects.toThrow('graph allocation failed');
    expect(observed).toEqual([trace]);
  });
  it('does not request raw input evidence on ordinary calls', async () => {
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
      expect(msg).not.toHaveProperty('traceInput');
      setImmediate(() => this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1] }));
    });
    expect(await embedViaWorker('q')).toEqual([1]);
  });
});

describe('real worker submitted tensors', () => {
  it.each([false, true])('emits exact truncated tensors before graph success/failure=%s', async (fail) => {
    const dir = mkdtempSync(join(tmpdir(), 'worker-input-evidence-'));
    const fake = join(dir, 'fake-transformers.mjs');
    writeFileSync(join(dir, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 3 }));
    writeFileSync(join(dir, 'tokenizer.json'), JSON.stringify({ version: '1.0', truncation: null, padding: null,
      added_tokens: [], normalizer: null, pre_tokenizer: { type: 'Whitespace' }, post_processor: null, decoder: null,
      model: { type: 'WordLevel', vocab: { '[UNK]': 0, a: 1, b: 2, c: 3, d: 4 }, unk_token: '[UNK]' } }));
    writeFileSync(fake, `export const env = {};
export class Tensor { constructor(type, data, dims) { this.type=type; this.data=data; this.dims=dims; } }
export async function pipeline() { return { model: async inputs => {
  if (JSON.stringify(Array.from(inputs.input_ids.data, Number)) !== '[1,2,3]') throw new Error('wrong submitted ids');
  if (${fail}) throw new Error('graph rejected exact tensors');
  return { last_hidden_state: { slice: () => ({ normalize: () => ({ data: [1,0] }) }) } };
} }; }
`);
    const worker = new Worker(new URL('./local-embedder-worker.script.mjs', import.meta.url), {
      workerData: { device: 'cpu', transformersSpecifier: pathToFileURL(fake).href },
    });
    const messages: any[] = [];
    try {
      await new Promise<void>((resolve, reject) => { worker.once('error', reject); worker.on('message', m => {
        messages.push(m); if (m.kind === 'ready') resolve();
      }); });
      const done = new Promise<any>((resolve, reject) => { worker.once('error', reject); worker.on('message', m => {
        if (m.id === 7 && ['embed_ok', 'embed_err'].includes(m.kind)) resolve(m);
      }); });
      worker.postMessage({ kind: 'embed', id: 7, text: 'a b c d', model: dir, tokenizerBackend: 'rust', pooling: 'cls', traceInput: true });
      const result = await done, input = messages.find(m => m.kind === 'embed_input');
      expect(input).toMatchObject({ id: 7, trace: { model: dir, device: 'cpu', inputShape: [1,3], inputIds: [1,2,3], attentionMask: [1,1,1] } });
      expect(messages.indexOf(input)).toBeLessThan(messages.indexOf(result));
      expect(result).toMatchObject(fail ? { kind: 'embed_err', error: 'graph rejected exact tensors' } : { kind: 'embed_ok', vector: [1,0] });
    } finally { await worker.terminate(); rmSync(dir, { recursive: true, force: true }); }
  });
});
