import { afterEach, describe, expect, it, vi } from 'vitest';
import { Worker } from 'node:worker_threads';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { embedViaWorker, getWorkerState, shutdownLocalEmbedder, _resetBeforeExitHookForTest,
  type WorkerInputTrace, type WorkerInferenceTrace } from './local-embedder-worker';

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
      expect(msg).not.toHaveProperty('traceInference');
      setImmediate(() => this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1] }));
    });
    expect(await embedViaWorker('q')).toEqual([1]);
  });
  it.each(['missing', 'end-only', 'request', 'thread', 'clock', 'reverse', 'unfinished', 'callback'])
    ('refuses a successful vector with %s graph lifecycle evidence', async (failure) => {
      vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
        expect(msg.traceInference).toBe(true);
        setImmediate(() => {
          const start: WorkerInferenceTrace = { requestId: msg.id, attempt: 1, processId: 123, workerThreadId: 1,
            nativeThreadId: 456, model: trace.model, device: 'cpu', observedAt: trace.observedAt,
            clock: 'node-hrtime', monotonicNs: '100', phase: 'start' };
          const end: WorkerInferenceTrace = { ...start, monotonicNs: '200', phase: 'end', outcome: 'success' };
          if (failure === 'request') start.requestId++;
          if (failure === 'thread') end.nativeThreadId++;
          if (failure === 'clock') start.monotonicNs = 'unknown';
          if (failure === 'reverse') end.monotonicNs = '99';
          if (!['missing', 'end-only'].includes(failure)) this.emit('message', { kind: 'embed_inference', id: msg.id, inference: start });
          if (!['missing', 'unfinished'].includes(failure)) this.emit('message', { kind: 'embed_inference', id: msg.id, inference: end });
          this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1] });
        });
      });
      await expect(embedViaWorker('q', { onInferenceTrace: () => {
        if (failure === 'callback') throw new Error('lifecycle persistence failed');
      } })).rejects.toThrow(failure === 'callback' ? 'persistence failed'
        : ['missing', 'unfinished'].includes(failure) ? 'without complete requested inference evidence' : 'invalid worker inference trace');
      expect(getWorkerState().pendingCount).toBe(0);
    });
  it('keeps a failed CUDA attempt separate from its CPU fallback', async () => {
    const observed: WorkerInferenceTrace[] = [];
    vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
      setImmediate(() => {
        const identity = { requestId: msg.id, processId: 123, workerThreadId: 1, nativeThreadId: 456,
          model: trace.model, observedAt: trace.observedAt, clock: 'node-hrtime' as const };
        for (const [attempt, device, outcome] of [[1, 'cuda', 'error'], [2, 'cpu', 'success']] as const) {
          for (const phase of ['start', 'end'] as const) this.emit('message', { kind: 'embed_inference', id: msg.id,
            inference: { ...identity, attempt, device, phase, monotonicNs: String(attempt * 100 + (phase === 'end' ? 50 : 0)),
              ...(phase === 'end' ? { outcome } : {}) } });
        }
        this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1] });
      });
    });
    expect(await embedViaWorker('q', { onInferenceTrace: (row) => observed.push(row) })).toEqual([1]);
    expect(observed.map(t => [t.attempt, t.device, t.phase, t.outcome])).toEqual([
      [1, 'cuda', 'start', undefined], [1, 'cuda', 'end', 'error'],
      [2, 'cpu', 'start', undefined], [2, 'cpu', 'end', 'success'],
    ]);
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
  const ids = Array.from(inputs.input_ids.data, Number);
  if (ids.length !== 3) throw new Error('wrong submitted ids');
  await new Promise(resolve => setTimeout(resolve, ids[0] === 1 ? 30 : 0));
  if (${fail}) throw new Error('graph rejected exact tensors');
  return { last_hidden_state: { slice: () => ({ normalize: () => ({ data: [ids[0],0] }) }) } };
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
      worker.postMessage({ kind: 'embed', id: 7, text: 'a b c d', model: dir, tokenizerBackend: 'rust', pooling: 'cls', traceInput: true, traceInference: true });
      const result = await done, input = messages.find(m => m.kind === 'embed_input');
      expect(input).toMatchObject({ id: 7, trace: { model: dir, device: 'cpu', inputShape: [1,3], inputIds: [1,2,3], attentionMask: [1,1,1] } });
      expect(messages.indexOf(input)).toBeLessThan(messages.indexOf(result));
      expect(result).toMatchObject(fail ? { kind: 'embed_err', error: 'graph rejected exact tensors' } : { kind: 'embed_ok', vector: [1,0] });
      const lifecycle = messages.filter(m => m.kind === 'embed_inference');
      expect(input.trace.request).toMatchObject({ requestId: 7, attempt: 1, processId: process.pid, workerThreadId: worker.threadId });
      if (process.platform === 'linux') expect(input.trace.request.nativeThreadId).toBeGreaterThan(0);
      else expect(input.trace.request.nativeThreadId).toBeNull();
      expect(lifecycle.map(m => [m.inference.phase, m.inference.outcome])).toEqual([
        ['start', undefined], ['end', fail ? 'error' : 'success'],
      ]);
      expect(lifecycle[0].inference).toMatchObject({ ...input.trace.request, clock: 'node-hrtime' });
      expect(BigInt(lifecycle[1].inference.monotonicNs)).toBeGreaterThan(BigInt(lifecycle[0].inference.monotonicNs));
      expect(messages.indexOf(input)).toBeLessThan(messages.indexOf(lifecycle[0]));
      expect(messages.indexOf(lifecycle[1])).toBeLessThan(messages.indexOf(result));
      if (!fail) {
        const concurrent = new Promise<any[]>((resolve, reject) => {
          const results: any[] = [];
          worker.once('error', reject); worker.on('message', m => {
            if ([8, 9].includes(m.id) && m.kind === 'embed_ok') { results.push(m); if (results.length === 2) resolve(results); }
          });
        });
        for (const [id, text] of [[8, 'a b c d'], [9, 'b a c d']]) worker.postMessage({
          kind: 'embed', id, text, model: dir, tokenizerBackend: 'rust', pooling: 'cls', traceInput: true, traceInference: true,
        });
        expect((await concurrent).map(m => [m.id, m.vector])).toEqual([[9, [2,0]], [8, [1,0]]]);
        for (const id of [8, 9]) {
          const events = messages.filter(m => m.kind === 'embed_inference' && m.id === id);
          expect(events.map(m => [m.inference.requestId, m.inference.phase])).toEqual([[id, 'start'], [id, 'end']]);
        }
        const starts = messages.filter(m => m.kind === 'embed_inference' && [8, 9].includes(m.id) && m.inference.phase === 'start');
        const firstEnd = messages.find(m => m.kind === 'embed_inference' && [8, 9].includes(m.id) && m.inference.phase === 'end');
        expect(starts).toHaveLength(2);
        expect(BigInt(starts[1].inference.monotonicNs)).toBeLessThan(BigInt(firstEnd.inference.monotonicNs));
      }
    } finally { await worker.terminate(); rmSync(dir, { recursive: true, force: true }); }
  });
});
