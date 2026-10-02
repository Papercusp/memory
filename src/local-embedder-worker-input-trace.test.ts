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
      expect(msg).not.toHaveProperty('traceNativeInference');
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
          if (failure === 'thread') end.nativeThreadId = 457;
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
  it.each(['valid', 'missing', 'tag', 'run-index', 'thread', 'unfinished', 'runtime-files', 'runtime-order', 'runtime-error'])
    ('checks %s native-call evidence independently of the outer graph interval', async (failure) => {
      const seen: any[] = [];
      vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: Worker, msg: any) {
        expect(msg.traceNativeInference).toBe(true);
        setImmediate(() => {
          const start = { requestId: msg.id, attempt: 1, processId: 123, workerThreadId: 1, nativeThreadId: 456,
            model: trace.model, device: 'cpu', observedAt: trace.observedAt, clock: 'node-hrtime', monotonicNs: '100',
            phase: 'start', runIndex: 1, runTag: `pc-embed:123:1:${msg.id}:1:1` };
          const end = { ...start, monotonicNs: '200', phase: 'end', outcome: 'success' };
          if (failure === 'tag') start.runTag = 'another-request';
          if (failure === 'run-index') end.runIndex++;
          if (failure === 'thread') end.nativeThreadId++;
          if (failure === 'runtime-files' || failure === 'runtime-order') (start as any).runtime = { platform: 'linux', clock: 'node-hrtime',
            beforeNs: '10', afterNs: failure === 'runtime-order' ? '101' : '90',
            libraries: failure === 'runtime-files' ? [] : [{ path: '/runtime/ort.node', bytes: 10, sha256: 'a'.repeat(64), mappedDevice: '08:01', mappedInode: '123' }],
            gpuMemory: { status: 'not-applicable' } };
          if (failure === 'runtime-error') (start as any).runtimeProbeError = 'invalid worker runtime sampler';
          if (failure !== 'missing') this.emit('message', { kind: 'embed_native_inference', id: msg.id, inference: start });
          if (!['missing', 'unfinished'].includes(failure)) this.emit('message', { kind: 'embed_native_inference', id: msg.id, inference: end });
          this.emit('message', { kind: 'embed_ok', id: msg.id, vector: [1] });
        });
      });
      const result = embedViaWorker('q', { onNativeInferenceTrace: (event) => seen.push(event) });
      if (failure === 'valid') { expect(await result).toEqual([1]); expect(seen.map(e => e.phase)).toEqual(['start', 'end']); }
      else await expect(result).rejects.toThrow(['missing', 'unfinished'].includes(failure)
        ? 'without complete requested native inference evidence' : 'invalid worker');
    });
});

describe('real worker submitted tensors', () => {
  it.each(['valid','invalid','separate-loader-output'])('retains %s capacity output through the real native-call sampler with a model-free query fixture', async (outcome) => {
    const dir = mkdtempSync(join(tmpdir(), 'worker-native-capacity-'));
    const fake = join(dir,'transformers.mjs'), native = join(dir,'native.mjs'), query = join(dir,'nvidia-query.mjs');
    writeFileSync(join(dir,'tokenizer_config.json'),JSON.stringify({model_max_length:3}));
    writeFileSync(join(dir,'tokenizer.json'),JSON.stringify({version:'1.0',truncation:null,padding:null,added_tokens:[],normalizer:null,
      pre_tokenizer:{type:'Whitespace'},post_processor:null,decoder:null,
      model:{type:'WordLevel',vocab:{'[UNK]':0,fixture:1},unk_token:'[UNK]'}}));
    writeFileSync(query, '#!/usr/bin/env node\n'
      + 'if(process.argv[2]!=="--query-gpu=uuid,pci.bus_id,memory.total,memory.used,memory.free" || process.argv[3]!=="--format=csv,noheader,nounits")process.exit(2);\n'
      + `console.log(${JSON.stringify(outcome==='valid' ? 'GPU-fixture, 00000000:01:00.0, 100, 20, 70' : 'GPU-fixture, 00000000:01:00.0, N/A, 20, 70')});\n`, { mode: 0o700 });
    writeFileSync(native, 'class Session { run() { return { last_hidden_state: { slice:()=>({normalize:()=>({data:[1,0]})}) } }; } }\nexport const binding={InferenceSession:Session};\n');
    writeFileSync(fake, `import {binding} from ${JSON.stringify(pathToFileURL(native).href)};\nexport const env={};\nexport class Tensor{constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims;}}\nexport async function pipeline(){const s=new binding.InferenceSession();return {model:async inputs=>s.run(inputs,{},{})};}\n`);
    const worker = new Worker(new URL('./local-embedder-worker.script.mjs',import.meta.url), { execArgv:[], workerData:{ device:'cuda',
      transformersSpecifier:pathToFileURL(fake).href, nativeBindingSpecifier:pathToFileURL(native).href, nativeGpuQueryExecutable:query },
      ...(outcome==='separate-loader-output' ? {env:{...process.env,LD_DEBUG:'files',LD_DEBUG_OUTPUT:join(dir,'loader')}} : {}) });
    const messages:any[]=[];
    try {
      const ready=new Promise<void>((resolve,reject)=>{worker.once('error',reject);worker.on('message',m=>{messages.push(m);if(m.kind==='ready')resolve();});});
      await ready;
      const finished=new Promise<any>((resolve,reject)=>{worker.once('error',reject);worker.on('message',m=>{if(['embed_ok','embed_err'].includes(m.kind))resolve(m);});});
      worker.postMessage({kind:'embed',id:0,text:'fixture',model:dir,tokenizerBackend:'rust',pooling:'cls',traceNativeInference:true});
      const result=await finished;
      if(outcome==='separate-loader-output') {
        expect(result).toMatchObject({kind:'embed_err',error:expect.stringContaining('requires shared stderr')});
        expect(messages.filter(m=>m.kind==='embed_native_inference')).toEqual([]);
        return;
      }
      expect(result).toMatchObject({kind:'embed_ok',vector:[1,0]});
      const events=messages.filter(m=>m.kind==='embed_native_inference');expect(events.map(m=>m.inference.phase)).toEqual(['start','end']);
      for(const event of events){
        const gpu=event.inference.runtime.gpuMemory;
        if(outcome==='valid')expect(gpu).toMatchObject({status:'measured',scope:'all-nvidia-smi-devices',executable:{path:query,sha256:expect.stringMatching(/^[a-f0-9]{64}$/)},
          devices:[{uuid:'GPU-fixture',pciBusId:'00000000:01:00.0',totalMiB:100,usedMiB:20,freeMiB:70}]});
        else expect(gpu).toMatchObject({status:'unknown',error:expect.stringContaining('invalid GPU capacity values')});
      }
    } finally {await worker.terminate();rmSync(dir,{recursive:true,force:true});}
  });
  it('traces the actual ORT native receiver and queued calls in an isolated CPU process', () => {
    const dir = mkdtempSync(join(tmpdir(), 'worker-native-run-'));
    const fake = join(dir, 'cpu-native-transformers.mjs');
    const ortUrl = import.meta.resolve('onnxruntime-node');
    // Upstream ORT v1.24.3 testdata/mul_1.onnx (MIT), 130 bytes,
    // sha256 71f431c4e9321ec6fbeb158d02ed240459a7dcc98673fa79a4f439ce42efaf10.
    const graph = 'CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4KDAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==';
    writeFileSync(join(dir, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 3 }));
    writeFileSync(join(dir, 'tokenizer.json'), JSON.stringify({ version: '1.0', truncation: null, padding: null,
      added_tokens: [], normalizer: null, pre_tokenizer: { type: 'Whitespace' }, post_processor: null, decoder: null,
      model: { type: 'WordLevel', vocab: { '[UNK]': 0, a: 1, b: 2, c: 3 }, unk_token: '[UNK]' } }));
    writeFileSync(fake, `import ort from ${JSON.stringify(ortUrl)};
export const env = {};
export class Tensor { constructor(type, data, dims) { this.type=type; this.data=data; this.dims=dims; } }
export async function pipeline() {
  const session = await ort.InferenceSession.create(Buffer.from(${JSON.stringify(graph)}, 'base64'),
    { executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1 });
  return { model: async inputs => {
    const first = Number(inputs.input_ids.data[0]);
    const output = await session.run({ X: new ort.Tensor('float32', Float32Array.from([first,1,1,1,1,1]), first === 3 ? [2,3] : [3,2]) });
    return { last_hidden_state: { slice: () => ({ normalize: () => ({ data: [output.Y.data[0],0] }) }) } };
  } };
}
`);
    const program = `import { Worker } from 'node:worker_threads';
const worker = new Worker(new URL(${JSON.stringify(new URL('./local-embedder-worker.script.mjs', import.meta.url).href)}),
 { workerData: { device:'cpu', transformersSpecifier:${JSON.stringify(pathToFileURL(fake).href)} }, execArgv:[] });
const messages=[];
try {
 await new Promise((resolve,reject)=>{worker.once('error',reject);worker.on('message',m=>{messages.push(m);if(m.kind==='ready')resolve();});});
 const finished=new Promise((resolve,reject)=>{const rows=[];worker.once('error',reject);worker.on('message',m=>{
  if(['embed_ok','embed_err'].includes(m.kind)){rows.push(m);if(rows.length===3)resolve(rows);}
 });});
 for(const [id,text] of [[1,'a b c'],[2,'b a c'],[3,'c a b']])worker.postMessage({kind:'embed',id,text,
  model:${JSON.stringify(dir)},tokenizerBackend:'rust',pooling:'cls',traceInput:true,traceInference:true,traceNativeInference:true});
 const results=await finished;
 console.log(JSON.stringify({results,native:messages.filter(m=>m.kind==='embed_native_inference'),
  graph:messages.filter(m=>m.kind==='embed_inference')}));
} finally {await worker.terminate();}
`;
    try {
      const env: NodeJS.ProcessEnv={...process.env,LD_DEBUG:'files'}; delete env.LD_DEBUG_OUTPUT;
      const child = spawnSync(process.execPath, ['--input-type=module'], { input: program, env, encoding: 'utf8', timeout: 20000, maxBuffer: 4*1024*1024 });
      expect(child.status, child.stderr).toBe(0);
      const observed = JSON.parse(child.stdout.trim());
      expect(observed.results.find((r: any) => r.id === 1)).toMatchObject({ kind: 'embed_ok', vector: [1,0] });
      expect(observed.results.find((r: any) => r.id === 2)).toMatchObject({ kind: 'embed_ok', vector: [2,0] });
      expect(observed.results.find((r: any) => r.id === 3)).toMatchObject({ kind: 'embed_err', error: expect.stringContaining('invalid dimensions') });
      for (const id of [1, 2, 3]) {
        const events = observed.native.filter((e: any) => e.id === id);
        expect(events.map((e: any) => [e.inference.phase, e.inference.outcome])).toEqual([
          ['start', undefined], ['end', id === 3 ? 'error' : 'success'],
        ]);
        expect(events[0].inference.runIndex).toBe(1);
        expect(events[0].inference.runTag).toContain(`:${id}:1:1`);
        expect(BigInt(events[1].inference.monotonicNs)).toBeGreaterThan(BigInt(events[0].inference.monotonicNs));
        if (process.platform === 'linux') {
          for (const event of events) {
            const clock = event.inference.rawClock;
            expect(clock).toMatchObject({ clock: 'linux-clock-monotonic-raw',
              executable: { path: expect.stringContaining('python3'), sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
              pythonVersion: expect.any(String) });
            expect(BigInt(clock.monotonicBeforeNs)).toBeGreaterThanOrEqual(BigInt(clock.nodeBeforeNs));
            expect(BigInt(clock.monotonicAfterNs)).toBeLessThanOrEqual(BigInt(clock.nodeAfterNs));
            const runtime = event.inference.runtime;
            expect(runtime).toMatchObject({ platform: 'linux', clock: 'node-hrtime', gpuMemory: { status: 'not-applicable' } });
            expect(runtime.libraries.length).toBeGreaterThan(0);
            expect(runtime.libraries.some((f: any)=>f.path.endsWith('.node'))).toBe(true);
            for (const file of runtime.libraries) {
              expect(file).toMatchObject({ path: expect.stringMatching(/^\//),
                bytes: expect.any(Number), sha256: expect.stringMatching(/^[a-f0-9]{64}$/), mappedInode: expect.stringMatching(/^[1-9]\d*$/) });
              expect(file.mappedRanges.length).toBeGreaterThan(0);
              for (const range of file.mappedRanges) {
                expect(BigInt('0x'+range.startAddress)).toBeLessThan(BigInt('0x'+range.endAddress));
                expect(range.permissions).toMatch(/^[r-][w-][x-][ps]$/);
              }
            }
            expect(runtime.libraries.find((file:any)=>file.path === process.execPath)?.mappedRanges
              .some((range:any)=>!range.permissions.includes('x'))).toBe(true);
            if (event.inference.phase === 'start') expect(BigInt(runtime.afterNs)).toBeLessThanOrEqual(BigInt(event.inference.monotonicNs));
            else expect(BigInt(runtime.beforeNs)).toBeGreaterThanOrEqual(BigInt(event.inference.monotonicNs));
          }
          expect(BigInt(events[1].inference.rawClock.rawNs)).toBeGreaterThan(BigInt(events[0].inference.rawClock.rawNs));
        }
      }
      expect(observed.native).toHaveLength(6);
      const markers = child.stderr.split('\n').filter(line=>line.startsWith('PC_NATIVE_RUN\t'))
        .map(line=>JSON.parse(line.slice('PC_NATIVE_RUN\t'.length)));
      expect(markers).toEqual(observed.native.map((event:any)=>({runTag:event.inference.runTag,phase:event.inference.phase,
        monotonicNs:event.inference.monotonicNs,processId:event.inference.processId,nativeThreadId:event.inference.nativeThreadId})));
      for (let i = 2; i < observed.native.length; i += 2) {
        expect(BigInt(observed.native[i].inference.monotonicNs)).toBeGreaterThan(BigInt(observed.native[i-1].inference.monotonicNs));
      }
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
  it.each([false, true])('emits exact truncated tensors before graph success/failure=%s', async (fail) => {
    const dir = mkdtempSync(join(tmpdir(), 'worker-input-evidence-'));
    const fake = join(dir, 'fake-transformers.mjs');
    const native = join(dir, 'fake-native-binding.mjs');
    writeFileSync(join(dir, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 3 }));
    writeFileSync(join(dir, 'tokenizer.json'), JSON.stringify({ version: '1.0', truncation: null, padding: null,
      added_tokens: [], normalizer: null, pre_tokenizer: { type: 'Whitespace' }, post_processor: null, decoder: null,
      model: { type: 'WordLevel', vocab: { '[UNK]': 0, a: 1, b: 2, c: 3, d: 4 }, unk_token: '[UNK]' } }));
    writeFileSync(native, `class Session {
  run(inputs, fetches, options) {
    if (!(this instanceof Session) || !options.tag.startsWith('pc-embed:')) throw new Error('native receiver/tag mismatch');
    const ids = Array.from(inputs.input_ids.data, Number);
    if (${fail}) throw new Error('graph rejected exact tensors');
    return { last_hidden_state: { slice: () => ({ normalize: () => ({ data: [ids[0],0] }) }) } };
  }
}
Object.defineProperty(Session.prototype, 'run', { writable: false, configurable: false });
export const binding = { InferenceSession: Session };
`);
    writeFileSync(fake, `import { binding } from ${JSON.stringify(pathToFileURL(native).href)};
export const env = {};
export class Tensor { constructor(type, data, dims) { this.type=type; this.data=data; this.dims=dims; } }
export async function pipeline() { const session = new binding.InferenceSession(); return { model: async inputs => {
  const ids = Array.from(inputs.input_ids.data, Number);
  if (ids.length !== 3) throw new Error('wrong submitted ids');
  await new Promise(resolve => setTimeout(resolve, ids[0] === 1 ? 30 : 0));
  return new Promise((resolve, reject) => setImmediate(() => {
    try { resolve(session.run(inputs, null, {})); } catch (error) { reject(error); }
  }));
} }; }
`);
    const worker = new Worker(new URL('./local-embedder-worker.script.mjs', import.meta.url), {
      workerData: { device: 'cpu', transformersSpecifier: pathToFileURL(fake).href, nativeBindingSpecifier: pathToFileURL(native).href },
    });
    const messages: any[] = [];
    try {
      await new Promise<void>((resolve, reject) => { worker.once('error', reject); worker.on('message', m => {
        messages.push(m); if (m.kind === 'ready') resolve();
      }); });
      const done = new Promise<any>((resolve, reject) => { worker.once('error', reject); worker.on('message', m => {
        if (m.id === 7 && ['embed_ok', 'embed_err'].includes(m.kind)) resolve(m);
      }); });
      worker.postMessage({ kind: 'embed', id: 7, text: 'a b c d', model: dir, tokenizerBackend: 'rust', pooling: 'cls', traceInput: true, traceInference: true, traceNativeInference: true });
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
      const nativeEvents = messages.filter(m => m.kind === 'embed_native_inference');
      expect(nativeEvents.map(m => [m.inference.phase, m.inference.outcome])).toEqual([
        ['start', undefined], ['end', fail ? 'error' : 'success'],
      ]);
      expect(nativeEvents[0].inference).toMatchObject({ ...input.trace.request, runIndex: 1,
        runTag: `pc-embed:${process.pid}:${worker.threadId}:7:1:1` });
      expect(BigInt(nativeEvents[0].inference.monotonicNs)).toBeGreaterThan(BigInt(lifecycle[0].inference.monotonicNs));
      expect(BigInt(nativeEvents[1].inference.monotonicNs)).toBeLessThan(BigInt(lifecycle[1].inference.monotonicNs));
      if (!fail) {
        const concurrent = new Promise<any[]>((resolve, reject) => {
          const results: any[] = [];
          worker.once('error', reject); worker.on('message', m => {
            if ([8, 9].includes(m.id) && m.kind === 'embed_ok') { results.push(m); if (results.length === 2) resolve(results); }
          });
        });
        for (const [id, text] of [[8, 'a b c d'], [9, 'b a c d']]) worker.postMessage({
          kind: 'embed', id, text, model: dir, tokenizerBackend: 'rust', pooling: 'cls', traceInput: true, traceInference: true, traceNativeInference: true,
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
        const nativeGroup = messages.filter(m => m.kind === 'embed_native_inference' && [8, 9].includes(m.id));
        expect(nativeGroup.map(m => [m.id, m.inference.phase])).toEqual([[9, 'start'], [9, 'end'], [8, 'start'], [8, 'end']]);
        expect(BigInt(nativeGroup[1].inference.monotonicNs)).toBeLessThan(BigInt(nativeGroup[2].inference.monotonicNs));
      }
    } finally { await worker.terminate(); rmSync(dir, { recursive: true, force: true }); }
  });
});
