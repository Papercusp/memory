/**
 * Worker script for local-embedder-worker.ts. Runs in a dedicated
 * worker_threads thread so ONNX inference doesn't block the Node.js
 * main event loop.
 *
 * Protocol (main → worker):
 *   { kind: 'embed', id: number, text: string,
 *     model?: string, pooling?: string, normalize?: boolean, output?: string }
 *
 * Protocol (worker → main):
 *   { kind: 'ready' }                                  on init complete
 *   { kind: 'embed_ok', id: number, vector: number[] } on success
 *   { kind: 'embed_err', id: number, error: string }   on failure
 *
 * Plain ESM .mjs because worker_threads spawn doesn't go through
 * Next.js's TypeScript transform.
 */

import { parentPort, workerData, threadId } from 'node:worker_threads';
import { readFileSync, readlinkSync, realpathSync, openSync, readSync, closeSync, fstatSync } from 'node:fs';
import { isAbsolute, join } from 'node:path';
import { availableParallelism } from 'node:os';
import { AsyncLocalStorage } from 'node:async_hooks';
import { createRequire } from 'node:module';
import { dirname } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const DEFAULT_MODEL = 'Xenova/bge-small-en-v1.5';
const nativeInferenceContext = new AsyncLocalStorage();
let nativeInferenceTracePrepared = false;
let nativeInferenceTracePreparation;
let nativeClockExecutable;
const nativeFileFingerprints = new Map();

/** Stream mapped ELF/addon bytes without a second model-sized buffer. Check
 * the mapped inode/device, so replacing an on-disk library cannot silently
 * fingerprint different bytes than those loaded in this process. */
function nativeFileFingerprint(file, mappedDevice, mappedInode) {
  const descriptor = openSync(file, 'r');
  try {
    const stat = fstatSync(descriptor, { bigint: true });
    const major = ((stat.dev >> 8n) & 0xfffn) | ((stat.dev >> 32n) & 0xfffff000n);
    const minor = (stat.dev & 0xffn) | ((stat.dev >> 12n) & 0xffffff00n);
    if (mappedInode !== undefined && (stat.ino !== BigInt(mappedInode)
      || major !== BigInt('0x'+mappedDevice.split(':')[0]) || minor !== BigInt('0x'+mappedDevice.split(':')[1]))) {
      throw new Error('mapped native library differs from on-disk file: '+file);
    }
    if (!stat.isFile() || stat.size < 1n || stat.size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('invalid native library file: '+file);
    const key = `${file}:${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeNs}:${stat.ctimeNs}`;
    let fingerprint = nativeFileFingerprints.get(key);
    if (!fingerprint) {
      const hash = createHash('sha256'), buffer = Buffer.allocUnsafe(1024*1024);
      let bytes = 0, length;
      while ((length = readSync(descriptor, buffer, 0, buffer.length, null)) > 0) { hash.update(buffer.subarray(0,length)); bytes += length; }
      const after = fstatSync(descriptor, { bigint: true });
      if (BigInt(bytes) !== stat.size || after.ino !== stat.ino || after.dev !== stat.dev
        || after.size !== stat.size || after.mtimeNs !== stat.mtimeNs || after.ctimeNs !== stat.ctimeNs) {
        throw new Error('native library changed while fingerprinting: '+file);
      }
      fingerprint = { path: file, bytes, sha256: hash.digest('hex') }; nativeFileFingerprints.set(key, fingerprint);
    }
    return fingerprint;
  } finally { closeSync(descriptor); }
}

function sampleNativeRuntime(device) {
  if (process.platform !== 'linux') return undefined;
  const beforeNs = process.hrtime.bigint().toString(), files = new Map();
  for (const line of readFileSync('/proc/self/maps', 'utf8').trim().split('\n')) {
    const row = line.match(/^[\da-f]+-[\da-f]+\s+(\S+)\s+[\da-f]+\s+([\da-f]+:[\da-f]+)\s+(\d+)\s*(.*)$/i);
    if (!row) throw new Error('native library maps row malformed');
    const [, permissions, mappedDevice, mappedInode, file] = row;
    if (!file.startsWith('/') || (!permissions.includes('x') && !/\.so(?:\.|$)|\.node(?:$| )/.test(file))) continue;
    if (file.endsWith(' (deleted)')) throw new Error('native library mapping is deleted: '+file);
    const prior = files.get(file);
    if (prior && (prior.mappedDevice !== mappedDevice || prior.mappedInode !== mappedInode)) throw new Error('ambiguous native library mapping: '+file);
    files.set(file, { ...nativeFileFingerprint(file, mappedDevice, mappedInode), mappedDevice, mappedInode });
  }
  if (!files.size) throw new Error('native library mapping population missing');
  let gpuMemory = { status: 'not-applicable' };
  if (device === 'cuda') {
    const queryBeforeNs = process.hrtime.bigint().toString();
    try {
      const executable = realpathSync('/usr/bin/nvidia-smi');
      const result = spawnSync(executable, ['--query-gpu=uuid,pci.bus_id,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
        { encoding: 'utf8', timeout: 10000, maxBuffer: 65536 });
      if (result.status !== 0) throw new Error(result.error?.message ?? result.stderr ?? String(result.signal));
      const devices = result.stdout.trim().split('\n').map(line => {
        const columns = line.split(',').map(x=>x.trim());
        if (columns.length !== 5) throw new Error('invalid GPU capacity columns');
        const [uuid, pciBusId] = columns, [totalMiB, usedMiB, freeMiB] = columns.slice(2).map(Number);
        if (!uuid.startsWith('GPU-') || !/^[\da-f]+:[\da-f]+:[\da-f]+\.[\da-f]+$/i.test(pciBusId)
          || !columns.slice(2).every(x=>/^\d+(?:\.\d+)?$/.test(x))
          || ![totalMiB, usedMiB, freeMiB].every(Number.isFinite) || totalMiB <= 0 || usedMiB < 0 || freeMiB < 0
          || usedMiB > totalMiB || freeMiB > totalMiB) throw new Error('invalid GPU capacity values');
        return { uuid, pciBusId, totalMiB, usedMiB, freeMiB };
      });
      if (!devices.length || new Set(devices.map(x=>x.uuid)).size !== devices.length) throw new Error('invalid GPU capacity population');
      gpuMemory = { status: 'measured', scope: 'all-nvidia-smi-devices', beforeNs: queryBeforeNs,
        afterNs: process.hrtime.bigint().toString(), executable: nativeFileFingerprint(executable), devices,
        cudaVisibleDevices: process.env.CUDA_VISIBLE_DEVICES ?? null };
    } catch (error) { gpuMemory = { status: 'unknown', beforeNs: queryBeforeNs,
      afterNs: process.hrtime.bigint().toString(), error: String(error) }; }
  }
  return { platform: 'linux', clock: 'node-hrtime', beforeNs, afterNs: process.hrtime.bigint().toString(),
    libraries: [...files.values()].sort((a,b)=>a.path.localeCompare(b.path)), gpuMemory };
}

/** Reuse Python's standard Linux clock_gettime bindings rather than adding an
 * FFI addon. CLOCK_MONOTONIC samples are bracketed by Node's same clock. RAW
 * samples bound the native call without estimating an offset or drift model. */
function sampleNativeRawClock() {
  if (process.platform !== 'linux') return undefined;
  if (!nativeClockExecutable) {
    const path = realpathSync('/usr/bin/python3'), bytes = readFileSync(path);
    nativeClockExecutable = { path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
  }
  const nodeBeforeNs = process.hrtime.bigint().toString();
  const result = spawnSync(nativeClockExecutable.path, ['-I', '-S', '-c',
    'import json,time,sys; before=time.clock_gettime_ns(time.CLOCK_MONOTONIC); raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW); after=time.clock_gettime_ns(time.CLOCK_MONOTONIC); print(json.dumps([str(before),str(raw),str(after),sys.version.split()[0]]))'],
    { encoding: 'utf8', timeout: 10000, maxBuffer: 4096 });
  const nodeAfterNs = process.hrtime.bigint().toString();
  if (result.status !== 0) throw new Error(`native RAW clock probe failed: ${result.error?.message ?? result.stderr ?? result.signal}`);
  const [monotonicBeforeNs, rawNs, monotonicAfterNs, pythonVersion] = JSON.parse(result.stdout);
  if ([monotonicBeforeNs, rawNs, monotonicAfterNs].some(n => typeof n !== 'string' || !/^[1-9]\d*$/.test(n))
    || BigInt(nodeBeforeNs) > BigInt(monotonicBeforeNs) || BigInt(monotonicBeforeNs) > BigInt(monotonicAfterNs)
    || BigInt(monotonicAfterNs) > BigInt(nodeAfterNs)) throw new Error('native RAW clock probe does not match Node monotonic clock');
  return { clock: 'linux-clock-monotonic-raw', rawNs, monotonicBeforeNs, monotonicAfterNs,
    nodeBeforeNs, nodeAfterNs, executable: nativeClockExecutable, pythonVersion };
}

/** ORT's JS handler schedules a synchronous addon Run with setImmediate. The
 * request context follows that queue; trace the addon instance, not the outer
 * Promise. Keep the installed constructor/prototype and native receiver. */
async function prepareNativeInferenceTrace() {
  if (nativeInferenceTracePrepared) return;
  if (nativeInferenceTracePreparation) return nativeInferenceTracePreparation;
  nativeInferenceTracePreparation = (async () => {
  if (pipelinesByModel.size) throw new Error('native inference evidence requires a fresh worker before model construction');
  const require = createRequire(import.meta.url);
  const binding = workerData?.nativeBindingSpecifier
    ? (await import(workerData.nativeBindingSpecifier)).binding
    : require(join(dirname(require.resolve('onnxruntime-node')), 'binding.js')).binding;
  const descriptor = Object.getOwnPropertyDescriptor(binding, 'InferenceSession');
  const Original = binding.InferenceSession;
  if (!descriptor?.writable || typeof Original?.prototype?.run !== 'function') throw new Error('native inference evidence binding contract mismatch');
  function TracedInferenceSession(...args) {
    const session = Reflect.construct(Original, args), originalRun = session.run;
    Object.defineProperty(session, 'run', { value: function (feeds, fetches, options) {
      const context = nativeInferenceContext.getStore();
      if (!context) return Reflect.apply(originalRun, this, [feeds, fetches, options]);
      const runIndex = ++context.runIndex;
      const runTag = `pc-embed:${context.request.processId}:${context.request.workerThreadId}:${context.request.requestId}:${context.request.attempt}:${runIndex}`;
      const emit = (phase, outcome, monotonicNs, observedAt, rawClock, clockProbeError, runtime, runtimeProbeError) => parentPort.postMessage({ kind: 'embed_native_inference', id: context.request.requestId,
        inference: { ...context.request, model: context.model, device: context.device, runIndex, runTag,
          clock: 'node-hrtime', monotonicNs, observedAt, phase,
          ...(rawClock ? { rawClock } : {}), ...(clockProbeError ? { clockProbeError } : {}),
          ...(runtime ? { runtime } : {}), ...(runtimeProbeError ? { runtimeProbeError } : {}),
          ...(outcome ? { outcome } : {}) } });
      const beforeRuntime = sampleNativeRuntime(context.device), beforeClock = sampleNativeRawClock();
      emit('start', undefined, process.hrtime.bigint().toString(), new Date().toISOString(), beforeClock, undefined, beforeRuntime);
      let result, nativeError;
      try {
        result = Reflect.apply(originalRun, this, [feeds, fetches, { ...options, tag: runTag }]);
        if (result && typeof result.then === 'function') throw new Error('native inference evidence requires a synchronous native Run');
      } catch (error) { nativeError = error; }
      const endNs = process.hrtime.bigint().toString(), endedAt = new Date().toISOString();
      let afterClock, clockProbeError, afterRuntime, runtimeProbeError;
      try { afterClock = sampleNativeRawClock(); } catch (error) { clockProbeError = String(error); }
      try { afterRuntime = sampleNativeRuntime(context.device); } catch (error) { runtimeProbeError = String(error); }
      emit('end', nativeError || clockProbeError || runtimeProbeError ? 'error' : 'success', endNs, endedAt, afterClock, clockProbeError, afterRuntime, runtimeProbeError);
      if (nativeError) throw nativeError;
      if (clockProbeError) throw new Error(clockProbeError);
      if (runtimeProbeError) throw new Error(runtimeProbeError);
      return result;
    } });
    return session;
  }
  Object.setPrototypeOf(TracedInferenceSession, Original);
  TracedInferenceSession.prototype = Original.prototype;
  binding.InferenceSession = TracedInferenceSession;
  nativeInferenceTracePrepared = true;
  })();
  return nativeInferenceTracePreparation;
}

/** Host parallelism, defensively — a 0/NaN/throwing reading must degrade to the
 *  single-thread floor, never to ONNX's all-cores default. Mirrors
 *  `hostParallelism` in local-embedder-worker.ts. */
function hostParallelism() {
  try {
    const n = availableParallelism();
    return Number.isFinite(n) && n >= 1 ? n : 1;
  } catch {
    return 1;
  }
}

// ONNX Runtime defaults its intra-op thread pool to EVERY core and SPIN-WAITS
// idle threads. On a 128-core host each operator process (main host + every
// sidecar + cluster workers) that lazily loads a pipeline grew a ~128-thread
// spin pool → hundreds of busy-waiting threads, loadavg 2000-3000, host-wide
// stutter (WI-3792, 2026-07-10 — the EmbeddingGemma-default rollout day).
// Embeds are single-request, latency-tolerant background work: cap the pool.
//
// GPU: these CPU thread caps still apply on a CUDA session (ORT runs the
// operators CUDA does not take on the CPU pool). The GPU was out of scope here
// until 2026-09-24 (EI-19363236885307403: cuDNN 9 was missing, so the CUDA
// provider could not load). It is now selected by the main thread and passed
// in as workerData.device, with a CPU fallback below — see the DEVICE block.
// One multi-process concern from that history still holds: only ONE process
// per host should embed on the GPU (the embed sidecar on a sidecar host, the
// operator on a packaged desktop), so contexts are not multiplied per process.
//
// EI-20493854163389792: WI-3792's fix was a hardcoded `intraOpNumThreads: 4`,
// an ABSOLUTE constant — it capped the 128-core host and never scaled DOWN. On
// the packaged 0.0.16 desktop's fresh 8-vCPU Ubuntu guest that is HALF the
// machine, and the first-run backfill kept it saturated: 415.2% average process
// CPU while the UI sat idle (loopLag pressure=ok — the burn is on native ORT
// threads, not the event loop). The cap is now a SHARE of the host, not a
// constant. KEEP IN SYNC with local-embedder-worker.ts
// (MAX_INTRA_OP_THREADS / BACKGROUND_HOST_SHARE_DIVISOR /
// resolveIntraOpNumThreads); ort-thread-cap.test.ts is the mechanical guard.
const MAX_INTRA_OP_THREADS = 4;
const BACKGROUND_HOST_SHARE_DIVISOR = 4;
const ORT_SESSION_OPTIONS = {
  intraOpNumThreads: Math.max(
    1,
    Math.min(MAX_INTRA_OP_THREADS, Math.floor(hostParallelism() / BACKGROUND_HOST_SHARE_DIVISOR)),
  ),
  interOpNumThreads: 1,
};

// DEVICE (memory-reduction-2026-09-24 P-008 / D-003). The main thread decides
// it (embed-device.ts: PAPERCUSP_EMBED_DEVICE auto|gpu|cpu, the installed CUDA
// provider, the NVIDIA driver, any earlier demotion) and hands it over as
// workerData.device — this file is copied into bundles as ONE file, so it cannot
// import that module. What is decided here is only the FALLBACK: a CUDA session
// that will not construct, or a forward pass that fails on the GPU, is rebuilt on
// the CPU, and the switch is sticky for this worker's life. Every construction
// is reported back as { kind: 'device', model, device, demotion } so /healthz
// states what actually runs, not what was asked for.
//
// Never pass `dtype`: stored vectors come from the default (fp32) weights, and
// only the device may move without drifting the vector space (CPU vs CUDA
// cosine 0.9999999, measured 2026-09-24). embed-device.test.ts pins this file.
const REQUESTED_DEVICE = workerData && workerData.device === 'cuda' ? 'cuda' : 'cpu';
let device = REQUESTED_DEVICE;
// Test seam only: the worker test points this at a fake module so the fallback
// can be exercised without a GPU. Production never sets it.
const TRANSFORMERS_SPECIFIER =
  (workerData && typeof workerData.transformersSpecifier === 'string' && workerData.transformersSpecifier) ||
  '@huggingface/transformers';

/** An error that means the GPU itself failed, not the input. */
const GPU_FAILURE_PATTERN = /\b(cuda|cudnn|cublas|curand|cufft|gpu|tensorrt)\b|out of memory/i;

function errorMessage(err) {
  return err && err.message ? err.message : String(err);
}

// One warm pipeline PER model id, so a process mixing BGE (default local) and
// EmbeddingGemma (via an explicit model) keeps both loaded rather than
// thrashing a single-model cache. Values are Promise<{ pipe, device }>.
const pipelinesByModel = new Map();

let transformersPromise = null;
function loadTransformers() {
  if (!transformersPromise) {
    // Dynamic import keeps the worker spawn cheap when @huggingface/transformers
    // isn't installed — the package only loads on first embed.
    transformersPromise = import(TRANSFORMERS_SPECIFIER).then((t) => {
      if (
        process.env.PAPERCUSP_DISTRIBUTION_PROFILE === 'vm-release' ||
        process.env.PAPERCUSP_TRANSFORMERS_LOCAL_ONLY === '1'
      ) {
        t.env.allowLocalModels = true;
        t.env.allowRemoteModels = false;
      }
      return t;
    });
    transformersPromise.catch(() => {
      transformersPromise = null;
    });
  }
  return transformersPromise;
}

function demote(stage, err) {
  const demotion = { from: device, stage, cause: errorMessage(err) };
  device = 'cpu';
  return demotion;
}

async function buildPipeline(key) {
  const t = await loadTransformers();
  const build = (d) => t.pipeline('feature-extraction', key, { session_options: ORT_SESSION_OPTIONS, device: d });
  if (device === 'cpu') {
    const pipe = await build('cpu');
    parentPort.postMessage({ kind: 'device', model: key, device: 'cpu', demotion: null });
    return { pipe, device: 'cpu' };
  }
  try {
    const pipe = await build(device);
    parentPort.postMessage({ kind: 'device', model: key, device, demotion: null });
    return { pipe, device };
  } catch (err) {
    const demotion = demote('construct', err);
    const pipe = await build('cpu');
    parentPort.postMessage({ kind: 'device', model: key, device: 'cpu', demotion });
    return { pipe, device: 'cpu' };
  }
}

function getPipeline(model) {
  const key = model || DEFAULT_MODEL;
  let p = pipelinesByModel.get(key);
  if (!p) {
    p = buildPipeline(key);
    // A failed build must not stay cached: without this, one transient load
    // failure made every later embed of that model reject for the worker's life.
    p.catch(() => {
      if (pipelinesByModel.get(key) === p) pipelinesByModel.delete(key);
    });
    pipelinesByModel.set(key, p);
  }
  return p;
}

const rustTokenizersByModel = new Map();
async function rustTokenize(model, text) {
  if (!isAbsolute(model)) throw new Error('Rust tokenizer requires an explicit local model directory');
  let tokenizer = rustTokenizersByModel.get(model);
  if (!tokenizer) {
    const { Tokenizer } = await import('tokenizers');
    const config = JSON.parse(readFileSync(join(model, 'tokenizer_config.json'), 'utf8'));
    if (!Number.isInteger(config.model_max_length) || config.model_max_length < 1) {
      throw new Error('Rust tokenizer requires a finite pinned context limit');
    }
    tokenizer = Tokenizer.fromFile(join(model, 'tokenizer.json'));
    tokenizer.setTruncation(config.model_max_length);
    rustTokenizersByModel.set(model, tokenizer);
  }
  return tokenizer.encode(text);
}

async function runEmbed(msg, entry, attempt = 1) {
  const { text } = msg;
  const pooling = msg.pooling || 'mean';
  const normalize = msg.normalize === undefined ? true : msg.normalize;
  const pipe = entry.pipe;
  if ((msg.traceInput || msg.traceInference || msg.traceNativeInference) && msg.tokenizerBackend !== 'rust') throw new Error('input evidence requires the explicit Rust tensor route');
  if (msg.tokenizerBackend === 'rust') {
    if (pooling !== 'cls' || msg.output) throw new Error('Rust candidate contract requires native CLS output');
    const encoding = await rustTokenize(msg.model, text);
    const t = await loadTransformers();
    const ids = encoding.getIds();
    const shape = [1, ids.length];
    const inputs = {
      input_ids: new t.Tensor('int64', BigInt64Array.from(ids, BigInt), shape),
      attention_mask: new t.Tensor('int64', BigInt64Array.from(encoding.getAttentionMask(), BigInt), shape),
    };
    const request = msg.traceInput || msg.traceInference || msg.traceNativeInference ? {
      requestId: msg.id, attempt, processId: process.pid, workerThreadId: threadId,
      nativeThreadId: process.platform === 'linux' ? Number(readlinkSync('/proc/thread-self').split('/').at(-1)) : null,
    } : undefined;
    if (msg.traceInput) parentPort.postMessage({ kind: 'embed_input', id: msg.id, trace: {
      model: msg.model, device: entry.device, observedAt: new Date().toISOString(),
      request,
      inputShape: [...inputs.input_ids.dims], inputIds: Array.from(inputs.input_ids.data, Number),
      attentionMask: Array.from(inputs.attention_mask.data, Number),
    } });
    const emitInference = (phase, outcome) => {
      if (msg.traceInference) parentPort.postMessage({ kind: 'embed_inference', id: msg.id, inference: {
        ...request, model: msg.model, device: entry.device, observedAt: new Date().toISOString(),
        clock: 'node-hrtime', monotonicNs: process.hrtime.bigint().toString(), phase,
        ...(outcome ? { outcome } : {}),
      } });
    };
    emitInference('start');
    let output;
    try {
      output = await (msg.traceNativeInference
        ? nativeInferenceContext.run({ request, model: msg.model, device: entry.device, runIndex: 0 }, () => pipe.model(inputs))
        : pipe.model(inputs));
    } catch (error) { emitInference('end', 'error'); throw error; }
    emitInference('end', 'success');
    if (!output.last_hidden_state) throw new Error('Rust candidate graph lacks last_hidden_state');
    const cls = output.last_hidden_state.slice(null, 0);
    return Array.from((normalize ? cls.normalize(2, -1) : cls).data);
  }
  // Models whose ONNX export bakes pooling+normalize INTO the graph expose a
  // single pre-pooled output (e.g. harrier's 'sentence_embedding') and have
  // no last_hidden_state for the pipeline's pooling path — `output` names
  // that graph output; tokenize + run the model directly and return it.
  if (msg.output) {
    const enc = pipe.tokenizer(text, { padding: true, truncation: true });
    const out = await pipe.model(enc);
    const tensor = out[msg.output];
    if (!tensor) {
      throw new Error(`model output '${msg.output}' missing (has: ${Object.keys(out).join(', ')})`);
    }
    return Array.from(tensor.data);
  }
  const result = await pipe(text, { pooling, normalize });
  return Array.from(result.data);
}

parentPort.on('message', async (msg) => {
  if (!msg || typeof msg !== 'object') return;
  if (msg.kind !== 'embed') return;

  const { id, model } = msg;
  // BGE-small defaults (mean pooling, normalized) when unspecified; Gemma passes
  // normalize:false and truncate-then-normalizes in the caller (MRL).
  //
  // ⚠ 'last_token' (Qwen3) is SAFE HERE ONLY BECAUSE WE EMBED ONE TEXT PER
  // MESSAGE. transformers.js implements it as `result.slice(null, -1)` — the
  // final sequence position, with no attention-mask check — and the tokenizer
  // is called with `padding: true`. Over a single text that pads nothing, so
  // the final position IS the last real token. If this protocol is ever
  // widened to a batch, right-padding would silently pool a PAD embedding for
  // every text shorter than the longest: no error, just quietly wrong vectors.
  // Batch this only alongside a mask-aware last-token gather.
  const key = model || DEFAULT_MODEL;
  try {
    if (msg.traceNativeInference) await prepareNativeInferenceTrace();
    const entry = await getPipeline(key);
    let vector;
    try {
      vector = await runEmbed(msg, entry);
    } catch (err) {
      // A forward pass that fails ON THE GPU (out of memory, a CUDA/cuDNN
      // error) is a device failure, not a bad input: rebuild this model on the
      // CPU and retry once, so the GPU stays opportunistic and never becomes a
      // way for embeds to fail. Any other error is the request's own.
      if (entry.device === 'cpu' || !GPU_FAILURE_PATTERN.test(errorMessage(err))) throw err;
      const demotion = device === 'cpu' ? null : demote('inference', err);
      pipelinesByModel.delete(key);
      const t = await loadTransformers();
      const pipe = await t.pipeline('feature-extraction', key, { session_options: ORT_SESSION_OPTIONS, device: 'cpu' });
      const cpuEntry = { pipe, device: 'cpu' };
      pipelinesByModel.set(key, Promise.resolve(cpuEntry));
      parentPort.postMessage({
        kind: 'device',
        model: key,
        device: 'cpu',
        demotion: demotion ?? { from: 'cuda', stage: 'inference', cause: errorMessage(err) },
      });
      vector = await runEmbed(msg, cpuEntry, 2);
    }
    parentPort.postMessage({ kind: 'embed_ok', id, vector });
  } catch (err) {
    parentPort.postMessage({
      kind: 'embed_err',
      id,
      error: err && err.message ? err.message : String(err),
    });
  }
});

// Signal ready as soon as the message handler is installed. The model
// loads lazily on first embed call.
parentPort.postMessage({ kind: 'ready' });
