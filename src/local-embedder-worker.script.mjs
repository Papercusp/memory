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

import { parentPort, workerData } from 'node:worker_threads';
import { availableParallelism } from 'node:os';

const DEFAULT_MODEL = 'Xenova/bge-small-en-v1.5';

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

async function runEmbed(msg, entry) {
  const { text } = msg;
  const pooling = msg.pooling || 'mean';
  const normalize = msg.normalize === undefined ? true : msg.normalize;
  const pipe = entry.pipe;
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
      vector = await runEmbed(msg, cpuEntry);
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
