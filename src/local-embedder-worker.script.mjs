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

import { parentPort } from 'node:worker_threads';

const DEFAULT_MODEL = 'Xenova/bge-small-en-v1.5';

// ONNX Runtime defaults its intra-op thread pool to EVERY core and SPIN-WAITS
// idle threads. On a 128-core host each operator process (main host + every
// sidecar + cluster workers) that lazily loads a pipeline grew a ~128-thread
// spin pool → hundreds of busy-waiting threads, loadavg 2000-3000, host-wide
// stutter (WI-3792, 2026-07-10 — the EmbeddingGemma-default rollout day).
// Embeds are single-request, latency-tolerant background work: cap the pool.
const ORT_SESSION_OPTIONS = { intraOpNumThreads: 4, interOpNumThreads: 1 };

// One warm pipeline PER model id, so a process mixing BGE (default local) and
// EmbeddingGemma (via an explicit model) keeps both loaded rather than
// thrashing a single-model cache.
const pipelinesByModel = new Map();

async function getPipeline(model) {
  const key = model || DEFAULT_MODEL;
  let p = pipelinesByModel.get(key);
  if (!p) {
    // Dynamic import keeps the worker spawn cheap when @huggingface/transformers
    // isn't installed — the package only loads on first embed.
    p = import('@huggingface/transformers').then((t) =>
      t.pipeline('feature-extraction', key, { session_options: ORT_SESSION_OPTIONS }),
    );
    pipelinesByModel.set(key, p);
  }
  return p;
}

parentPort.on('message', async (msg) => {
  if (!msg || typeof msg !== 'object') return;
  if (msg.kind !== 'embed') return;

  const { id, text, model } = msg;
  // BGE-small defaults (mean pooling, normalized) when unspecified; Gemma passes
  // normalize:false and truncate-then-normalizes in the caller (MRL).
  const pooling = msg.pooling || 'mean';
  const normalize = msg.normalize === undefined ? true : msg.normalize;
  try {
    const pipe = await getPipeline(model);
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
      parentPort.postMessage({ kind: 'embed_ok', id, vector: Array.from(tensor.data) });
      return;
    }
    const result = await pipe(text, { pooling, normalize });
    parentPort.postMessage({
      kind: 'embed_ok',
      id,
      vector: Array.from(result.data),
    });
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
