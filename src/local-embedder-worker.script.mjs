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
// GPU (CUDA) IS DELIBERATELY OUT OF SCOPE — this is CPU-only ON PURPOSE, not
// an oversight (EI-19363236885307403, 2026-08-02). `onnxruntime-node` ships
// libonnxruntime_providers_cuda.so, which makes CUDA LOOK available even on
// this host's idle RTX 3090 — but loading it fails at pipeline-construction
// time (not at import, not in `nvidia-smi`) on `libcudnn.so.9: cannot open
// shared object file`: cuDNN 9 is not installed, and Ubuntu's `nvidia-cudnn`
// apt package only offers cuDNN **8**, so apt cannot close the gap at all.
// A real cuDNN 9 IS available with no apt/driver involvement at all — the
// `nvidia-cudnn-cu12` PyPI wheel ships libcudnn.so.9 as plain userspace .so
// files (no dkms, no kernel module, no risk to the host's live display
// session) — but wiring that in for real needs: (1) installing + pointing
// LD_LIBRARY_PATH at it for every process context that spawns this worker
// (main host, every sidecar, cluster workers — not just this file), (2) a
// `device:'cuda'` opt-in here with a try/catch fallback to the CPU path
// above so a host without cuDNN 9 still works, and (3) a thread/session
// guard for the CUDA provider analogous to the CPU one above so a shared,
// multi-process host doesn't repeat the WI-3792 failure mode against GPU
// memory/contexts instead of CPU threads. That is real infra work across a
// shared production host, not a one-line flag flip — it was deliberately
// NOT done inside this fix. If the perf need (see EI-19363236885307403 —
// ~24h single-worker for a one-time 395k-vector re-embed, ~30min/day
// steady-state CPU) becomes pressing, start from the pip-wheel path above,
// not from apt.
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
  //
  // ⚠ 'last_token' (Qwen3) is SAFE HERE ONLY BECAUSE WE EMBED ONE TEXT PER
  // MESSAGE. transformers.js implements it as `result.slice(null, -1)` — the
  // final sequence position, with no attention-mask check — and the tokenizer
  // is called with `padding: true`. Over a single text that pads nothing, so
  // the final position IS the last real token. If this protocol is ever
  // widened to a batch, right-padding would silently pool a PAD embedding for
  // every text shorter than the longest: no error, just quietly wrong vectors.
  // Batch this only alongside a mask-aware last-token gather.
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
