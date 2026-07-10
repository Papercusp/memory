/**
 * Worker-thread isolated local embedder.
 *
 * The @huggingface/transformers BGE-small pipeline runs ONNX inference
 * that can take 100-500ms per embedding on a modest CPU. Running it on
 * the Node.js main event loop blocks every concurrent request during
 * that window. This module wraps the pipeline in a `worker_threads`
 * Worker so embedding work happens off the main thread.
 *
 * Step B1 of Tier-3 follow-up arc.
 *
 * Architecture: one persistent worker per process (lazy-spawned on first
 * embed call). The worker holds the warm pipeline; main-thread requests
 * marshall {id, text} → worker via `postMessage`, await on a pending
 * Promise keyed by id, and resolve when {id, vector} comes back.
 *
 * Falls back to inline (main-thread) embedding if worker_threads can't
 * be loaded — keeps behavior backward-compatible.
 */

import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

interface PendingRequest {
  resolve: (v: number[]) => void;
  reject: (err: Error) => void;
}

let _worker: Worker | null = null;
let _workerReady: Promise<void> | null = null;
let _nextId = 0;
const _pending = new Map<number, PendingRequest>();
let _workerDisabled = false;

function workerPath(): string {
  // The worker script is co-located in the same dir as this module.
  // Resolved at runtime so the path works after build-time bundling
  // (Next standalone copies the file alongside).
  const here = typeof __filename !== 'undefined'
    ? __filename
    : fileURLToPath(import.meta.url);
  return resolve(dirname(here), 'local-embedder-worker.script.mjs');
}

function ensureWorker(): Promise<void> {
  if (_workerDisabled) return Promise.reject(new Error('worker disabled'));
  if (_workerReady) return _workerReady;

  _workerReady = new Promise<void>((resolveReady, rejectReady) => {
    try {
      _worker = new Worker(workerPath(), {
        // execArgv passthrough is fine — the script is plain JS,
        // no ts-node loader needed.
      });
    } catch (err) {
      _workerDisabled = true;
      rejectReady(err as Error);
      return;
    }

    let initialized = false;
    _worker.on('message', (msg: { kind: string; id?: number; vector?: number[]; error?: string }) => {
      if (msg.kind === 'ready') {
        initialized = true;
        resolveReady();
        return;
      }
      if (typeof msg.id !== 'number') return;
      const p = _pending.get(msg.id);
      if (!p) return;
      _pending.delete(msg.id);
      if (msg.kind === 'embed_ok' && Array.isArray(msg.vector)) {
        p.resolve(msg.vector);
      } else {
        p.reject(new Error(msg.error ?? 'worker error'));
      }
    });
    _worker.on('error', (err) => {
      // Reject every pending request — the worker crashed.
      for (const [, p] of _pending) p.reject(err);
      _pending.clear();
      _workerDisabled = true;
      _worker = null;
      _workerReady = null;
      if (!initialized) rejectReady(err);
    });
    _worker.on('exit', (code) => {
      if (code !== 0 && !initialized) {
        rejectReady(new Error(`worker exited with code ${code} before ready`));
      }
      _worker = null;
      _workerReady = null;
    });
  });

  return _workerReady;
}

/** Per-embed options for the worker. Omitted fields keep the BGE-small
 *  defaults (model `Xenova/bge-small-en-v1.5`, mean pooling, normalized) so
 *  existing callers are unchanged; EmbeddingGemma passes `model` +
 *  `normalize: false` (MRL truncate-then-normalize happens in the caller).
 *  `output` bypasses the pipeline's pooling path entirely and returns the
 *  named graph output from a direct model call — for exports that bake
 *  pooling+normalize into the ONNX graph (harrier's 'sentence_embedding');
 *  `pooling`/`normalize` are ignored when it is set. */
export interface EmbedViaWorkerOpts {
  model?: string;
  pooling?: 'mean' | 'cls' | 'none';
  normalize?: boolean;
  output?: string;
}

/**
 * Embed `text` via the persistent worker thread. Returns a vector
 * (Array<number>) sized to the loaded model's output dimension.
 *
 * The worker caches one pipeline PER model, so mixing models (BGE + Gemma) in
 * one process is safe — each `model` gets its own warm pipeline.
 *
 * Throws when worker_threads is unavailable or the worker has failed
 * — callers should fall back to inline embedding in that case.
 */
export async function embedViaWorker(text: string, opts: EmbedViaWorkerOpts = {}): Promise<number[]> {
  await ensureWorker();
  if (!_worker) throw new Error('worker not initialized');

  const id = _nextId++;
  return new Promise<number[]>((resolveEmbed, rejectEmbed) => {
    _pending.set(id, { resolve: resolveEmbed, reject: rejectEmbed });
    _worker!.postMessage({
      kind: 'embed',
      id,
      text,
      model: opts.model,
      pooling: opts.pooling,
      normalize: opts.normalize,
      output: opts.output,
    });
  });
}

/** Test seam — drop the worker and reset state. */
export async function _resetWorker(): Promise<void> {
  if (_worker) {
    try { await _worker.terminate(); } catch { /* noop */ }
  }
  _worker = null;
  _workerReady = null;
  _workerDisabled = false;
  _pending.clear();
  _nextId = 0;
}

/** Telemetry for /settings/user/memory diagnostics. */
export function getWorkerState(): {
  alive: boolean;
  disabled: boolean;
  pendingCount: number;
} {
  return {
    alive: _worker !== null,
    disabled: _workerDisabled,
    pendingCount: _pending.size,
  };
}

export const LOCAL_EMBEDDER_MODEL = 'Xenova/bge-small-en-v1.5';
const TRANSFORMERS_PACKAGE = '@huggingface/transformers';

type TransformersModule = {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<Pipeline>;
};
type Pipeline = (text: string, opts: unknown) => Promise<{ data: Float32Array }>;

/** ONNX Runtime defaults intraOp threads to EVERY core and spin-waits them —
 *  on a 128-core host each embedding-loading process grew a ~128-thread spin
 *  pool (loadavg 2000-3000 host stutter, WI-3792). Cap it: embeds are
 *  latency-tolerant background work. Mirrored in local-embedder-worker.script.mjs
 *  (plain-JS worker, can't import this) — keep the two in sync. */
export const ORT_SESSION_OPTIONS = { intraOpNumThreads: 4, interOpNumThreads: 1 } as const;

// Dodge the bundler's static analysis so the optional @huggingface
// dependency is only required when local mode is actually selected.
const dynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as <T>(specifier: string) => Promise<T>;

/**
 * Build a local (free, offline) BGE-small embedder.
 *
 * Prefers the worker-thread isolated path (`embedViaWorker`) so ONNX
 * inference doesn't block the main event loop; falls back to an inline
 * main-thread pipeline when worker_threads is unavailable or the worker
 * fails to spawn (older Node, some vitest environments). The fallback is
 * sticky per returned closure — once a worker spawn fails we stop
 * retrying it for that embedder instance.
 */
export async function buildLocalEmbedder(): Promise<(text: string) => Promise<number[]>> {
  let pipelinePromise: Promise<Pipeline> | null = null;
  let workerDisabled = false;

  return async (text: string): Promise<number[]> => {
    if (!workerDisabled) {
      try {
        return await embedViaWorker(text);
      } catch {
        // Worker spawn or embedding failed — disable for subsequent
        // calls on this closure and fall through to inline.
        workerDisabled = true;
      }
    }

    // Inline (main-thread) fallback path.
    if (!pipelinePromise) {
      const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
      pipelinePromise = transformers.pipeline('feature-extraction', LOCAL_EMBEDDER_MODEL, {
        session_options: ORT_SESSION_OPTIONS,
      });
    }
    const pipe = await pipelinePromise;
    const result = await pipe(text, { pooling: 'mean', normalize: true });
    return Array.from(result.data);
  };
}
