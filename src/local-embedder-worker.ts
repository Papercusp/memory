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
import { dynamicImport } from './dynamic-import';

interface PendingRequest {
  resolve: (v: number[]) => void;
  reject: (err: Error) => void;
}

let _worker: Worker | null = null;
let _workerReady: Promise<void> | null = null;
let _nextId = 0;
const _pending = new Map<number, PendingRequest>();
let _workerDisabled = false;
/** Guards the process-level `beforeExit` hook below so it is registered at
 *  most once for the life of the process, no matter how many times
 *  `ensureWorker()` (re)spawns a worker. */
let _beforeExitHookInstalled = false;

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
        // EI-19464316359123796: a persistent, REF'd worker thread keeps the
        // event loop alive forever, so any one-off driver that just wants an
        // embedding and then exits has no way to finish naturally — which is
        // exactly why the reported repro script reaches for `process.exit(0)`.
        // `process.exit()` tears the whole process (and every worker thread's
        // native addon state, mid-flight) down WITHOUT running Node's normal
        // per-Environment cleanup hooks, and that abrupt teardown is what
        // surfaces as the unlabelled `Napi::Error` at termination — the ONNX
        // native addon never gets the clean shutdown `Worker#terminate()`
        // (or a natural process exit) gives it. Unref'ing lets a script with
        // no other pending work exit ON ITS OWN once it's done, which is the
        // fix that removes the NEED for `process.exit()` in future scripts.
        _worker?.unref();
        installBeforeExitHook();
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
 *  `pooling`/`normalize` are ignored when it is set.
 *
 *  ⚠ POOLING IS A PROPERTY OF THE MODEL, NOT A TUNABLE. Each embedder must
 *  pass the pooling its own training used — read from that model's
 *  `1_Pooling/config.json`, never guessed and never carried over from a
 *  sibling. Scoring a model under the wrong pooling does not error: it returns
 *  a plausible vector from a subtly wrong space, so a bake-off reads it as a
 *  fair loss when it is really a measurement bug. The three in use here:
 *  `mean` (BGE, gemma), `cls` (granite r2), `last_token` (Qwen3 — and harrier,
 *  which instead bakes it into its graph and so uses `output`). */
export interface EmbedViaWorkerOpts {
  model?: string;
  /** `last_token` is transformers.js's `last_token`/`eos` pooling. ⚠ It takes
   *  the FINAL sequence position, which is the last REAL token only because
   *  this worker embeds exactly one text per call (`padding: true` over a
   *  single text pads nothing). If this is ever batched, right-padding would
   *  make it pool a PAD token — see the note in the worker script. */
  pooling?: 'mean' | 'cls' | 'none' | 'last_token';
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

/** Test seam — drop the worker and reset state. Kept as the underlying
 *  implementation `_resetWorker` for backward compatibility (tests import it
 *  directly); {@link shutdownLocalEmbedder} is the same function under a
 *  discoverable public name — see its doc for why both exist. */
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

/**
 * Gracefully terminate the persistent embedding worker, if one is running.
 *
 * `_resetWorker` under a name that does not read as test-only-private (EI-19464316359123796):
 * the underscore prefix on the original name signals "don't call this" to anyone scanning the
 * package's exports, which is backwards — this is the one thing an ad-hoc script SHOULD call.
 *
 * **Call this and `await` it before an explicit `process.exit(...)`** in any standalone
 * script/driver that touches a local (BGE/Gemma/harrier) embedder — directly, or indirectly via
 * `sync-resolver` / anything that resolves a `learning.*`/search-backed query. `Worker#terminate()`
 * runs Node's normal per-Environment cleanup for the worker's isolate (including the ONNX native
 * addon's own finalizers); `process.exit()` does not — it tears the whole process down mid-flight,
 * which is what surfaces as an unlabelled `terminate called after throwing an instance of
 * 'Napi::Error'` AFTER your script's real output has already printed.
 *
 * Scripts that never call `process.exit()` at all no longer need this: the worker unrefs itself
 * once ready, so a script with no other pending work exits on its own once it's done, running the
 * same graceful worker teardown automatically (see the `beforeExit` hook below). This export exists
 * for the case an explicit `process.exit()` is unavoidable.
 */
export const shutdownLocalEmbedder = _resetWorker;

/**
 * Best-effort automatic teardown for the common case: a script that just lets
 * itself exit naturally (no explicit `process.exit()`). Installed once, lazily,
 * the first time a worker actually spins up — never eagerly at module load, so
 * merely importing this module never adds a process-level listener.
 *
 * `beforeExit` (unlike `exit`) permits async work, which is required here —
 * `Worker#terminate()` returns a Promise. It only fires once the event loop
 * would otherwise go idle, which is exactly why the worker unrefs itself
 * above: a still-ref'd worker keeps the loop alive and `beforeExit` would
 * never be reached at all.
 *
 * Deliberately NOT a substitute for `shutdownLocalEmbedder()` before an
 * explicit `process.exit()` — that call skips `beforeExit` entirely by
 * design (Node's own semantics), so this hook cannot help that case. The
 * two are complementary, not redundant: this covers "the script just ends";
 * the exported function covers "the script forces itself to end".
 */
function installBeforeExitHook(): void {
  if (_beforeExitHookInstalled) return;
  _beforeExitHookInstalled = true;
  process.on('beforeExit', () => {
    // Fire-and-forget: `beforeExit` cannot be `async` itself, but scheduling
    // this promise is enough to keep the loop alive until it settles (Node
    // re-checks for idle after every `beforeExit` emission), and a worker
    // left over from a previous cycle is harmless — `ensureWorker` respawns
    // one lazily on the next real embed call, exactly as it already does
    // after any other worker crash/reset.
    void _resetWorker();
  });
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

/**
 * Rate-limited warning for a builder (gemma/harrier/local) falling back to
 * inline (main-thread-blocking) embedding for ONE call (EI-16184). Every
 * embedViaWorker() failure used to permanently stick that embedder CLOSURE
 * to the inline path for the rest of the process's life (see the removed
 * per-closure `workerDisabled` booleans this replaces) — completely silent
 * for two of the three builders — so a single transient worker hiccup (a
 * crash mid-request, a spawn race during a noisy post-restart boot window)
 * condemned every subsequent embed on that closure to block the event loop
 * for the full ONNX inference duration, often for the closure's remaining
 * hour-long cache lifetime. Builders now retry the worker on EVERY call
 * instead (ensureWorker() already self-heals a crashed worker by respawning;
 * only genuine unavailability — getWorkerState().disabled — skips straight to
 * inline), so a SUSTAINED failure could otherwise warn on every single call;
 * rate-limit it to one line per cooldown window instead.
 */
let _lastFallbackWarnAt = 0;
const FALLBACK_WARN_COOLDOWN_MS = 30_000;
export function warnEmbedFallback(model: string, err: unknown): void {
  const now = Date.now();
  if (now - _lastFallbackWarnAt < FALLBACK_WARN_COOLDOWN_MS) return;
  _lastFallbackWarnAt = now;
  if (process.env.NODE_ENV !== 'test') {
    console.warn(
      `[embed] ${model} worker path failed — falling back to inline (main-thread, blocks the event loop) for this call: ` +
        `${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/** Test-only: reset the fallback-warn cooldown so tests can assert on it independently. */
export function _resetFallbackWarnForTest(): void {
  _lastFallbackWarnAt = 0;
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

/**
 * Build a local (free, offline) BGE-small embedder.
 *
 * Prefers the worker-thread isolated path (`embedViaWorker`) so ONNX
 * inference doesn't block the main event loop; falls back to an inline
 * main-thread pipeline for THIS call when the worker is unavailable or the
 * call fails. EI-16184: this used to be sticky per closure (a single
 * worker failure permanently disabled the worker path for the rest of the
 * closure's life, silently — see warnEmbedFallback's doc comment for the
 * full incident). Every call now re-checks the module's own liveness
 * (`getWorkerState().disabled` — set only for genuine, permanent
 * unavailability; a crashed worker instead self-heals via `ensureWorker`'s
 * respawn), so a transient failure costs at most one degraded call.
 */
export async function buildLocalEmbedder(): Promise<(text: string) => Promise<number[]>> {
  let pipelinePromise: Promise<Pipeline> | null = null;

  return async (text: string): Promise<number[]> => {
    if (!getWorkerState().disabled) {
      try {
        return await embedViaWorker(text);
      } catch (err) {
        warnEmbedFallback('local', err);
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
