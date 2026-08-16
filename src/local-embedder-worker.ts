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
import { availableParallelism } from 'node:os';
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
/**
 * Set ONLY for genuine, permanent unavailability — the worker could not be
 * CONSTRUCTED at all (no `worker_threads`, missing script, spawn threw). A
 * runtime crash deliberately does NOT set this: it clears the worker handle so
 * the next call respawns one.
 *
 * That asymmetry is the point (EI-16184's lesson, re-learned here as
 * EI-20012631851693581): treating a transient crash as permanent condemns every
 * later call in the process to the inline, main-thread-blocking path — i.e. one
 * hiccup silently undoes this whole module for the rest of the process's life.
 * EI-16184 removed that stickiness from the per-CLOSURE booleans; an equivalent
 * one had survived at module scope, here.
 *
 * ⚠ If you add a new assignment site, it must be a CONSTRUCTION failure. There
 * is a guard test for exactly this (`a transient runtime crash does not latch`).
 */
let _workerDisabled = false;
/** Guards the process-level `beforeExit` hook below so it is registered at
 *  most once for the life of the process, no matter how many times
 *  `ensureWorker()` (re)spawns a worker. */
let _beforeExitHookInstalled = false;
/** The actual listener function, kept so a test can remove exactly it. */
let _beforeExitListener: (() => Promise<void>) | null = null;

/**
 * Whether the worker is currently holding the event loop open. Mirrors the last
 * `ref()`/`unref()` we issued, because `Worker` exposes no way to read it back.
 */
let _refd = false;

/**
 * Hold the loop open for EXACTLY as long as a request is in flight, and not one
 * moment longer.
 *
 * WHY BOTH HALVES ARE LOAD-BEARING (WI-37683, the sibling of WI-37680 in
 * `libs/generic/rerank/src/local-reranker-worker.ts` — this module is the one
 * that was copied from). An always-ref'd worker keeps a one-shot script alive
 * forever, which is the bug the original `unref()` below fixed. But an
 * always-UNREF'd worker is worse in a way that is *silent*: while the caller
 * awaits its vector, neither the unref'd worker nor the awaited Promise counts
 * as loop work, so a host with nothing else ref'd is considered IDLE
 * **mid-request**. `beforeExit` then fires, the hook terminates the worker, and
 * because `_resetWorker` used to CLEAR `_pending` rather than reject it, the
 * caller's promise never settled at all — the process just exited 0 having
 * produced neither a vector nor an error.
 *
 * So the whole worker-thread mechanism silently did not apply on any host whose
 * loop is otherwise empty: every CLI, bench, migration driver and one-off
 * script. The operator was never affected, because an HTTP server's loop is
 * never idle — i.e. it failed only where nobody watches and worked everywhere
 * anybody looks, which is how it survived this long.
 *
 * The window is the whole model load, not just inference: the worker script
 * posts `{kind:'ready'}` as soon as its message handler is installed, BEFORE
 * building any pipeline. Measured 2026-08-10: a standalone script awaiting one
 * `embedViaWorker` was stranded 3/3, with `beforeExit` observing
 * `pendingCount: 1`; the byte-identical script with a `setInterval` holding the
 * loop ref'd returned a 384-dim vector in 683ms.
 *
 * Ref'ing only while `_pending` is non-empty satisfies both: a script that
 * awaits an embedding stays alive until its vector arrives, then exits on its
 * own.
 */
function syncWorkerRef(): void {
  const want = _pending.size > 0;
  if (!_worker || want === _refd) return;
  if (want) _worker.ref();
  else _worker.unref();
  _refd = want;
}

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
        //
        // WI-37683: unconditionally unref'ing here strands any request already
        // in flight, so let `syncWorkerRef` decide — idle ⇒ unref (the
        // behaviour above), a request pending ⇒ ref until it lands.
        syncWorkerRef();
        installBeforeExitHook();
        resolveReady();
        return;
      }
      if (typeof msg.id !== 'number') return;
      const p = _pending.get(msg.id);
      if (!p) return;
      _pending.delete(msg.id);
      // Release the loop as soon as the LAST request lands, so a one-off script
      // still exits on its own (WI-37683 — the other half of syncWorkerRef).
      syncWorkerRef();
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
      // EI-20012631851693581: deliberately NOT `_workerDisabled`. A runtime
      // crash is TRANSIENT; clearing the handle is what makes the next call
      // respawn. Latching here condemned every later embed in the process to
      // the inline, main-thread-blocking path (~6s vs ~36ms per embed,
      // WI-4196's numbers) for the rest of its life — one hiccup silently
      // undoing this whole module. That is the exact stickiness EI-16184
      // removed from the per-closure booleans; it had survived one level up,
      // at module scope, while this file's own doc claimed a crash self-heals.
      //
      // Measured before the fix: injecting one 'error' event left
      // getWorkerState() at `disabled: true` and the next embed rejected
      // `worker disabled` permanently.
      _worker = null;
      _workerReady = null;
      _refd = false;
      if (!initialized) rejectReady(err);
    });
    _worker.on('exit', (code) => {
      if (code !== 0 && !initialized) {
        rejectReady(new Error(`worker exited with code ${code} before ready`));
      }
      // WI-37683: a worker that exits with requests still pending must REJECT
      // them. Dropping them silently is what turned the old unref bug into a
      // process that exited 0 with neither a vector nor an error — the caller's
      // promise simply never settled, so there was nothing to notice.
      if (_pending.size > 0) {
        const err = new Error(`embedder worker exited with code ${code} while ${_pending.size} request(s) were in flight`);
        for (const [, p] of _pending) p.reject(err);
        _pending.clear();
      }
      _worker = null;
      _workerReady = null;
      _refd = false;
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
    // Ref BEFORE posting: between the post and the reply the caller is awaiting
    // a Promise, which is not loop work — an unref'd worker would leave the loop
    // looking idle and let `beforeExit` terminate this very request (WI-37683).
    syncWorkerRef();
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
  // WI-37683: reject, never silently drop. `terminate()` normally fires the
  // `exit` handler above (which rejects), but that is not guaranteed to have
  // run by the time we get here, and a `_pending.clear()` on its own is exactly
  // how a stranded caller ends up awaiting a promise that settles never.
  if (_pending.size > 0) {
    const err = new Error(`embedder worker was shut down while ${_pending.size} request(s) were in flight`);
    for (const [, p] of _pending) p.reject(err);
  }
  _pending.clear();
  _refd = false;
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
 * Scripts that never call `process.exit()` at all no longer need this: the worker holds the loop
 * open only while a request is actually in flight and releases it the moment the last one lands
 * (`syncWorkerRef`), so a script with no other pending work exits on its own once it's done,
 * running the same graceful worker teardown automatically (see the `beforeExit` hook below). This
 * export exists for the case an explicit `process.exit()` is unavoidable.
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
  // `beforeExit` listeners cannot be declared `async`, but returning the
  // promise (rather than `void`-ing it) is still correct and matters for two
  // reasons: (1) it is what lets `Worker#terminate()`'s own pending work keep
  // the event loop alive long enough to finish — Node re-checks for idle
  // after a `beforeExit` listener returns, and a still-in-flight termination
  // naturally does that on its own, the return value itself isn't what
  // Node awaits; (2) it makes the hook directly testable by invoking the
  // registered listener function and awaiting what it returns, instead of
  // needing to emit a real process-wide `beforeExit`. A worker left torn
  // down after this fires is harmless either way — `ensureWorker` respawns
  // one lazily on the next real embed call, exactly as it already does after
  // any other worker crash/reset.
  //
  // The pending guard is belt-and-braces: `syncWorkerRef` should make an idle
  // loop impossible while a request is in flight, so this branch is unreachable
  // by design. It stays because the failure it prevents (terminating a worker
  // mid-request) was SILENT for the whole life of this module, and because a
  // future ref bug would otherwise re-open it (WI-37683).
  _beforeExitListener = async () => {
    if (_pending.size > 0) return;
    await _resetWorker();
  };
  process.on('beforeExit', _beforeExitListener);
}

/** Test-only: remove the installed `beforeExit` hook (if any) and clear the
 *  guard, so a test can assert on install-lifecycle behavior from a clean
 *  baseline instead of inheriting whatever an earlier test in the same file
 *  already installed. Mirrors `_resetFallbackWarnForTest` above. */
export function _resetBeforeExitHookForTest(): void {
  if (_beforeExitListener) process.off('beforeExit', _beforeExitListener);
  _beforeExitListener = null;
  _beforeExitHookInstalled = false;
}

/** Telemetry for /settings/user/memory diagnostics. */
export function getWorkerState(): {
  alive: boolean;
  disabled: boolean;
  pendingCount: number;
  /**
   * Whether the worker is currently holding the event loop open. The invariant
   * is `keepAlive === (pendingCount > 0)`; a worker NOT holding the loop while
   * it still owes an answer is the WI-37683 defect (`beforeExit` fires
   * mid-request and terminates the worker), so this is exported to be asserted
   * rather than inferred.
   */
  keepAlive: boolean;
} {
  return {
    alive: _worker !== null,
    disabled: _workerDisabled,
    pendingCount: _pending.size,
    keepAlive: _refd,
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

/** Ceiling on the intra-op pool, whatever the host size (WI-3792): a 128-core
 *  box must never get a 128-thread spin pool back. */
export const MAX_INTRA_OP_THREADS = 4;

/** Embedding is BACKGROUND work behind an idle UI, so it may claim at most
 *  ~1/Nth of the host. See `resolveIntraOpNumThreads` for why this exists. */
export const BACKGROUND_HOST_SHARE_DIVISOR = 4;

/**
 * Size the ONNX intra-op thread pool RELATIVE TO THE HOST.
 *
 * ONNX Runtime defaults intraOp threads to EVERY core and spin-waits them — on
 * the 128-core dev host each embedding-loading process grew a ~128-thread spin
 * pool (loadavg 2000-3000 host stutter, WI-3792). That incident was fixed with
 * a hardcoded `intraOpNumThreads: 4`, which capped the big host and left the
 * SMALL host unfixed: 4 is an ABSOLUTE constant, so it never scaled DOWN.
 *
 * EI-20493854163389792: the packaged 0.0.16 desktop on a fresh 8-vCPU Ubuntu
 * guest measured 415.2% average process CPU (360/418/422/465/411 over 5×1s)
 * while the first-run UI sat idle — ≈4 saturated intra-op threads plus main.
 * `/api/health/deep` simultaneously reported loopLag pressure=ok, which is the
 * tell: the burn is on NATIVE ORT threads, not the JS event loop. First run is
 * the worst case because the whole seeded corpus is unembedded, so the
 * embed-backfill sweep keeps the pool hot; and the packaged default runs the
 * embedder IN-PROCESS (the embed sidecar is opt-in via PAPERCUSP_EMBED_SIDECAR),
 * which is why the cost lands on the operator's own PID. On that guest the
 * hardcoded 4 is HALF the machine — a first-run user watching an idle screen
 * sees the app peg their CPU.
 *
 * So the cap is now a SHARE, not a constant: at most `MAX_INTRA_OP_THREADS`,
 * and never more than 1/`BACKGROUND_HOST_SHARE_DIVISOR` of the host, floor 1.
 *   1-7 cores → 1  ·  8 → 2  ·  16+ → 4 (WI-3792's ceiling, unchanged)
 * Mirrored in local-embedder-worker.script.mjs (plain-JS worker, can't import
 * this) — keep the two in sync; ort-thread-cap.test.ts is the mechanical guard.
 */
export function resolveIntraOpNumThreads(hostCores: number): number {
  if (!Number.isFinite(hostCores) || hostCores < 1) return 1;
  return Math.max(
    1,
    Math.min(MAX_INTRA_OP_THREADS, Math.floor(hostCores / BACKGROUND_HOST_SHARE_DIVISOR)),
  );
}

/** Host parallelism, defensively: `availableParallelism` respects the CPU
 *  affinity mask (so a pinned/containerised process sizes to what it may
 *  actually use) and exists on every Node we ship, but a 0/NaN reading must
 *  degrade to the single-thread floor rather than to ORT's all-cores default. */
function hostParallelism(): number {
  try {
    return availableParallelism();
  } catch {
    return 1;
  }
}

export const ORT_SESSION_OPTIONS: {
  readonly intraOpNumThreads: number;
  readonly interOpNumThreads: number;
} = {
  intraOpNumThreads: resolveIntraOpNumThreads(hostParallelism()),
  interOpNumThreads: 1,
};

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
