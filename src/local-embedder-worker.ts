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

import { pinModuleState } from '@papercusp/module-singleton';
import { Worker } from 'node:worker_threads';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { availableParallelism } from 'node:os';
import { fileURLToPath } from 'node:url';
import { resolve, dirname, isAbsolute, join } from 'node:path';
import { dynamicImport } from './dynamic-import';
import { applyWorkerDeviceReport, constructEmbedPipeline, currentEmbedDeviceDecision } from './embed-device';

interface PendingRequest {
  resolve: (v: number[]) => void;
  reject: (err: Error) => void;
  onInputTrace?: (trace: WorkerInputTrace) => void;
  onInferenceTrace?: (trace: WorkerInferenceTrace) => void;
  inferenceTraces?: WorkerInferenceTrace[];
  onNativeInferenceTrace?: (trace: WorkerNativeInferenceTrace) => void;
  nativeInferenceTraces?: WorkerNativeInferenceTrace[];
  inputTraceCount?: number;
  inputTraceError?: Error;
}

/** Explicit qualification evidence, emitted before the graph executes. */
export interface WorkerInputTrace {
  model: string; device: 'cpu' | 'cuda'; observedAt: string;
  inputShape: number[]; inputIds: number[]; attentionMask: number[];
  request?: WorkerRequestIdentity;
}

export interface WorkerRequestIdentity {
  requestId: number; attempt: number; processId: number; workerThreadId: number;
  /** Linux OS TID, distinct from Node's workerThreadId; unavailable elsewhere. */
  nativeThreadId: number | null;
}

/** Graph-call boundaries, not proof that a CUDA kernel belongs to this request.
 * node-hrtime must be calibrated to the profiler's clock before joining. */
export interface WorkerInferenceTrace extends WorkerRequestIdentity {
  model: string; device: 'cpu' | 'cuda'; observedAt: string;
  clock: 'node-hrtime'; monotonicNs: string; phase: 'start' | 'end';
  outcome?: 'success' | 'error';
}

/** Boundaries around the synchronous native addon call, after its JS queue.
 * This still requires profiler clock calibration before attributing kernels. */
export interface WorkerNativeInferenceTrace extends WorkerInferenceTrace {
  runIndex: number; runTag: string;
  rawClock?: WorkerRawClockSample;
  clockProbeError?: string;
  runtime?: WorkerNativeRuntimeSample;
  runtimeProbeError?: string;
}

export interface WorkerNativeRuntimeSample {
  platform: 'linux'; clock: 'node-hrtime'; beforeNs: string; afterNs: string;
  /** Optional for historical observations. Required by complete loader closure
   * qualification; these addresses come from this process's kernel auxv. */
  loaderProcess?: { executablePath: string; interpreterPath: string; nodeModuleVersion?: string; programHeaderAddress: string;
    programHeaderEntryBytes: number; programHeaderCount: number; entryAddress: string; interpreterBaseAddress: string;
    vdso: { startAddress: string; endAddress: string; fileOffset: string; permissions: string;
      bytes: number; sha256: string; origin: 'kernel-auxv-AT_SYSINFO_EHDR' } };
  libraries: { path: string; bytes: number; sha256: string; mappedDevice: string; mappedInode: string;
    /** All segments of the selected file, including non-executable ELF headers.
     * Older saved observations lack these and cannot qualify address joins. */
    mappedRanges?: { startAddress: string; endAddress: string; fileOffset: string; permissions: string }[] }[];
  gpuMemory: { status: 'not-applicable' } | { status: 'unknown'; beforeNs: string; afterNs: string; error: string }
    | { status: 'measured'; scope: 'all-nvidia-smi-devices'; beforeNs: string; afterNs: string;
      executable: { path: string; bytes: number; sha256: string }; cudaVisibleDevices: string | null;
      devices: { uuid: string; pciBusId: string; totalMiB: number; usedMiB: number; freeMiB: number }[] };
}

/** Validate the observation, separately from proving a complete library
 * closure or binding one of the sampled GPUs to the model's actual device. */
export function validNativeRuntimeSample(event: WorkerNativeInferenceTrace): boolean {
  const sample = event.runtime, ns = (n: unknown): n is string => typeof n === 'string' && /^[1-9]\d*$/.test(n);
  const fp = (f: { path: string; bytes: number; sha256: string }) => f && typeof f.path === 'string' && f.path.startsWith('/')
    && Number.isSafeInteger(f.bytes) && f.bytes > 0 && typeof f.sha256 === 'string' && /^[a-f0-9]{64}$/.test(f.sha256);
  if (!sample || event.runtimeProbeError !== undefined || sample.platform !== 'linux' || sample.clock !== 'node-hrtime'
    || !ns(event.monotonicNs) || !ns(sample.beforeNs) || !ns(sample.afterNs) || BigInt(sample.beforeNs) > BigInt(sample.afterNs)
    || (event.phase === 'start' && BigInt(sample.afterNs) > BigInt(event.monotonicNs))
    || (event.phase === 'end' && BigInt(sample.beforeNs) < BigInt(event.monotonicNs))
    || !Array.isArray(sample.libraries) || !sample.libraries.length
    || sample.libraries.some(f=>!fp(f) || typeof f.mappedDevice !== 'string' || typeof f.mappedInode !== 'string'
      || !/^[\da-f]+:[\da-f]+$/i.test(f.mappedDevice) || !/^[1-9]\d*$/.test(f.mappedInode))
    || new Set(sample.libraries.map(f=>f.path)).size !== sample.libraries.length
    ) return false;
  for (const file of sample.libraries) {
    if (file.mappedRanges === undefined) continue;
    if (!Array.isArray(file.mappedRanges) || !file.mappedRanges.length) return false;
    let previousEnd = 0n;
    for (const range of file.mappedRanges) {
      if (!range || ![range.startAddress, range.endAddress, range.fileOffset].every(value =>
        typeof value === 'string' && /^[\da-f]+$/i.test(value))
        || typeof range.permissions !== 'string' || !/^[r-][w-][x-][ps]$/.test(range.permissions)) return false;
      const start = BigInt('0x'+range.startAddress), end = BigInt('0x'+range.endAddress);
      if (start < previousEnd || start >= end) return false;
      previousEnd = end;
    }
  }
  const gpu = sample.gpuMemory;
  if (event.device === 'cpu') return gpu?.status === 'not-applicable';
  if (!gpu || !['measured','unknown'].includes(gpu.status) || gpu.status === 'not-applicable'
    || !ns(gpu.beforeNs) || !ns(gpu.afterNs) || BigInt(gpu.beforeNs) < BigInt(sample.beforeNs)
    || BigInt(gpu.beforeNs) > BigInt(gpu.afterNs) || BigInt(gpu.afterNs) > BigInt(sample.afterNs)) return false;
  if (gpu.status === 'unknown') return typeof gpu.error === 'string' && !!gpu.error;
  return gpu.scope === 'all-nvidia-smi-devices' && fp(gpu.executable)
    && (gpu.cudaVisibleDevices === null || typeof gpu.cudaVisibleDevices === 'string')
    && Array.isArray(gpu.devices) && gpu.devices.length > 0
    && gpu.devices.every(d=>typeof d.uuid === 'string' && d.uuid.startsWith('GPU-') && typeof d.pciBusId === 'string'
      && /^[\da-f]+:[\da-f]+:[\da-f]+\.[\da-f]+$/i.test(d.pciBusId)
      && [d.totalMiB,d.usedMiB,d.freeMiB].every(Number.isFinite) && d.totalMiB > 0 && d.usedMiB >= 0 && d.freeMiB >= 0
      && d.usedMiB <= d.totalMiB && d.freeMiB <= d.totalMiB)
    && new Set(gpu.devices.map(d=>d.uuid)).size === gpu.devices.length;
}

export interface WorkerRawClockSample {
  clock: 'linux-clock-monotonic-raw'; rawNs: string;
  monotonicBeforeNs: string; monotonicAfterNs: string;
  nodeBeforeNs: string; nodeAfterNs: string;
  executable: { path: string; bytes: number; sha256: string }; pythonVersion: string;
}

function validRequestIdentity(value: WorkerRequestIdentity, id: number): boolean {
  return value.requestId === id && Number.isSafeInteger(value.requestId) && value.requestId >= 0
    && [value.attempt, value.processId, value.workerThreadId]
    .every((n) => Number.isSafeInteger(n) && n > 0)
    && (value.nativeThreadId === null || (Number.isSafeInteger(value.nativeThreadId) && value.nativeThreadId > 0));
}

interface WorkerState {
  worker: Worker | null;
  workerReady: Promise<void> | null;
  nextId: number;
  pending: Map<number, PendingRequest>;
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
  workerDisabled: boolean;
  /** Guards the process-level `beforeExit` hook below so it is registered at
   *  most once for the life of the process, no matter how many times
   *  `ensureWorker()` (re)spawns a worker. */
  beforeExitHookInstalled: boolean;
  /** The actual listener function, kept so a test can remove exactly it. */
  beforeExitListener: (() => Promise<void>) | null;

  /**
   * Whether the worker is currently holding the event loop open. Mirrors the last
   * `ref()`/`unref()` we issued, because `Worker` exposes no way to read it back.
   */
  refd: boolean;
  lastFallbackWarnAt: number;
  /** A graceful recycle in progress (`recycleEmbedWorker`); new embeds wait on it. */
  recycling: Promise<void> | null;
  /** Resolvers waiting for `pending` to empty (a recycle's drain). */
  drainWaiters: Array<() => void>;
  /**
   * Outcome of pinning the ONNX native binding in the spawning thread
   * (`pinOnnxRuntimeBinding`, WI-10005090). Null until the first spawn tries it.
   * Optional because a module record from an older build may have created this
   * pinned state object before the field existed.
   */
  onnxBindingPin?: OnnxBindingPin | null;
  /** The armed idle-unload timer (`armIdleUnload`, WI-10005070), if any. */
  idleTimer?: ReturnType<typeof setTimeout> | null;
  /** How many times the idle unload has released the worker in this process. */
  idleUnloads?: number;
  /**
   * Workers `_resetWorker` detached while they still owed answers (WI-10006567):
   * each is finishing the native inference already running in it and is
   * terminated once it reports `retired`. Keyed by the Worker so its own
   * handlers find their record. Optional for the same older-record reason as
   * `onnxBindingPin`.
   */
  retiring?: Map<Worker, RetiringWorker>;
}

type RetireOutcome = 'retired' | 'exited' | 'errored' | 'timed-out';
interface RetiringWorker {
  /** Resolves once the worker is terminated (or had already exited). */
  done: Promise<void>;
  settle: (outcome: RetireOutcome) => void;
}

/**
 * Upper bound on how long `_resetWorker` waits for a retiring worker to finish
 * its in-flight inference before terminating it anyway (WI-10006567). A shutdown
 * must never hang on a wedged worker, and the sidecar's SIGTERM path awaits
 * this before `process.exit` inside systemd's 90 s stop window.
 */
export const EMBED_WORKER_RETIRE_DRAIN_MS = 30_000;

/** What `pinOnnxRuntimeBinding` achieved. Never thrown — reported. */
export type OnnxBindingPin =
  | { status: 'pinned'; path: string }
  | { status: 'unavailable'; reason: string }
  | { status: 'failed'; reason: string };

// tsx can evaluate this module through both CJS and ESM in one process.
// Shutdown and health reads must see the worker started through either loader.
const state = pinModuleState<WorkerState>('@papercusp/memory.local-embedder-worker', () => ({
  worker: null, workerReady: null, nextId: 0, pending: new Map(),
  workerDisabled: false, beforeExitHookInstalled: false, beforeExitListener: null,
  refd: false, lastFallbackWarnAt: 0, recycling: null, drainWaiters: [],
  onnxBindingPin: null, idleTimer: null, idleUnloads: 0,
}));

/** Env override for the idle-unload window, in ms. `0` disables the unload. */
export const EMBED_WORKER_IDLE_MS_ENV = 'PAPERCUSP_EMBED_WORKER_IDLE_MS';
/**
 * Default idle-unload window: 0, i.e. OFF. Measured net-NEGATIVE on a real Server
 * (P-010 VM run, cap-p011, 2026-10-02, WI-10005070): terminating the worker
 * returns only ~0.35 GiB of the model to the OS, and a Server re-embeds within
 * ~15 min anyway, so the respawned worker allocates the model afresh. Three
 * tenants with a 10 min window settled at 4.1-4.4 GiB main-process anon against
 * 2.61 GiB for the never-unloaded control. A worker_thread shares the process
 * heap, so freed model memory stays in the allocator. Getting the memory back
 * needs the model in a separate PROCESS whose exit returns everything; until
 * then, keep the worker loaded. Set the env to a positive window only to
 * re-measure.
 */
export const DEFAULT_EMBED_WORKER_IDLE_MS = 0;

/** The effective idle-unload window. A malformed or negative value keeps the default. */
export function embedWorkerIdleMs(env: Record<string, string | undefined> = process.env): number {
  const raw = env[EMBED_WORKER_IDLE_MS_ENV];
  if (raw === undefined || raw.trim() === '') return DEFAULT_EMBED_WORKER_IDLE_MS;
  const ms = Number(raw);
  return Number.isFinite(ms) && ms >= 0 ? ms : DEFAULT_EMBED_WORKER_IDLE_MS;
}

function cancelIdleUnload(): void {
  if (state.idleTimer) clearTimeout(state.idleTimer);
  state.idleTimer = null;
}

/**
 * Release the worker, and with it the model, once the host has embedded nothing
 * for `embedWorkerIdleMs()` (WI-10005070). The next embed respawns a worker and
 * reloads the model (~2 s for EmbeddingGemma on CPU).
 *
 * WHY: every Server loads the model at boot (the transcript-search warm-up) and
 * an idle Papercusp Server held ~1 GB more anon with the embedder than without
 * it (plan agent-capacity-and-cost-gcp-2026-09-30, D-022). A local probe saw
 * 752 → 307 MB anon 5 s after terminate (WI-10005090), but on a real Server the
 * unload/re-embed cycle is a net LOSS (see DEFAULT_EMBED_WORKER_IDLE_MS), so it
 * is OFF by default and only arms when the env sets a positive window. An
 * active host never reaches the window, because every settled embed re-arms it.
 *
 * Armed only when the ONNX binding is pinned: without the pin a terminated
 * worker leaves the process unable to load the binding again (WI-10005090), so
 * unloading would trade memory for broken embedding. Unref'd, so it never
 * keeps a process alive. Goes through `recycleEmbedWorker`, so it waits for any
 * request in flight and new embeds wait for it.
 */
function armIdleUnload(): void {
  cancelIdleUnload();
  if (state.pending.size > 0 || !state.worker) return;
  if (state.onnxBindingPin?.status !== 'pinned') return;
  const ms = embedWorkerIdleMs();
  if (ms <= 0) return;
  const timer = setTimeout(() => {
    state.idleTimer = null;
    if (state.pending.size > 0 || !state.worker || state.recycling) return;
    state.idleUnloads = (state.idleUnloads ?? 0) + 1;
    // The ONE observable trace of this lever on a deployed host: without it the
    // only evidence an unload happened is an RSS drop (the P-010 VM run had to
    // infer it that way). Greppable as "idle unload".
    console.log(
      `[embed-worker] idle unload #${state.idleUnloads}: no embed for ${Math.round(ms / 1000)}s, releasing the worker and its model`,
    );
    recycleEmbedWorker().catch(() => {
      /* the next embed respawns regardless; nothing to report here */
    });
  }, ms);
  timer.unref?.();
  state.idleTimer = timer;
}

/** Idle-unload telemetry: the window, whether a timer is armed, and unloads so far. */
export function getEmbedWorkerIdleStats(): { idleMs: number; armed: boolean; idleUnloads: number } {
  return { idleMs: embedWorkerIdleMs(), armed: state.idleTimer != null, idleUnloads: state.idleUnloads ?? 0 };
}

/** Wake every drain waiter once nothing is in flight. Call after any `pending` removal. */
function notifyIfDrained(): void {
  if (state.pending.size > 0 || state.drainWaiters.length === 0) return;
  const waiters = state.drainWaiters.splice(0);
  for (const wake of waiters) wake();
}

function waitForDrain(): Promise<void> {
  if (state.pending.size === 0) return Promise.resolve();
  return new Promise<void>((resolve) => state.drainWaiters.push(resolve));
}

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
 * because `_resetWorker` used to CLEAR `state.pending` rather than reject it, the
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
 * Ref'ing only while `state.pending` is non-empty satisfies both: a script that
 * awaits an embedding stays alive until its vector arrives, then exits on its
 * own.
 */
function syncWorkerRef(): void {
  const want = state.pending.size > 0;
  if (!state.worker || want === state.refd) return;
  if (want) state.worker.ref();
  else state.worker.unref();
  state.refd = want;
}

const WORKER_SCRIPT_NAME = 'local-embedder-worker.script.mjs';

interface WorkerScriptPathProbe {
  /** Override the runtime module filename (used by the pseudo-path regression test). */
  filename?: string | null;
  /** Override the runtime module URL (used by the pseudo-path regression test). */
  metaUrl?: string | null;
  /** Override cwd and package resolution for deterministic tests. */
  cwd?: string | null;
  packageEntry?: string | null;
  exists?: (path: string) => boolean;
}

function runtimeFilename(): string | undefined {
  try {
    return typeof __filename === 'string' ? __filename : undefined;
  } catch {
    return undefined;
  }
}

function runtimeMetaUrl(): string | undefined {
  try {
    return typeof import.meta.url === 'string' ? import.meta.url : undefined;
  } catch {
    return undefined;
  }
}

function packageEntry(): string | undefined {
  try {
    // In tsx's CJS eval mode `__filename` is `[eval]`, but the imported module
    // can still have a working package resolver. In ESM, bind require to this
    // module URL instead. Both paths are optional: the co-located module path
    // and bounded cwd search below cover normal source/bundle layouts.
    const requireFn =
      typeof require === 'function'
        ? require
        : (() => {
            const url = runtimeMetaUrl();
            const filename = runtimeFilename();
            if (url?.startsWith('file:')) return createRequire(url);
            if (filename && isAbsolute(filename)) return createRequire(filename);
            return undefined;
          })();
    return requireFn?.resolve('@papercusp/memory');
  } catch {
    return undefined;
  }
}

function ancestorDirectories(start: string, maxDepth = 8): string[] {
  const roots: string[] = [];
  let current = resolve(start);
  for (let i = 0; i <= maxDepth; i += 1) {
    roots.push(current);
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return roots;
}

/**
 * Resolve the plain-JS worker script without trusting pseudo module paths.
 *
 * `tsx -e` gives imported CJS modules the literal `__filename` value
 * `[eval]`; `dirname('[eval]')` is `.`, which turns a bad module identity into
 * a plausible but wrong cwd-relative path. Candidate paths are therefore
 * validated before Worker construction, and the package/cwd layouts are
 * bounded fallbacks for eval and bundled entrypoints.
 */
export function resolveWorkerScriptPath(probe: WorkerScriptPathProbe = {}): string {
  const exists = probe.exists ?? existsSync;
  const filename = probe.filename === undefined ? runtimeFilename() : probe.filename ?? undefined;
  const metaUrl = probe.metaUrl === undefined ? runtimeMetaUrl() : probe.metaUrl ?? undefined;
  const cwd = probe.cwd === undefined ? process.cwd() : probe.cwd ?? undefined;
  const entry = probe.packageEntry === undefined ? packageEntry() : probe.packageEntry ?? undefined;
  const candidates: string[] = [];
  const add = (candidate: string | undefined): void => {
    if (!candidate) return;
    const absolute = resolve(candidate);
    if (!candidates.includes(absolute)) candidates.push(absolute);
  };

  // Prefer a real file URL, then a real absolute filename. A pseudo filename
  // such as `[eval]` is deliberately ignored rather than normalized.
  if (metaUrl?.startsWith('file:')) {
    try {
      add(join(dirname(fileURLToPath(metaUrl)), WORKER_SCRIPT_NAME));
    } catch {
      // Try the remaining bounded candidates.
    }
  }
  if (filename && isAbsolute(filename)) add(join(dirname(filename), WORKER_SCRIPT_NAME));

  if (entry) {
    add(join(dirname(entry), WORKER_SCRIPT_NAME));
    add(join(dirname(dirname(entry)), 'src', WORKER_SCRIPT_NAME));
    add(join(dirname(dirname(entry)), 'dist', WORKER_SCRIPT_NAME));
  }

  if (cwd) {
    for (const root of ancestorDirectories(cwd)) {
      add(join(root, WORKER_SCRIPT_NAME));
      add(join(root, 'src', WORKER_SCRIPT_NAME));
      add(join(root, 'dist', WORKER_SCRIPT_NAME));
      add(join(root, 'libs', 'generic', 'memory', 'src', WORKER_SCRIPT_NAME));
      add(join(root, 'libs', 'generic', 'memory', 'dist', WORKER_SCRIPT_NAME));
    }
  }

  const found = candidates.find((candidate) => {
    try {
      return exists(candidate);
    } catch {
      return false;
    }
  });
  if (found) return found;

  throw new Error(
    `Unable to locate ${WORKER_SCRIPT_NAME}; checked ${candidates.slice(0, 8).join(', ') || '(no candidates)'}.`,
  );
}

function workerPath(): string {
  return resolveWorkerScriptPath();
}

/**
 * Load the onnxruntime-node binding ONCE in the thread that spawns the worker,
 * before any worker loads it, and keep it loaded for the life of the process.
 *
 * WHY (WI-10005090, measured 2026-10-01, onnxruntime-node 1.24.3 / node 25.9):
 * the binding registers itself with Node only when its shared library is first
 * mapped. Node keeps a per-process map so later threads reuse that registration,
 * but drops the entry when the LAST thread holding the binding goes away, while
 * the library itself stays mapped. So once the embed worker terminates (a crash
 * respawn, or `recycleEmbedWorker()` on a device change), every later load in
 * the process fails with "Module did not self-register": the respawned worker
 * AND the embedders' main-thread fallback. Local embedding is then dead until
 * the process restarts. Holding one reference here keeps the entry alive, so a
 * respawned worker and the inline fallback both load normally. Probe:
 * .papercusp/scratch/wi5090-inline-fallback-probe.mjs (logs
 * ~/.cache/agent-capacity/wi5090/). The pin costs ~10 MB anon and holds no
 * model, so terminating the worker still frees the model's memory.
 *
 * `fromPath` is the worker script: the binding is resolved from the
 * transformers package that script imports, the same way transformers' own
 * `import 'onnxruntime-node'` resolves, so the pinned file is the file the
 * worker loads. Never throws: no transformers install is `unavailable`
 * (embedding cannot work there anyway), and a load error is `failed`, recorded
 * for `getOnnxBindingPin()`. Idempotent: the first outcome stands, because a
 * binding cannot be un-pinned and a failed pin will not succeed on retry.
 *
 * The reranker worker (libs/generic/rerank/src/local-reranker-worker.ts) pins
 * the same binding the same way; whichever spawns first holds it.
 */
export function pinOnnxRuntimeBinding(fromPath: string): OnnxBindingPin {
  if (!state.onnxBindingPin) state.onnxBindingPin = loadOnnxBindingFrom(fromPath);
  return state.onnxBindingPin;
}

function loadOnnxBindingFrom(fromPath: string): OnnxBindingPin {
  let transformersEntry: string;
  try {
    transformersEntry = createRequire(fromPath).resolve(TRANSFORMERS_PACKAGE);
  } catch {
    return { status: 'unavailable', reason: `${TRANSFORMERS_PACKAGE} is not resolvable from ${fromPath}` };
  }
  try {
    const fromTransformers = createRequire(transformersEntry);
    const path = fromTransformers.resolve('onnxruntime-node');
    fromTransformers('onnxruntime-node');
    return { status: 'pinned', path };
  } catch (err) {
    return { status: 'failed', reason: err instanceof Error ? err.message : String(err) };
  }
}

/** The pin outcome, or null when no worker has been spawned yet in this process. */
export function getOnnxBindingPin(): OnnxBindingPin | null {
  return state.onnxBindingPin ?? null;
}

/**
 * Forget the recorded pin outcome so a test can observe a first-time pin. The
 * binding itself stays loaded (Node's require cache holds it), so this cannot
 * re-open the WI-10005090 hazard; it only clears the record.
 */
export function _resetOnnxBindingPinForTest(): void {
  state.onnxBindingPin = null;
}

function ensureWorker(): Promise<void> {
  if (state.workerDisabled) return Promise.reject(new Error('worker disabled'));
  if (state.workerReady) return state.workerReady;

  state.workerReady = new Promise<void>((resolveReady, rejectReady) => {
    let spawned: Worker;
    try {
      const scriptPath = workerPath();
      // WI-10005090: before the first worker can load the ONNX binding (and so
      // before it can ever exit holding the last reference to it).
      pinOnnxRuntimeBinding(scriptPath);
      spawned = new Worker(scriptPath, {
        // The plain JS file needs no parent loader or entry-point flags.
        // A stdin parent's --input-type is invalid for this file worker.
        execArgv: [],
        //
        // The DEVICE is decided here, not in the worker (which is copied into
        // bundles as one file and cannot import embed-device.ts). A respawn
        // after a crash re-reads the decision, so it inherits any demotion the
        // previous worker reported instead of re-trying a GPU that failed.
        workerData: { device: currentEmbedDeviceDecision().device },
      });
    } catch (err) {
      state.workerDisabled = true;
      rejectReady(err as Error);
      return;
    }
    // WI-10006567: every handler below is scoped to THIS worker. `_resetWorker`
    // detaches a worker that still owes answers and lets it finish before
    // terminating it, while a replacement serves new requests. Request ids
    // restart at 0, so a retired worker's late `embed_ok` reaching
    // `state.pending` would resolve the REPLACEMENT's request with the wrong
    // vector, and its late `exit` would null the live handle. A retired worker's
    // messages therefore only ever feed its own retirement record.
    const w = spawned;
    state.worker = w;

    let initialized = false;
    w.on('message', (msg: { kind: string; id?: number; vector?: number[]; error?: string;
      trace?: WorkerInputTrace; inference?: WorkerInferenceTrace | WorkerNativeInferenceTrace }) => {
      if (state.worker !== w) {
        if (msg.kind === 'retired') state.retiring?.get(w)?.settle('retired');
        else if (msg.kind === 'ready') rejectReady(new Error('embedder worker was shut down before it became ready'));
        return;
      }
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
      // What a pipeline ACTUALLY constructed on (and any GPU→CPU fallback), so
      // /healthz reports the device in use rather than the one requested.
      if (msg.kind === 'device') {
        applyWorkerDeviceReport(msg);
        return;
      }
      if (typeof msg.id !== 'number') return;
      const p = state.pending.get(msg.id);
      if (!p) return;
      if (msg.kind === 'embed_inference' || msg.kind === 'embed_native_inference') {
        const native = msg.kind === 'embed_native_inference';
        const callback = native ? p.onNativeInferenceTrace : p.onInferenceTrace;
        if (callback) {
          try {
            const trace = msg.inference;
            if (!trace || !validRequestIdentity(trace, msg.id) || !['cpu', 'cuda'].includes(trace.device)
              || typeof trace.model !== 'string' || !Number.isFinite(Date.parse(trace.observedAt))
              || trace.clock !== 'node-hrtime' || typeof trace.monotonicNs !== 'string'
              || !/^[1-9]\d*$/.test(trace.monotonicNs)) throw new Error('invalid worker inference trace');
            const traces = native ? (p.nativeInferenceTraces ??= []) : (p.inferenceTraces ??= []);
            const prior = traces.at(-1);
            const nativeTrace = trace as WorkerNativeInferenceTrace;
            if (native && (!Number.isSafeInteger(nativeTrace.runIndex) || nativeTrace.runIndex < 1
              || nativeTrace.runTag !== `pc-embed:${trace.processId}:${trace.workerThreadId}:${trace.requestId}:${trace.attempt}:${nativeTrace.runIndex}`)) {
              throw new Error('invalid worker native inference identity');
            }
            if (native && nativeTrace.clockProbeError !== undefined) throw new Error(nativeTrace.clockProbeError);
            if (native && nativeTrace.runtimeProbeError !== undefined) throw new Error(nativeTrace.runtimeProbeError);
            if (native && nativeTrace.runtime !== undefined && !validNativeRuntimeSample(nativeTrace)) {
              throw new Error('invalid worker native runtime evidence');
            }
            if (native && nativeTrace.rawClock !== undefined) {
              const clock = nativeTrace.rawClock;
              if (clock.clock !== 'linux-clock-monotonic-raw'
                || [clock.rawNs, clock.monotonicBeforeNs, clock.monotonicAfterNs, clock.nodeBeforeNs, clock.nodeAfterNs]
                  .some((n) => typeof n !== 'string' || !/^[1-9]\d*$/.test(n))
                || BigInt(clock.nodeBeforeNs) > BigInt(clock.monotonicBeforeNs)
                || BigInt(clock.monotonicBeforeNs) > BigInt(clock.monotonicAfterNs)
                || BigInt(clock.monotonicAfterNs) > BigInt(clock.nodeAfterNs)
                || !clock.executable || typeof clock.executable.path !== 'string' || !Number.isSafeInteger(clock.executable.bytes)
                || clock.executable.bytes < 1 || !/^[a-f0-9]{64}$/.test(clock.executable.sha256)
                || typeof clock.pythonVersion !== 'string' || !clock.pythonVersion) throw new Error('invalid worker native clock evidence');
              if ((trace.phase === 'start' && BigInt(clock.nodeAfterNs) > BigInt(trace.monotonicNs))
                || (trace.phase === 'end' && BigInt(clock.nodeBeforeNs) < BigInt(trace.monotonicNs))) {
                throw new Error('invalid worker native clock order');
              }
            }
            if (trace.phase === 'start') {
              const nextAttempt = trace.attempt === (prior?.attempt ?? 0) + 1;
              const nextNativeRun = native && prior && trace.attempt === prior.attempt
                && nativeTrace.runIndex === (prior as WorkerNativeInferenceTrace).runIndex + 1;
              if (trace.outcome !== undefined || (!nextAttempt && !nextNativeRun)
                || (native && nextAttempt && nativeTrace.runIndex !== 1)
                || (prior && (prior.phase !== 'end' || (nextAttempt && prior.outcome !== 'error')
                  || BigInt(trace.monotonicNs) < BigInt(prior.monotonicNs)))) throw new Error('invalid worker inference trace order');
            } else if (trace.phase !== 'end' || !['success', 'error'].includes(trace.outcome ?? '') || !prior
              || prior.phase !== 'start' || ['requestId', 'attempt', 'processId', 'workerThreadId', 'nativeThreadId', 'model', 'device']
                .some((key) => trace[key as keyof WorkerInferenceTrace] !== prior[key as keyof WorkerInferenceTrace])
              || (native && (nativeTrace.runIndex !== (prior as WorkerNativeInferenceTrace).runIndex
                || nativeTrace.runTag !== (prior as WorkerNativeInferenceTrace).runTag))
              || BigInt(trace.monotonicNs) < BigInt(prior.monotonicNs)) throw new Error('invalid worker inference trace order');
            if (native) {
              p.nativeInferenceTraces!.push(nativeTrace);
              p.onNativeInferenceTrace!(nativeTrace);
            } else {
              p.inferenceTraces!.push(trace);
              p.onInferenceTrace!(trace);
            }
          } catch (error) { p.inputTraceError ??= error instanceof Error ? error : new Error(String(error)); }
        }
        return;
      }
      if (msg.kind === 'embed_input') {
        if (p.onInputTrace) {
          try {
            const trace = msg.trace;
            if (!trace || !['cpu', 'cuda'].includes(trace.device) || typeof trace.model !== 'string'
              || !Number.isFinite(Date.parse(trace.observedAt)) || !Array.isArray(trace.inputIds) || !trace.inputIds.length
              || trace.inputIds.some((n) => !Number.isSafeInteger(n) || n < 0)
              || (trace.request !== undefined && !validRequestIdentity(trace.request, msg.id))
              || JSON.stringify(trace.inputShape) !== JSON.stringify([1, trace.inputIds.length])
              || !Array.isArray(trace.attentionMask) || trace.attentionMask.length !== trace.inputIds.length
              || trace.attentionMask.some((n) => n !== 0 && n !== 1)) throw new Error('invalid worker input trace');
            p.onInputTrace(trace);
            p.inputTraceCount = (p.inputTraceCount ?? 0) + 1;
          } catch (error) { p.inputTraceError ??= error instanceof Error ? error : new Error(String(error)); }
        }
        return;
      }
      state.pending.delete(msg.id);
      // Release the loop as soon as the LAST request lands, so a one-off script
      // still exits on its own (WI-37683 — the other half of syncWorkerRef).
      syncWorkerRef();
      if (msg.kind === 'embed_ok' && Array.isArray(msg.vector)) {
        if (p.inputTraceError) p.reject(p.inputTraceError);
        else if (p.onInputTrace && !p.inputTraceCount) p.reject(new Error('worker returned a vector without requested input evidence'));
        else if (p.onInferenceTrace && p.inferenceTraces?.at(-1)?.outcome !== 'success') {
          p.reject(new Error('worker returned a vector without complete requested inference evidence'));
        }
        else if (p.onNativeInferenceTrace && p.nativeInferenceTraces?.at(-1)?.outcome !== 'success') {
          p.reject(new Error('worker returned a vector without complete requested native inference evidence'));
        }
        else p.resolve(msg.vector);
      } else {
        p.reject(new Error(msg.error ?? 'worker error'));
      }
      notifyIfDrained();
      // WI-10005070: the last request landed, so start the idle window.
      if (state.pending.size === 0) armIdleUnload();
    });
    w.on('error', (err) => {
      if (!initialized) rejectReady(err);
      // A retired worker dying mid-drain ends its retirement; it owes nobody now.
      if (state.worker !== w) { state.retiring?.get(w)?.settle('errored'); return; }
      // Reject every pending request — the worker crashed.
      for (const [, p] of state.pending) p.reject(err);
      state.pending.clear();
      notifyIfDrained();
      // EI-20012631851693581: deliberately NOT `state.workerDisabled`. A runtime
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
      state.worker = null;
      state.workerReady = null;
      state.refd = false;
      cancelIdleUnload();
    });
    w.on('exit', (code) => {
      if (code !== 0 && !initialized) {
        rejectReady(new Error(`worker exited with code ${code} before ready`));
      }
      if (state.worker !== w) { state.retiring?.get(w)?.settle('exited'); return; }
      // WI-37683: a worker that exits with requests still pending must REJECT
      // them. Dropping them silently is what turned the old unref bug into a
      // process that exited 0 with neither a vector nor an error — the caller's
      // promise simply never settled, so there was nothing to notice.
      if (state.pending.size > 0) {
        const err = new Error(`embedder worker exited with code ${code} while ${state.pending.size} request(s) were in flight`);
        for (const [, p] of state.pending) p.reject(err);
        state.pending.clear();
      }
      notifyIfDrained();
      state.worker = null;
      state.workerReady = null;
      state.refd = false;
      cancelIdleUnload();
    });
  });

  return state.workerReady;
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
  /** Explicit candidate contract; the default retains the SDK tokenizer. */
  tokenizerBackend?: 'rust';
  /** Opt-in raw tensor evidence for private qualification; normal calls emit none. */
  onInputTrace?: (trace: WorkerInputTrace) => void;
  /** Opt-in request/attempt/thread identities and graph-call clock boundaries. */
  onInferenceTrace?: (trace: WorkerInferenceTrace) => void;
  /** Fresh-worker qualification only: synchronous native Run boundaries. */
  onNativeInferenceTrace?: (trace: WorkerNativeInferenceTrace) => void;
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
  // A device change is recycling the worker: wait for it rather than land on
  // the old worker (which would keep embedding on the old device) or race its
  // termination.
  while (state.recycling) await state.recycling;
  await ensureWorker();
  if (!state.worker) throw new Error('worker not initialized');
  // WI-10005070: a request is starting, so the host is not idle.
  cancelIdleUnload();

  const id = state.nextId++;
  return new Promise<number[]>((resolveEmbed, rejectEmbed) => {
    state.pending.set(id, { resolve: resolveEmbed, reject: rejectEmbed, onInputTrace: opts.onInputTrace,
      onInferenceTrace: opts.onInferenceTrace, onNativeInferenceTrace: opts.onNativeInferenceTrace });
    // Ref BEFORE posting: between the post and the reply the caller is awaiting
    // a Promise, which is not loop work — an unref'd worker would leave the loop
    // looking idle and let `beforeExit` terminate this very request (WI-37683).
    syncWorkerRef();
    state.worker!.postMessage({
      kind: 'embed',
      id,
      text,
      model: opts.model,
      pooling: opts.pooling,
      normalize: opts.normalize,
      output: opts.output,
      tokenizerBackend: opts.tokenizerBackend,
      ...(opts.onInputTrace ? { traceInput: true } : {}),
      ...(opts.onInferenceTrace ? { traceInference: true } : {}),
      ...(opts.onNativeInferenceTrace ? { traceNativeInference: true } : {}),
    });
  });
}

/** Test seam — drop the worker and reset state. Kept as the underlying
 *  implementation `_resetWorker` for backward compatibility (tests import it
 *  directly); {@link shutdownLocalEmbedder} is the same function under a
 *  discoverable public name — see its doc for why both exist. */
export async function _resetWorker(opts: { drainMs?: number } = {}): Promise<void> {
  cancelIdleUnload();
  const old = state.worker;
  const owed = state.pending.size;
  // Detach first: from here the old worker's handlers see `state.worker !== w`
  // and never touch module state again (WI-10006567), and the next embed spawns
  // a replacement immediately instead of waiting out the drain below.
  state.worker = null;
  state.workerReady = null;
  state.workerDisabled = false;
  // WI-37683: reject, never silently drop — a `state.pending.clear()` on its own
  // is exactly how a stranded caller ends up awaiting a promise that settles never.
  if (state.pending.size > 0) {
    const err = new Error(`embedder worker was shut down while ${state.pending.size} request(s) were in flight`);
    for (const [, p] of state.pending) p.reject(err);
  }
  state.pending.clear();
  notifyIfDrained();
  state.refd = false;
  state.nextId = 0;
  if (old) {
    // Owing nothing means no inference is running in it, so it is safe to
    // terminate now; a worker that owed answers may be mid-run, and terminating
    // THAT one aborts the process (WI-10006567).
    // `opts?.`: this is exported as `shutdownLocalEmbedder` and handed around as
    // a callback, so a `.then(shutdownLocalEmbedder)` can pass null here.
    if (owed > 0) void retireWorker(old, opts?.drainMs ?? EMBED_WORKER_RETIRE_DRAIN_MS);
    else { try { await old.terminate(); } catch { /* already gone */ } }
  }
  // The contract callers rely on (the sidecar's SIGTERM path, every bench before
  // `process.exit`): once this resolves, no worker of this module is alive —
  // including one an EARLIER reset is still retiring.
  await Promise.all([...(state.retiring?.values() ?? [])].map((r) => r.done));
}

/**
 * Terminate a detached worker only once no ONNX inference is running in it
 * (WI-10006567).
 *
 * `Worker#terminate()` while an onnxruntime-node session is mid-run destroys the
 * worker's environment under the native call; when the run completes it throws a
 * `Napi::Error` nothing can catch, and the whole PROCESS aborts (SIGABRT, exit
 * 134). Measured 2026-10-06 in the P-007 real-weight robustness run, and
 * reproduced on BGE-small by `local-embedder-worker-retire.test.ts`; an idle
 * terminate is fine.
 *
 * So the worker is asked to retire: it starts no new inference, finishes the runs
 * already started, then answers `{kind:'retired'}`, and only then is terminated.
 * The wait is bounded by `drainMs`: past it the worker is terminated anyway with a
 * loud warning, because a shutdown must never hang on a wedged worker.
 */
function retireWorker(w: Worker, drainMs: number): Promise<void> {
  const retiring = (state.retiring ??= new Map());
  const existing = retiring.get(w);
  if (existing) return existing.done;
  let settle: (outcome: RetireOutcome) => void = () => {};
  const outcome = new Promise<RetireOutcome>((resolve) => { settle = resolve; });
  const timer = setTimeout(() => settle('timed-out'), drainMs);
  timer.unref?.();
  const done = outcome.then(async (how) => {
    clearTimeout(timer);
    if (how === 'timed-out') {
      console.warn(`[embed-worker] retiring worker did not report retired within ${drainMs}ms; terminating it anyway `
        + '(WI-10006567: terminating during a native run can abort the process)');
    }
    if (how !== 'exited') {
      try { await w.terminate(); } catch { /* already gone */ }
    }
    retiring.delete(w);
  });
  retiring.set(w, { done, settle });
  // Hold the loop while it drains: a host that went idle now would exit with
  // the native run still in flight, which aborts the same way.
  try { w.ref(); } catch { /* exiting already; its exit handler settles */ }
  try { w.postMessage({ kind: 'retire' }); } catch { settle('errored'); }
  return done;
}

/**
 * Restart the embedding worker so its pipelines are rebuilt on the CURRENT
 * device decision — the live half of a Settings device change (plan D-008).
 *
 * Graceful, unlike `_resetWorker`: new embeds wait on the recycle instead of
 * reaching the old worker, the requests already in flight finish on it, and only
 * then is it terminated. The next embed spawns a fresh worker, which reads
 * `currentEmbedDeviceDecision()` at spawn. Concurrent calls share one recycle.
 * With no worker running there is nothing to rebuild: the next spawn already
 * reads the new decision.
 */
export function recycleEmbedWorker(): Promise<void> {
  if (state.recycling) return state.recycling;
  if (!state.worker && !state.workerReady) return Promise.resolve();
  const run = (async () => {
    await waitForDrain();
    await _resetWorker();
  })().finally(() => {
    state.recycling = null;
  });
  state.recycling = run;
  return run;
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
  if (state.beforeExitHookInstalled) return;
  state.beforeExitHookInstalled = true;
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
  state.beforeExitListener = async () => {
    if (state.pending.size > 0) return;
    await _resetWorker();
  };
  process.on('beforeExit', state.beforeExitListener);
}

/** Test-only: remove the installed `beforeExit` hook (if any) and clear the
 *  guard, so a test can assert on install-lifecycle behavior from a clean
 *  baseline instead of inheriting whatever an earlier test in the same file
 *  already installed. Mirrors `_resetFallbackWarnForTest` above. */
export function _resetBeforeExitHookForTest(): void {
  if (state.beforeExitListener) process.off('beforeExit', state.beforeExitListener);
  state.beforeExitListener = null;
  state.beforeExitHookInstalled = false;
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
  /** A device-change recycle is draining the worker; new embeds are waiting on it. */
  recycling: boolean;
} {
  return {
    alive: state.worker !== null,
    disabled: state.workerDisabled,
    pendingCount: state.pending.size,
    keepAlive: state.refd,
    recycling: state.recycling !== null,
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
const FALLBACK_WARN_COOLDOWN_MS = 30_000;
export function warnEmbedFallback(model: string, err: unknown): void {
  const now = Date.now();
  if (now - state.lastFallbackWarnAt < FALLBACK_WARN_COOLDOWN_MS) return;
  state.lastFallbackWarnAt = now;
  if (process.env.NODE_ENV !== 'test') {
    console.warn(
      `[embed] ${model} worker path failed — falling back to inline (main-thread, blocks the event loop) for this call: ` +
        `${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/** Test-only: reset the fallback-warn cooldown so tests can assert on it independently. */
export function _resetFallbackWarnForTest(): void {
  state.lastFallbackWarnAt = 0;
}

export const LOCAL_EMBEDDER_MODEL = 'Xenova/bge-small-en-v1.5';
const TRANSFORMERS_PACKAGE = '@huggingface/transformers';

type TransformersModule = {
  env?: { allowRemoteModels: boolean; allowLocalModels: boolean };
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
 * P-314 customer-runtime boundary. A verified vm-release carries the complete,
 * digest-pinned model pack and must never turn a missing file into an implicit
 * vendor request on an embedding call. Dogfood/development remain unchanged.
 */
export function applyTransformersRuntimePolicy<T>(
  transformers: T & { env?: { allowRemoteModels: boolean; allowLocalModels: boolean } },
  env: NodeJS.ProcessEnv = process.env,
): T {
  if (
    env.PAPERCUSP_DISTRIBUTION_PROFILE !== 'vm-release' &&
    env.PAPERCUSP_TRANSFORMERS_LOCAL_ONLY !== '1'
  ) {
    return transformers;
  }
  if (!transformers.env) {
    throw new Error(
      '[vm-release] Transformers.js exposes no env policy; refusing a runtime that cannot disable remote model fetch',
    );
  }
  transformers.env.allowLocalModels = true;
  transformers.env.allowRemoteModels = false;
  return transformers;
}

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
      const transformers = applyTransformersRuntimePolicy(
        await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE),
      );
      pipelinePromise = constructEmbedPipeline(transformers, LOCAL_EMBEDDER_MODEL, ORT_SESSION_OPTIONS);
    }
    const pipe = await pipelinePromise;
    const result = await pipe(text, { pooling: 'mean', normalize: true });
    return Array.from(result.data);
  };
}
