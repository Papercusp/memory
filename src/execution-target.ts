/**
 * execution-target.ts — what device the EMBEDDERS are actually running on.
 *
 * ## Why this exists
 *
 * `@papercusp/rerank` has reported `activeExecutionTarget()` /
 * `rerankExecutionHealth()` on the sidecar's `/healthz` since plan D-006, so a
 * reranker silently demoted from GPU to CPU is visible without anyone opening a
 * process. The embedders had no equivalent, and the cost of that asymmetry is
 * measured, not hypothetical: between 2026-08-27 and 2026-09-06, FOURTEEN
 * separate work-items were filed against embed latency (query embeds blowing the
 * 4000ms `WORK_ITEM_EMBED_TIMEOUT_MS` budget, taking semantic recall and
 * work-item dedup down fleet-wide with them). Every one of those filings looked
 * at the request path — the sidecar's bounded-concurrency scheduler, its FIFO
 * queue, the client-side query-embed cache — because the request path is the
 * only thing `/healthz` could describe.
 *
 * A 14-probe discriminant on 2026-09-06 refuted all three of those mechanisms at
 * once: embed latency does NOT track queue depth (12 of 14 probes dispatched at
 * depth 0, and the three WORST results — 8521, 8016 and 6590 ms — all landed
 * there, while a *saturated* sample returned in 1543ms). The real mechanism was
 * never in the scheduler. It is that a single forward pass on this host costs
 * ~1.5-2.5s median with a tail past 8.5s, because the embedders run on the CPU:
 * no embedder requests a device, so `pipeline()` takes onnxruntime's CPU default,
 * and the shipped onnxruntime build bundles no GPU execution provider anyway.
 *
 * Establishing that took reading `/proc/<pid>/maps` of the live sidecar to prove
 * which shared objects were mapped. That is not a thing the 15th filing should
 * have to do. This module makes the same fact a field on `/healthz`.
 *
 * ## What it reports, and what it deliberately does not claim
 *
 * Two things are true at different confidence levels, and conflating them is how
 * a detector becomes a liar:
 *
 *   1. **The embed path requests no device.** Static, always knowable, and the
 *      actual root cause. `EMBED_REQUESTED_EXECUTION` states it, and
 *      `execution-target.test.ts` PINS it against the embedder sources — if
 *      someone starts passing `device:`/`dtype:` to `pipeline()`, that test
 *      fails rather than letting this constant quietly become false (the
 *      derived-truth ladder's rung 2).
 *
 *   2. **This onnxruntime build bundles no GPU execution provider.** Runtime
 *      truth, from `onnxruntime-node`'s own `listSupportedBackends()`, which
 *      answers e.g. `[{cpu,bundled:true},{webgpu,bundled:true},
 *      {cuda,bundled:false},{tensorrt,bundled:false}]`. Derived, never
 *      hand-maintained — rung 1.
 *
 * The probe is async and the sidecar's `/healthz` handler is sync, so
 * `gpuBundled` is `null` — never a convenient `false` — until
 * `ensureEmbedBackendsProbed()` resolves. An unresolved probe is UNKNOWN, and
 * saying so is the whole point: a detector that reports a confident `false`
 * from a measurement it never took reproduces the failure it was built to end.
 *
 * This module reports; it does not choose. Moving the embedders onto a GPU is an
 * infrastructure decision (shared `node_modules` under ~100 live agents, and a
 * GPU already ~86% utilized by another workload) and is deliberately NOT taken
 * here. Raising the timeout would only hide the 8.5s tail.
 */

import { pinModuleState } from '@papercusp/module-singleton';
import { dynamicImport } from './dynamic-import';
import { ORT_SESSION_OPTIONS } from './local-embedder-worker';

/** The ONNX Runtime binding transformers.js uses under Node. */
export const EMBED_ORT_PACKAGE = 'onnxruntime-node';

/** One entry of `onnxruntime-node`'s `listSupportedBackends()`. */
export type EmbedBackend = { name: string; bundled: boolean };

/** The (device, dtype) pair the embedders actually run on, and the basis for it. */
export type EmbedExecutionTarget = {
  device: string;
  /** `null` = no dtype is pinned, so the transformers.js default applies. */
  dtype: string | null;
  /** What evidence supports `device` — read this before quoting the field. */
  why: string;
};

/** Whether the bundled-backend probe has run, and how it went. */
export type EmbedExecutionProbe = 'pending' | 'ok' | 'failed';

export type EmbedExecutionHealth = {
  /**
   * What the embed path ASKS FOR. Both `null` is the finding, not a placeholder:
   * no embedder passes `device` or `dtype` to `pipeline()`, only `session_options`.
   */
  requested: { device: string | null; dtype: string | null };
  /** What it therefore runs on. */
  active: EmbedExecutionTarget;
  /**
   * Whether ANY GPU-capable execution provider is bundled in this build.
   * `null` means the probe has not resolved — UNKNOWN, not "no".
   */
  gpuBundled: boolean | null;
  /** The raw `listSupportedBackends()` answer; `null` until the probe resolves. */
  backends: EmbedBackend[] | null;
  probe: EmbedExecutionProbe;
  probeError: string | null;
  /**
   * The thread caps that DO get set. These govern CPU forward-pass latency
   * directly, so they belong beside the device that makes them load-bearing.
   */
  sessionOptions: { intraOpNumThreads: number; interOpNumThreads: number };
};

/**
 * The device/dtype the embed path requests. See the header: this is PINNED by
 * `execution-target.test.ts` against the embedder sources, so it cannot drift
 * into a lie the way a hand-maintained `exists:` boolean does.
 */
export const EMBED_REQUESTED_EXECUTION: { device: string | null; dtype: string | null } = {
  device: null,
  dtype: null,
};

/** Backends that would put a forward pass anywhere other than the CPU. */
const GPU_BACKENDS = new Set(['cuda', 'tensorrt', 'webgpu', 'dml', 'coreml', 'rocm']);

type ExecutionTargetState = {
  probe: EmbedExecutionProbe;
  backends: EmbedBackend[] | null;
  probeError: string | null;
  inFlight: Promise<void> | null;
};

const STATE_KEY = '@papercusp/memory.embed-execution-target';

const state = pinModuleState<ExecutionTargetState>(STATE_KEY, () => ({
  probe: 'pending',
  backends: null,
  probeError: null,
  inFlight: null,
}));

type OrtModule = {
  listSupportedBackends?: () => unknown;
  default?: { listSupportedBackends?: () => unknown };
};

/**
 * Normalize `listSupportedBackends()` output without trusting its shape. A
 * binding that answers something unexpected must degrade to `probe:'failed'`
 * with the reason attached — never to a plausible-looking empty list, which
 * would render as "no GPU backend bundled" and be indistinguishable from a real
 * measurement.
 */
function normalizeBackends(raw: unknown): EmbedBackend[] {
  if (!Array.isArray(raw)) throw new Error(`listSupportedBackends() returned ${typeof raw}, expected an array`);
  return raw.map((entry, i) => {
    const e = entry as { name?: unknown; bundled?: unknown };
    if (!e || typeof e.name !== 'string') {
      throw new Error(`listSupportedBackends()[${i}] has no string 'name'`);
    }
    return { name: e.name, bundled: e.bundled === true };
  });
}

/**
 * Resolve the bundled-backend list once per process and cache it. Idempotent,
 * concurrency-safe, and never throws: a failure is recorded as `probe:'failed'`
 * with `probeError` so `/healthz` can say the measurement did not happen.
 *
 * Call it at sidecar startup so the first `/healthz` after warm-up is already
 * answering from a real measurement.
 */
export function ensureEmbedBackendsProbed(): Promise<void> {
  if (state.probe !== 'pending') return Promise.resolve();
  if (state.inFlight) return state.inFlight;
  const run = (async () => {
    try {
      const mod = await dynamicImport<OrtModule>(EMBED_ORT_PACKAGE);
      const list = mod?.listSupportedBackends ?? mod?.default?.listSupportedBackends;
      if (typeof list !== 'function') {
        throw new Error(`${EMBED_ORT_PACKAGE} exposes no listSupportedBackends()`);
      }
      state.backends = normalizeBackends(list.call(mod?.listSupportedBackends ? mod : mod.default));
      state.probeError = null;
      state.probe = 'ok';
    } catch (e) {
      state.backends = null;
      state.probeError = e instanceof Error ? e.message : String(e);
      state.probe = 'failed';
    }
  })().finally(() => {
    state.inFlight = null;
  });
  state.inFlight = run;
  return run;
}

/** `true`/`false` once measured; `null` while the probe is pending or failed. */
export function embedGpuBundled(): boolean | null {
  if (state.probe !== 'ok' || !state.backends) return null;
  return state.backends.some((b) => b.bundled && GPU_BACKENDS.has(b.name.toLowerCase()));
}

/**
 * The (device, dtype) pair the embedders are running on.
 *
 * `device` is `cpu` in every branch — because nothing requests otherwise, and
 * transformers.js defaults to CPU under Node — so what varies is `why`, i.e.
 * how well-evidenced that answer is. Read it before quoting the pair.
 */
export function embedExecutionTarget(): EmbedExecutionTarget {
  const gpu = embedGpuBundled();
  const base = 'the embed path requests no device, so transformers.js applies its CPU default';
  if (gpu === false) {
    return {
      device: 'cpu',
      dtype: EMBED_REQUESTED_EXECUTION.dtype,
      why: `${base}; this onnxruntime build also bundles no GPU execution provider, so no other device is reachable`,
    };
  }
  if (gpu === true) {
    return {
      device: 'cpu',
      dtype: EMBED_REQUESTED_EXECUTION.dtype,
      why: `${base} — note a GPU execution provider IS bundled here, so this CPU target is a choice nobody made explicitly, not a hardware limit`,
    };
  }
  return {
    device: 'cpu',
    dtype: EMBED_REQUESTED_EXECUTION.dtype,
    why: `${base}; the bundled-backend probe has not resolved, so whether a GPU provider exists is UNKNOWN`,
  };
}

/**
 * The full embed-execution report for `/healthz`.
 *
 * Sync, so the sidecar's sync handler can call it. It reports cached probe
 * state rather than blocking; `probe` says whether that state is a measurement.
 */
export function embedExecutionHealth(): EmbedExecutionHealth {
  return {
    requested: { ...EMBED_REQUESTED_EXECUTION },
    active: embedExecutionTarget(),
    gpuBundled: embedGpuBundled(),
    backends: state.backends ? state.backends.map((b) => ({ ...b })) : null,
    probe: state.probe,
    probeError: state.probeError,
    sessionOptions: {
      intraOpNumThreads: ORT_SESSION_OPTIONS.intraOpNumThreads,
      interOpNumThreads: ORT_SESSION_OPTIONS.interOpNumThreads,
    },
  };
}

/** Test-only: drop the cached probe so a test can drive it from a clean state. */
export function _resetEmbedExecutionProbe(): void {
  state.probe = 'pending';
  state.backends = null;
  state.probeError = null;
  state.inFlight = null;
}
