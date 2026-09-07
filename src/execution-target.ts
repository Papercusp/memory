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
 * no embedder requests a device, so `pipeline()` takes onnxruntime's CPU default.
 *
 * ## ⚠ `bundled` DOES NOT MEAN AVAILABLE — read this before adding a field
 *
 * The first version of this module (2026-09-07, same day) reported a
 * `gpuBundled` boolean derived from `onnxruntime-node`'s
 * `listSupportedBackends()`, and documented it as "a GPU-capable execution
 * provider is compiled into this build". **That was wrong, and it was wrong in
 * the dangerous direction: it under-reports.** Measured the same day:
 *
 *   - A default `npm install onnxruntime-node@1.24.3` yields
 *     `[{cpu,bundled:true},{webgpu,bundled:true},{cuda,bundled:false},…]`.
 *   - Re-installing the SAME version with `ONNXRUNTIME_NODE_INSTALL_CUDA=v12`
 *     downloads `libonnxruntime_providers_cuda.so` (~315MB) — and
 *     `listSupportedBackends()` STILL reports `cuda: bundled:false`.
 *   - Yet in that tree `InferenceSession.create(model, { executionProviders:
 *     ['cuda'] })` **constructs successfully**, emitting genuine
 *     `CUDAExecutionProvider` graph-partition logs.
 *   - The `onnxruntime_binding.node` addon is byte-identical (384040 bytes) in
 *     both trees. Only the provider `.so` files differ.
 *
 * So `bundled` is **publish-time packaging metadata** — "does this provider ship
 * in the default npm tarball" — and it is NOT a capability probe. A host can run
 * CUDA perfectly while `bundled:false`. Reporting it as availability would have
 * told the next reader "no GPU here" on a working GPU box, which is exactly the
 * class of confidently-wrong detector this module exists to replace.
 *
 * What this module reports instead, at three honest confidence levels:
 *
 *   1. **The embed path requests no device.** Static, always knowable, and the
 *      actual root cause. `EMBED_REQUESTED_EXECUTION` states it, and
 *      `execution-target.test.ts` PINS it against the embedder sources.
 *   2. **Which provider libraries are present on disk.** The actionable
 *      availability signal (`providerLibraries` / `gpuProviderAvailable`) —
 *      this is what actually changed between the two trees above.
 *   3. **What ships by default.** `defaultBundledBackends`, kept because it
 *      explains WHY a provider is absent (nobody passed the install flag), but
 *      explicitly named so it can never again be read as availability.
 *
 * None of the three proves a GPU session would CONSTRUCT. Only constructing one
 * proves that, which is precisely what `@papercusp/rerank`'s `demoted` /
 * `demotionCause` pair reports — and on this host it reports
 * `demoted: true, cause: "OrtSessionOptionsAppendExecutionProvider_Cuda: Failed
 * to load shared library"`, the observable that started this correction.
 *
 * An unresolved probe reports `null` everywhere — never a convenient `false`.
 * UNKNOWN is in-band and asserted by test.
 *
 * This module reports; it does not choose. Moving the embedders onto a GPU stays
 * an infrastructure decision (shared `node_modules` under ~100 live agents, and a
 * GPU already ~88% utilized by another workload).
 */

import { createRequire } from 'node:module';
import { readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import { pinModuleState } from '@papercusp/module-singleton';
import { dynamicImport } from './dynamic-import';
import { ORT_SESSION_OPTIONS } from './local-embedder-worker';

/** The ONNX Runtime binding transformers.js uses under Node. */
export const EMBED_ORT_PACKAGE = 'onnxruntime-node';

/**
 * One entry of `onnxruntime-node`'s `listSupportedBackends()`.
 *
 * ⚠ `bundled` means "ships in the default npm tarball", NOT "usable here" — see
 * the module header. Never branch on it to decide whether a device is reachable.
 */
export type EmbedBackend = { name: string; bundled: boolean };

/** The (device, dtype) pair the embedders actually run on, and the basis for it. */
export type EmbedExecutionTarget = {
  device: string;
  /** `null` = no dtype is pinned, so the transformers.js default applies. */
  dtype: string | null;
  /** What evidence supports `device` — read this before quoting the field. */
  why: string;
};

/** Whether the runtime probe has run, and how it went. */
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
   * Execution-provider shared libraries actually present next to the binding
   * (bare filenames). THIS is the availability signal — it is what differed
   * between a default install and a `ONNXRUNTIME_NODE_INSTALL_CUDA=v12` one.
   * `null` = not measured.
   */
  providerLibraries: string[] | null;
  /**
   * A DISCRETE GPU provider library (cuda / tensorrt / rocm / migraphx / dml)
   * is present on disk. `null` = not measured.
   *
   * Two deliberate limits, both of which have already caught a wrong reading:
   *
   *  - Presence is NOT a promise that a session would construct. The library can
   *    be present and still fail to load (missing CUDA/cuDNN runtime, ABI
   *    mismatch) — that failure is what `@papercusp/rerank`'s `demotionCause`
   *    reports, and on this host it says exactly that.
   *  - It does NOT cover **webgpu**, which is compiled INTO `libonnxruntime.so.1`
   *    rather than shipping as its own `.so`, so no disk scan can see it. A host
   *    can therefore read `gpuProviderAvailable: false` while `webgpu` is
   *    genuinely usable. Consult `defaultBundledBackends` for that one — it is
   *    the case where the packaging metadata is the better signal, which is why
   *    both fields are reported side by side instead of one being "the answer".
   */
  gpuProviderAvailable: boolean | null;
  /**
   * What `listSupportedBackends()` says ships by DEFAULT. Diagnostic only —
   * deliberately NOT named `bundled`/`available` so it cannot be misread as
   * capability. See the module header for the measurement that proves the gap.
   */
  defaultBundledBackends: EmbedBackend[] | null;
  probe: EmbedExecutionProbe;
  probeError: string | null;
  /**
   * The thread caps that DO get set. These govern CPU forward-pass latency
   * directly, so they belong beside the device that makes them load-bearing.
   */
  sessionOptions: { intraOpNumThreads: number; interOpNumThreads: number };
};

/**
 * The device/dtype the embed path requests. PINNED by
 * `execution-target.test.ts` against the embedder sources, so it cannot drift
 * into a lie the way a hand-maintained `exists:` boolean does.
 */
export const EMBED_REQUESTED_EXECUTION: { device: string | null; dtype: string | null } = {
  device: null,
  dtype: null,
};

/** Provider-library basenames that indicate a non-CPU execution path. */
const GPU_PROVIDER_LIB_PATTERN = /^libonnxruntime_providers_(cuda|tensorrt|rocm|migraphx|dml)\./i;

type ExecutionTargetState = {
  probe: EmbedExecutionProbe;
  defaultBundledBackends: EmbedBackend[] | null;
  providerLibraries: string[] | null;
  probeError: string | null;
  inFlight: Promise<void> | null;
};

const STATE_KEY = '@papercusp/memory.embed-execution-target';

const state = pinModuleState<ExecutionTargetState>(STATE_KEY, () => ({
  probe: 'pending',
  defaultBundledBackends: null,
  providerLibraries: null,
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
 * with the reason attached — never to a plausible-looking empty list.
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
 * List the execution-provider shared libraries sitting next to the resolved
 * binding. Returns null when the location cannot be resolved — a failure to
 * MEASURE, never an assertion that none exist.
 */
function scanProviderLibraries(): string[] | null {
  try {
    const req = createRequire(import.meta.url);
    // onnxruntime-node's entry resolves inside dist/; the binaries live at
    // <pkg>/bin/napi-v6/<platform>/<arch>/. Walk up from the entry to the
    // package root rather than hardcoding the dist layout.
    const entry = req.resolve(EMBED_ORT_PACKAGE);
    const pkgRoot = dirname(dirname(entry));
    const binRoot = join(pkgRoot, 'bin', 'napi-v6', process.platform, process.arch);
    return readdirSync(binRoot)
      .filter((f) => f.startsWith('libonnxruntime_providers_'))
      .sort();
  } catch {
    return null;
  }
}

/**
 * Resolve the runtime execution-provider facts once per process and cache them.
 * Idempotent, concurrency-safe, and never throws: a failure is recorded as
 * `probe:'failed'` with `probeError` so `/healthz` can say the measurement did
 * not happen.
 */
export function ensureEmbedBackendsProbed(): Promise<void> {
  if (state.probe !== 'pending') return Promise.resolve();
  if (state.inFlight) return state.inFlight;
  const run = (async () => {
    try {
      const mod = await dynamicImport<OrtModule>(EMBED_ORT_PACKAGE);
      const holder = mod?.listSupportedBackends ? mod : mod?.default;
      const list = holder?.listSupportedBackends;
      if (typeof list !== 'function') {
        throw new Error(`${EMBED_ORT_PACKAGE} exposes no listSupportedBackends()`);
      }
      state.defaultBundledBackends = normalizeBackends(list.call(holder));
      state.providerLibraries = scanProviderLibraries();
      state.probeError = null;
      state.probe = 'ok';
    } catch (e) {
      state.defaultBundledBackends = null;
      state.providerLibraries = null;
      state.probeError = e instanceof Error ? e.message : String(e);
      state.probe = 'failed';
    }
  })().finally(() => {
    state.inFlight = null;
  });
  state.inFlight = run;
  return run;
}

/** Provider libraries present on disk; `null` while unmeasured. */
export function embedProviderLibraries(): string[] | null {
  return state.probe === 'ok' && state.providerLibraries ? [...state.providerLibraries] : null;
}

/**
 * Whether a GPU-capable provider library is present on disk. `null` = not
 * measured. Presence is necessary but NOT sufficient for a working GPU session.
 */
export function embedGpuProviderAvailable(): boolean | null {
  const libs = embedProviderLibraries();
  if (libs === null) return null;
  return libs.some((f) => GPU_PROVIDER_LIB_PATTERN.test(f));
}

/**
 * The (device, dtype) pair the embedders are running on.
 *
 * `device` is `cpu` in every branch — because nothing requests otherwise, and
 * transformers.js defaults to CPU under Node — so what varies is `why`, i.e.
 * how well-evidenced that answer is. Read it before quoting the pair.
 */
export function embedExecutionTarget(): EmbedExecutionTarget {
  const gpu = embedGpuProviderAvailable();
  const base = 'the embed path requests no device, so transformers.js applies its CPU default';
  const dtype = EMBED_REQUESTED_EXECUTION.dtype;
  if (gpu === null) {
    return {
      device: 'cpu',
      dtype,
      why: `${base}; the provider-library probe has not resolved, so whether a GPU provider is installed is UNKNOWN`,
    };
  }
  if (!gpu) {
    return {
      device: 'cpu',
      dtype,
      why:
        `${base}; no GPU execution-provider library is installed beside the onnxruntime binding, so no other ` +
        'device is reachable without reinstalling onnxruntime-node with a provider (e.g. ONNXRUNTIME_NODE_INSTALL_CUDA)',
    };
  }
  return {
    device: 'cpu',
    dtype,
    // Deliberately hedged: a provider library on disk can still fail to LOAD
    // (missing CUDA/cuDNN runtime, ABI mismatch). Only constructing a session
    // proves usability — see rerank's demoted/demotionCause pair.
    why:
      `${base}. A GPU execution-provider library IS installed (${embedProviderLibraries()?.join(', ')}), so the CPU ` +
      'target is by omission rather than a hardware limit — but an installed provider can still fail to load, and ' +
      'only constructing a session proves otherwise',
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
    providerLibraries: embedProviderLibraries(),
    gpuProviderAvailable: embedGpuProviderAvailable(),
    defaultBundledBackends: state.defaultBundledBackends
      ? state.defaultBundledBackends.map((b) => ({ ...b }))
      : null,
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
  state.defaultBundledBackends = null;
  state.providerLibraries = null;
  state.probeError = null;
  state.inFlight = null;
}
