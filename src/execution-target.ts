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
 * What this module reports instead:
 *
 *   1. **What the embed path requests.** Until 2026-09-24 that was NO device —
 *      the root cause above. Since then `embed-device.ts` chooses one
 *      (`PAPERCUSP_EMBED_DEVICE` auto|gpu|cpu, plan memory-reduction-2026-09-24
 *      D-003) and every embedder constructs through it; `execution-target.test.ts`
 *      PINS that against the embedder sources, the worker script included.
 *   2. **What actually constructed.** `active` / `pipelines` / `demotion` come
 *      from real session construction (a GPU that will not build falls back to
 *      the CPU with its cause recorded). Before any pipeline exists, `active`
 *      carries `verified: false` — the device the next one will TRY.
 *   3. **Which provider libraries are present on disk.** The availability
 *      signal (`providerLibraries` / `gpuProviderAvailable`).
 *   4. **What ships by default.** `defaultBundledBackends`, kept because it
 *      explains WHY a provider is absent (nobody passed the install flag), but
 *      explicitly named so it can never again be read as availability.
 *
 * An unresolved probe reports `null` everywhere — never a convenient `false`.
 * UNKNOWN is in-band and asserted by test.
 *
 * This module reports; `embed-device.ts` chooses.
 */

import { pinModuleState } from '@papercusp/module-singleton';
import { dynamicImport } from './dynamic-import';
import {
  EMBED_ORT_PACKAGE,
  currentEmbedDeviceDecision,
  embedDeviceDemotion,
  embedPipelineDevices,
  nvidiaDriverPresent,
  scanEmbedProviderLibraries,
  type EmbedDevice,
  type EmbedDeviceDemotion,
  type EmbedDeviceSelection,
} from './embed-device';
import { ORT_SESSION_OPTIONS } from './local-embedder-worker';

export { EMBED_ORT_PACKAGE };

/**
 * One entry of `onnxruntime-node`'s `listSupportedBackends()`.
 *
 * ⚠ `bundled` means "ships in the default npm tarball", NOT "usable here" — see
 * the module header. Never branch on it to decide whether a device is reachable.
 */
export type EmbedBackend = { name: string; bundled: boolean };

/** The (device, dtype) pair the embedders actually run on, and the basis for it. */
export type EmbedExecutionTarget = {
  /** `cpu` | `cuda`, or `mixed` when models constructed on different devices. */
  device: string;
  /** `null` = no dtype is pinned, so the transformers.js default applies. */
  dtype: string | null;
  /**
   * `true` once a pipeline has actually CONSTRUCTED on `device` in this
   * process. `false` = no pipeline exists yet, so `device` is only what the next
   * one will try (a GPU choice can still fall back to the CPU at construction).
   */
  verified: boolean;
  /** What evidence supports `device` — read this before quoting the field. */
  why: string;
};

/** What the embed path asks for, and why. */
export type EmbedRequestedExecution = {
  device: EmbedDevice;
  /** Always `null`: only the device may move (the vector-space invariant in embed-device.ts). */
  dtype: null;
  preference: EmbedDeviceSelection['preference'];
  source: EmbedDeviceSelection['source'];
  invalidValue: string | null;
  why: string;
};

/** Whether the runtime probe has run, and how it went. */
export type EmbedExecutionProbe = 'pending' | 'ok' | 'failed';

export type EmbedExecutionHealth = {
  /** What the embed path ASKS FOR (PAPERCUSP_EMBED_DEVICE + host checks + any demotion). */
  requested: EmbedRequestedExecution;
  /** What it actually runs on — see `verified`. */
  active: EmbedExecutionTarget;
  /** model id → the device its pipeline constructed on, in this process. */
  pipelines: Record<string, EmbedDevice>;
  /** The GPU→CPU fallback, if one happened in this process; `null` = none. */
  demotion: EmbedDeviceDemotion | null;
  /** NVIDIA kernel driver loaded? `null` = no cheap probe on this platform. */
  nvidiaDriverPresent: boolean | null;
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
 * The device/dtype the embed path requests right now. Every embedder constructs
 * through `constructEmbedPipeline` (or, in the worker, the device handed over
 * from `currentEmbedDeviceDecision`) — PINNED by `execution-target.test.ts`
 * against the embedder sources, so this cannot drift into a lie.
 */
export function embedRequestedExecution(): EmbedRequestedExecution {
  const d = currentEmbedDeviceDecision();
  return {
    device: d.device,
    dtype: null,
    preference: d.selection.preference,
    source: d.selection.source,
    invalidValue: d.selection.invalidValue,
    why: d.why,
  };
}

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
      state.providerLibraries = scanEmbedProviderLibraries();
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
 * Answered from real session construction when any pipeline exists in this
 * process (`verified: true`); before that it is the device the next pipeline
 * will TRY (`verified: false`). Read `why` before quoting the pair.
 */
export function embedExecutionTarget(): EmbedExecutionTarget {
  const requested = embedRequestedExecution();
  const pipelines = embedPipelineDevices();
  const models = Object.keys(pipelines).sort();
  const dtype = requested.dtype;
  if (models.length === 0) {
    return {
      device: requested.device,
      dtype,
      verified: false,
      why:
        `${requested.why}. No embed pipeline has been constructed in this process yet, so this is the device ` +
        'the next one will try — unverified until it constructs',
    };
  }
  const devices = [...new Set(models.map((m) => pipelines[m]))];
  const byModel = models.map((m) => `${m}=${pipelines[m]}`).join(', ');
  const demotion = embedDeviceDemotion();
  const device = devices.length === 1 ? devices[0] : 'mixed';
  const why = demotion
    ? `constructed: ${byModel}. DEMOTED from ${demotion.from} for ${demotion.model} (${demotion.stage}): ${demotion.cause}`
    : `constructed: ${byModel}. ${requested.why}`;
  return { device, dtype, verified: true, why };
}

/**
 * The full embed-execution report for `/healthz`.
 *
 * Sync, so the sidecar's sync handler can call it. It reports cached probe
 * state rather than blocking; `probe` says whether that state is a measurement.
 */
export function embedExecutionHealth(): EmbedExecutionHealth {
  return {
    requested: embedRequestedExecution(),
    active: embedExecutionTarget(),
    pipelines: embedPipelineDevices(),
    demotion: embedDeviceDemotion(),
    nvidiaDriverPresent: nvidiaDriverPresent(),
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
