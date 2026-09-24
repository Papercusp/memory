/**
 * embed-device.ts — WHICH device the embedders run on, and the fallback when
 * that device will not work (plan memory-reduction-2026-09-24, P-008 / D-003).
 *
 * ## The contract
 *
 * `PAPERCUSP_EMBED_DEVICE` = `auto` (default) | `gpu`/`cuda` | `cpu`.
 *
 *  - `auto` picks CUDA only when BOTH cheap preconditions hold: a CUDA/TensorRT
 *    execution-provider library sits beside the onnxruntime binding (the
 *    installer put it there), and — on Linux — the NVIDIA kernel driver is
 *    loaded (`/proc/driver/nvidia/version`). Either one missing PROVES the GPU
 *    is unusable, so a CPU-only host never pays for a failed GPU attempt.
 *  - Both preconditions holding does NOT prove the GPU works (a missing cuDNN,
 *    an ABI mismatch, a full GPU). So every GPU choice — auto or forced — is
 *    VERIFIED by constructing the session: a CUDA session that will not
 *    construct is retried on the CPU, and the cause is recorded
 *    (`embedDeviceDemotion()`), mirroring `@papercusp/rerank`'s
 *    `demotedTarget`. A demotion is sticky for the process: the host's GPU does
 *    not start working again without a restart, and re-trying would re-read
 *    a multi-GB model file on every new pipeline just to fail again.
 *
 * ## Why dtype is NOT touched (the vector-space invariant)
 *
 * `@papercusp/rerank` moves (device, dtype) together because its fp16 GPU
 * weights are a different format. Embeddings must NOT do that: stored vectors
 * were computed with the default (fp32) weights, and a different weight format
 * would drift every new vector away from the stored corpus. Only the device
 * moves. Measured 2026-09-24 on the RTX 3090 with Xenova/bge-small-en-v1.5:
 * CPU vs CUDA cosine 0.99999988 / 0.99999950 — the same vector space, so no
 * re-embed is needed. `constructEmbedPipeline` never passes `dtype`, and
 * `embed-device.test.ts` asserts it.
 *
 * ## Where this runs
 *
 * The hot path is the worker thread (`local-embedder-worker.script.mjs`), which
 * is plain JS copied as ONE file into bundles, so it cannot import this module.
 * The main thread resolves the device here and hands it to the worker as
 * `workerData.device`; the worker reports back what actually constructed
 * (`{ kind: 'device', … }`), which lands in `recordEmbedPipelineDevice`.
 * The inline (main-thread) fallback paths call `constructEmbedPipeline`
 * directly. Either way, `/healthz` reads one state.
 */

import { existsSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

import { pinModuleState } from '@papercusp/module-singleton';

/** Devices the embed path can run on. */
export type EmbedDevice = 'cpu' | 'cuda';

/** What the host asked for. `auto` = decide from the host. */
export type EmbedDevicePreference = 'auto' | 'cuda' | 'cpu';

/** The environment variable that forces a device (D-003's Settings override). */
export const EMBED_DEVICE_ENV = 'PAPERCUSP_EMBED_DEVICE';

/** The ONNX Runtime binding transformers.js uses under Node. */
export const EMBED_ORT_PACKAGE = 'onnxruntime-node';

/** Provider libraries that make CUDA reachable (TensorRT needs the CUDA provider too). */
const CUDA_PROVIDER_LIB_PATTERN = /^libonnxruntime_providers_(cuda|tensorrt)\./i;

/** Present iff the NVIDIA kernel driver is loaded (Linux). */
export const NVIDIA_DRIVER_PROBE_PATH = '/proc/driver/nvidia/version';

export type EmbedDeviceSelection = {
  preference: EmbedDevicePreference;
  /** `env` = set explicitly; `default` = unset; `env-invalid` = set to something unrecognised (treated as auto). */
  source: 'env' | 'default' | 'env-invalid';
  /** The raw value when `source` is `env-invalid`, so a typo is visible rather than silently ignored. */
  invalidValue: string | null;
};

export type EmbedDeviceDecision = {
  /** The device the next pipeline will be constructed on. */
  device: EmbedDevice;
  /** Why — read this before quoting `device`. */
  why: string;
};

export type EmbedDeviceDemotion = {
  from: EmbedDevice;
  to: EmbedDevice;
  /** The model whose session failed on `from`. */
  model: string;
  /** `construct` = the session never built; `inference` = it built but a forward pass failed on the GPU. */
  stage: 'construct' | 'inference';
  cause: string;
  at: string;
};

type EmbedDeviceState = {
  /** `undefined` = not scanned yet; `null` = the scan could not resolve the binding. */
  providerLibraries: string[] | null | undefined;
  demotion: EmbedDeviceDemotion | null;
  /** model id → the device its pipeline actually constructed on. */
  pipelines: Record<string, EmbedDevice>;
};

const state = pinModuleState<EmbedDeviceState>('@papercusp/memory.embed-device', () => ({
  providerLibraries: undefined,
  demotion: null,
  pipelines: {},
}));

/** Parse the device preference. Never throws: a typo must not take embeds down. */
export function resolveEmbedDevicePreference(
  env: Record<string, string | undefined> = typeof process === 'undefined' ? {} : process.env,
): EmbedDeviceSelection {
  const raw = (env[EMBED_DEVICE_ENV] ?? '').trim();
  const v = raw.toLowerCase();
  if (v === '') return { preference: 'auto', source: 'default', invalidValue: null };
  if (v === 'auto') return { preference: 'auto', source: 'env', invalidValue: null };
  if (v === 'cpu') return { preference: 'cpu', source: 'env', invalidValue: null };
  if (v === 'cuda' || v === 'gpu') return { preference: 'cuda', source: 'env', invalidValue: null };
  return { preference: 'auto', source: 'env-invalid', invalidValue: raw };
}

/**
 * List the execution-provider shared libraries next to the resolved binding.
 * `null` = the location could not be resolved — a failure to MEASURE, never an
 * assertion that none exist.
 */
export function scanEmbedProviderLibraries(): string[] | null {
  try {
    const req = createRequire(import.meta.url);
    // The entry resolves inside dist/; the binaries live at
    // <pkg>/bin/napi-v6/<platform>/<arch>/.
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

/** Provider libraries, scanned once per process. */
export function embedDeviceProviderLibraries(): string[] | null {
  if (state.providerLibraries === undefined) state.providerLibraries = scanEmbedProviderLibraries();
  return state.providerLibraries ? [...state.providerLibraries] : null;
}

/**
 * Is the NVIDIA kernel driver loaded? `true`/`false` on Linux; `null` elsewhere
 * (no equally cheap probe — the construct-time fallback covers those hosts).
 */
export function nvidiaDriverPresent(
  platform: string = process.platform,
  exists: (path: string) => boolean = existsSync,
): boolean | null {
  if (platform !== 'linux') return null;
  try {
    return exists(NVIDIA_DRIVER_PROBE_PATH);
  } catch {
    return null;
  }
}

/**
 * Decide the device from explicit inputs. Pure, so every branch is testable
 * without a GPU.
 */
export function decideEmbedDevice(input: {
  selection: EmbedDeviceSelection;
  providerLibraries: string[] | null;
  nvidiaDriver: boolean | null;
  demotion: EmbedDeviceDemotion | null;
}): EmbedDeviceDecision {
  const { selection, providerLibraries, nvidiaDriver, demotion } = input;
  const invalid =
    selection.source === 'env-invalid'
      ? ` (${EMBED_DEVICE_ENV}=${JSON.stringify(selection.invalidValue)} is not auto|gpu|cuda|cpu, so auto applies)`
      : '';

  if (selection.preference === 'cpu') {
    return { device: 'cpu', why: `${EMBED_DEVICE_ENV}=cpu forces the CPU` };
  }
  if (demotion) {
    return {
      device: 'cpu',
      why:
        `DEMOTED from ${demotion.from}: a ${demotion.stage} on ${demotion.from} failed for ${demotion.model} ` +
        `in this process, so embeds run on the CPU (correct, but slower). Cause: ${demotion.cause}`,
    };
  }
  if (selection.preference === 'cuda') {
    return {
      device: 'cuda',
      why: `${EMBED_DEVICE_ENV} forces the GPU; the session is still verified at construction and falls back to the CPU if it will not build`,
    };
  }

  // auto
  const cudaLibs = providerLibraries?.filter((f) => CUDA_PROVIDER_LIB_PATTERN.test(f)) ?? [];
  if (providerLibraries === null) {
    return {
      device: 'cpu',
      why: `auto${invalid}: the onnxruntime provider directory could not be read, so no GPU provider is known to be installed`,
    };
  }
  if (cudaLibs.length === 0) {
    return {
      device: 'cpu',
      why:
        `auto${invalid}: no CUDA execution-provider library is installed beside the onnxruntime binding ` +
        '(the installer adds it where the platform supports it — scripts/install-onnxruntime-node.mjs)',
    };
  }
  if (nvidiaDriver === false) {
    return {
      device: 'cpu',
      why: `auto${invalid}: the CUDA provider is installed but no NVIDIA driver is loaded (${NVIDIA_DRIVER_PROBE_PATH} is absent)`,
    };
  }
  return {
    device: 'cuda',
    why:
      `auto${invalid}: the CUDA provider is installed (${cudaLibs.join(', ')})` +
      (nvidiaDriver ? ' and the NVIDIA driver is loaded' : '') +
      '; the session is verified at construction and falls back to the CPU if it will not build',
  };
}

/** The live decision for this process (env + host + any demotion so far). */
export function currentEmbedDeviceDecision(
  env: Record<string, string | undefined> = typeof process === 'undefined' ? {} : process.env,
): EmbedDeviceDecision & { selection: EmbedDeviceSelection } {
  const selection = resolveEmbedDevicePreference(env);
  const decision = decideEmbedDevice({
    selection,
    providerLibraries: embedDeviceProviderLibraries(),
    nvidiaDriver: nvidiaDriverPresent(),
    demotion: state.demotion,
  });
  return { ...decision, selection };
}

/** Record which device a model's pipeline actually constructed on. */
export function recordEmbedPipelineDevice(model: string, device: EmbedDevice): void {
  state.pipelines[model] = device;
}

/**
 * Record a GPU failure. The first one wins (it names the original cause); later
 * failures in the same process are consequences, not new information.
 */
export function recordEmbedDeviceDemotion(demotion: Omit<EmbedDeviceDemotion, 'at'> & { at?: string }): void {
  if (state.demotion) return;
  state.demotion = { ...demotion, at: demotion.at ?? new Date().toISOString() };
  console.warn(
    `[embed-device] ${demotion.model}: ${demotion.stage} on ${demotion.from} failed — embeds fall back to ` +
      `${demotion.to} for the rest of this process. Cause: ${demotion.cause}`,
  );
}

export function embedDeviceDemotion(): EmbedDeviceDemotion | null {
  return state.demotion ? { ...state.demotion } : null;
}

/** model id → device its pipeline constructed on (worker and inline paths alike). */
export function embedPipelineDevices(): Record<string, EmbedDevice> {
  return { ...state.pipelines };
}

/** A worker's `{ kind: 'device' }` report. Unknown shapes are ignored, never trusted. */
export function applyWorkerDeviceReport(msg: unknown): boolean {
  const m = msg as {
    kind?: unknown;
    model?: unknown;
    device?: unknown;
    demotion?: { from?: unknown; stage?: unknown; cause?: unknown } | null;
  };
  if (!m || m.kind !== 'device' || typeof m.model !== 'string') return false;
  if (m.device !== 'cpu' && m.device !== 'cuda') return false;
  recordEmbedPipelineDevice(m.model, m.device);
  const d = m.demotion;
  if (d && (d.from === 'cuda' || d.from === 'cpu') && (d.stage === 'construct' || d.stage === 'inference')) {
    recordEmbedDeviceDemotion({
      from: d.from,
      to: 'cpu',
      model: m.model,
      stage: d.stage,
      cause: typeof d.cause === 'string' ? d.cause : String(d.cause),
    });
  }
  return true;
}

type PipelineFactory<P> = {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<P>;
};

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Construct a feature-extraction pipeline on the decided device, verifying a
 * GPU choice by construction and falling back to the CPU when it will not build.
 *
 * Never passes `dtype` — see the module header's vector-space invariant.
 */
export async function constructEmbedPipeline<P>(
  transformers: PipelineFactory<P>,
  model: string,
  sessionOptions: Readonly<Record<string, unknown>>,
): Promise<P> {
  const { device } = currentEmbedDeviceDecision();
  const build = (d: EmbedDevice) =>
    transformers.pipeline('feature-extraction', model, { session_options: sessionOptions, device: d });
  if (device === 'cpu') {
    const p = await build('cpu');
    recordEmbedPipelineDevice(model, 'cpu');
    return p;
  }
  try {
    const p = await build(device);
    recordEmbedPipelineDevice(model, device);
    return p;
  } catch (err) {
    recordEmbedDeviceDemotion({ from: device, to: 'cpu', model, stage: 'construct', cause: errMessage(err) });
    const p = await build('cpu');
    recordEmbedPipelineDevice(model, 'cpu');
    return p;
  }
}

/** Test-only: forget the scan, the demotion and the per-model record. */
export function _resetEmbedDeviceState(): void {
  state.providerLibraries = undefined;
  state.demotion = null;
  state.pipelines = {};
}
