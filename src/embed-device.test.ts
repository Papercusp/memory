/**
 * embed-device.test.ts — the device choice for the embedders (plan
 * memory-reduction-2026-09-24 P-008 / D-003: Auto + Settings override).
 *
 * Three properties are guarded here, each of which fails SILENTLY in production
 * if it regresses — the engine keeps answering, just slower or subtly wrong:
 *
 *   1. auto never picks a GPU a host provably cannot use (no CUDA provider on
 *      disk, or no NVIDIA driver), so a CPU-only install pays nothing;
 *   2. a GPU that will not work falls back to the CPU with the CAUSE recorded,
 *      and the fallback is sticky (no re-reading a multi-GB model per pipeline
 *      just to fail again) — in the main-thread path AND in the worker script,
 *      which is the hot path and cannot import this module;
 *   3. `dtype` is never passed: only the device may move, or new vectors drift
 *      away from the stored corpus (CPU vs CUDA cosine 0.9999999, measured).
 */

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  EMBED_DEVICE_ENV,
  NVIDIA_DRIVER_PROBE_PATH,
  _resetEmbedDeviceState,
  applyWorkerDeviceReport,
  constructEmbedPipeline,
  currentEmbedDeviceDecision,
  decideEmbedDevice,
  embedDeviceDemotion,
  embedDeviceSetting,
  embedPipelineDevices,
  nvidiaDriverPresent,
  parseEmbedDevicePreference,
  resolveEmbedDevicePreference,
  setEmbedDeviceSetting,
  type EmbedDeviceSelection,
} from './embed-device';

const SRC = dirname(fileURLToPath(import.meta.url));
const CUDA_LIBS = ['libonnxruntime_providers_cuda.so', 'libonnxruntime_providers_shared.so'];
const AUTO: EmbedDeviceSelection = { preference: 'auto', source: 'default', invalidValue: null, setting: null };

describe('resolveEmbedDevicePreference — PAPERCUSP_EMBED_DEVICE', () => {
  it('defaults to auto when unset', () => {
    expect(resolveEmbedDevicePreference({})).toEqual(AUTO);
  });

  it('accepts auto / cpu / cuda / gpu, case- and space-insensitively', () => {
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: ' Auto ' }).preference).toBe('auto');
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'CPU' }).preference).toBe('cpu');
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'cuda' }).preference).toBe('cuda');
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'GPU' }).preference).toBe('cuda');
  });

  it('treats a typo as auto but keeps the raw value visible', () => {
    // A typo must never take embeds down, and must never be silently ignored.
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'cdua' })).toEqual({
      preference: 'auto',
      source: 'env-invalid',
      invalidValue: 'cdua',
      setting: null,
    });
  });
});

describe('the Settings choice vs PAPERCUSP_EMBED_DEVICE (plan D-008)', () => {
  beforeEach(() => _resetEmbedDeviceState());
  afterEach(() => _resetEmbedDeviceState());

  it('the setting applies when the env is unset', () => {
    expect(resolveEmbedDevicePreference({}, 'cpu')).toEqual({
      preference: 'cpu',
      source: 'setting',
      invalidValue: null,
      setting: 'cpu',
    });
  });

  it('env=auto defers to the setting — an explicit auto is not a host override', () => {
    const s = resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'auto' }, 'cuda');
    expect(s.preference).toBe('cuda');
    expect(s.source).toBe('setting');
  });

  it('a CONCRETE env device outranks the setting, which stays visible', () => {
    // How one host (a headless verify instance) is pinned without touching the shared row.
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'cpu' }, 'cuda')).toEqual({
      preference: 'cpu',
      source: 'env',
      invalidValue: null,
      setting: 'cuda',
    });
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'gpu' }, 'cpu').preference).toBe('cuda');
  });

  it('an unrecognised env value is ignored in favour of the setting, but still reported', () => {
    expect(resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'cdua' }, 'cpu')).toEqual({
      preference: 'cpu',
      source: 'setting',
      invalidValue: 'cdua',
      setting: 'cpu',
    });
  });

  it('parseEmbedDevicePreference: gpu is the user word for cuda; junk is null, never a throw', () => {
    expect(parseEmbedDevicePreference('GPU')).toBe('cuda');
    expect(parseEmbedDevicePreference(' cpu ')).toBe('cpu');
    expect(parseEmbedDevicePreference('auto')).toBe('auto');
    expect(parseEmbedDevicePreference('tpu')).toBeNull();
    expect(parseEmbedDevicePreference(undefined)).toBeNull();
    expect(parseEmbedDevicePreference(3)).toBeNull();
  });

  it('setEmbedDeviceSetting reports a change only when the EFFECTIVE preference moves', () => {
    expect(setEmbedDeviceSetting('cpu', {}).changed).toBe(true);
    expect(embedDeviceSetting()).toBe('cpu');
    expect(setEmbedDeviceSetting('cpu', {}).changed).toBe(false);
    // Env pins cpu: moving the setting cpu → gpu changes nothing effective.
    expect(setEmbedDeviceSetting('cuda', { [EMBED_DEVICE_ENV]: 'cpu' }).changed).toBe(false);
    expect(embedDeviceSetting()).toBe('cuda');
    expect(setEmbedDeviceSetting(null, {}).changed).toBe(true); // cuda → auto
  });

  it('a change clears the demotion and the per-model record (the user asked for a retry)', () => {
    applyWorkerDeviceReport({
      kind: 'device',
      model: 'm',
      device: 'cpu',
      demotion: { from: 'cuda', stage: 'construct', cause: 'no cudnn' },
    });
    expect(embedDeviceDemotion()).not.toBeNull();
    setEmbedDeviceSetting('cpu', {});
    expect(embedDeviceDemotion()).toBeNull();
    expect(embedPipelineDevices()).toEqual({});
  });

  it('a no-op set keeps the demotion (nothing about the host changed)', () => {
    applyWorkerDeviceReport({
      kind: 'device',
      model: 'm',
      device: 'cpu',
      demotion: { from: 'cuda', stage: 'construct', cause: 'no cudnn' },
    });
    setEmbedDeviceSetting('auto', {}); // auto → auto
    expect(embedDeviceDemotion()).not.toBeNull();
  });

  it('the why names the layer that decided', () => {
    const fromSetting = decideEmbedDevice({
      selection: resolveEmbedDevicePreference({}, 'cpu'),
      providerLibraries: CUDA_LIBS,
      nvidiaDriver: true,
      demotion: null,
    });
    expect(fromSetting).toEqual({ device: 'cpu', why: 'the Settings choice (CPU) forces the CPU' });
    const fromEnv = decideEmbedDevice({
      selection: resolveEmbedDevicePreference({ [EMBED_DEVICE_ENV]: 'cpu' }, 'cuda'),
      providerLibraries: CUDA_LIBS,
      nvidiaDriver: true,
      demotion: null,
    });
    expect(fromEnv.why).toMatch(/PAPERCUSP_EMBED_DEVICE=cpu \(a host override, which outranks the Settings choice\)/);
  });

  it('currentEmbedDeviceDecision reads the held setting', () => {
    setEmbedDeviceSetting('cpu', {});
    const d = currentEmbedDeviceDecision({});
    expect(d.device).toBe('cpu');
    expect(d.selection.source).toBe('setting');
  });
});

describe('decideEmbedDevice — auto only picks a GPU the host could use', () => {
  const decide = (over: Partial<Parameters<typeof decideEmbedDevice>[0]>) =>
    decideEmbedDevice({ selection: AUTO, providerLibraries: CUDA_LIBS, nvidiaDriver: true, demotion: null, ...over });

  it('auto + CUDA provider installed + NVIDIA driver loaded ⇒ cuda', () => {
    expect(decide({}).device).toBe('cuda');
  });

  it('auto + no CUDA provider on disk ⇒ cpu, and says why', () => {
    const d = decide({ providerLibraries: ['libonnxruntime_providers_shared.so'] });
    expect(d.device).toBe('cpu');
    expect(d.why).toMatch(/no CUDA execution-provider library/);
  });

  it('auto + provider directory unreadable ⇒ cpu (unknown is not a yes)', () => {
    expect(decide({ providerLibraries: null }).device).toBe('cpu');
  });

  it('auto + provider installed but no NVIDIA driver ⇒ cpu', () => {
    const d = decide({ nvidiaDriver: false });
    expect(d.device).toBe('cpu');
    expect(d.why).toContain(NVIDIA_DRIVER_PROBE_PATH);
  });

  it('auto + provider installed + driver probe unavailable (non-Linux) ⇒ cuda, verified at construction', () => {
    expect(decide({ nvidiaDriver: null }).device).toBe('cuda');
  });

  it('TensorRT alone also counts as a CUDA provider', () => {
    expect(decide({ providerLibraries: ['libonnxruntime_providers_tensorrt.so'] }).device).toBe('cuda');
  });

  it('forced cpu wins over everything', () => {
    const d = decide({ selection: { preference: 'cpu', source: 'env', invalidValue: null, setting: null } });
    expect(d.device).toBe('cpu');
  });

  it('forced gpu is honoured even without a provider on disk (construction is the proof)', () => {
    const d = decide({
      selection: { preference: 'cuda', source: 'env', invalidValue: null, setting: null },
      providerLibraries: [],
      nvidiaDriver: false,
    });
    expect(d.device).toBe('cuda');
  });

  it('a recorded demotion forces cpu for the rest of the process, and names the cause', () => {
    const d = decide({
      selection: { preference: 'cuda', source: 'env', invalidValue: null, setting: null },
      demotion: { from: 'cuda', to: 'cpu', model: 'm', stage: 'construct', cause: 'libcudnn.so.9 missing', at: 'x' },
    });
    expect(d.device).toBe('cpu');
    expect(d.why).toContain('DEMOTED');
    expect(d.why).toContain('libcudnn.so.9 missing');
  });
});

describe('nvidiaDriverPresent', () => {
  it('probes the driver file on linux and reports null elsewhere', () => {
    expect(nvidiaDriverPresent('linux', (p) => p === NVIDIA_DRIVER_PROBE_PATH)).toBe(true);
    expect(nvidiaDriverPresent('linux', () => false)).toBe(false);
    expect(nvidiaDriverPresent('win32', () => true)).toBeNull();
    expect(nvidiaDriverPresent('darwin', () => true)).toBeNull();
  });
});

describe('constructEmbedPipeline — verify the GPU by construction, fall back to CPU', () => {
  const saved = process.env[EMBED_DEVICE_ENV];
  beforeEach(() => {
    _resetEmbedDeviceState();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    if (saved === undefined) delete process.env[EMBED_DEVICE_ENV];
    else process.env[EMBED_DEVICE_ENV] = saved;
    _resetEmbedDeviceState();
    vi.restoreAllMocks();
  });

  function fakeTransformers(failOn: Set<string>) {
    const calls: Array<{ model: string; opts: Record<string, unknown> }> = [];
    return {
      calls,
      pipeline: async (_task: string, model: string, opts: Record<string, unknown> = {}) => {
        calls.push({ model, opts });
        if (failOn.has(String(opts.device))) throw new Error(`cannot load ${String(opts.device)} provider`);
        return { model, device: opts.device };
      },
    };
  }

  it('builds on cuda when forced and it constructs, and records it', async () => {
    process.env[EMBED_DEVICE_ENV] = 'gpu';
    const t = fakeTransformers(new Set());
    const p = await constructEmbedPipeline(t, 'model-a', { intraOpNumThreads: 2 });
    expect(p).toEqual({ model: 'model-a', device: 'cuda' });
    expect(embedPipelineDevices()).toEqual({ 'model-a': 'cuda' });
    expect(embedDeviceDemotion()).toBeNull();
  });

  it('falls back to cpu when the cuda session will not construct, records the cause, and stays on cpu', async () => {
    process.env[EMBED_DEVICE_ENV] = 'gpu';
    const t = fakeTransformers(new Set(['cuda']));
    const p = await constructEmbedPipeline(t, 'model-a', {});
    expect(p).toEqual({ model: 'model-a', device: 'cpu' });
    expect(embedPipelineDevices()).toEqual({ 'model-a': 'cpu' });
    expect(embedDeviceDemotion()).toMatchObject({
      from: 'cuda',
      to: 'cpu',
      model: 'model-a',
      stage: 'construct',
      cause: 'cannot load cuda provider',
    });
    // Sticky: the next model must NOT pay for another failed GPU attempt.
    const before = t.calls.length;
    await constructEmbedPipeline(t, 'model-b', {});
    expect(t.calls.slice(before).map((c) => c.opts.device)).toEqual(['cpu']);
    expect(currentEmbedDeviceDecision().device).toBe('cpu');
  });

  it('forced cpu never attempts cuda', async () => {
    process.env[EMBED_DEVICE_ENV] = 'cpu';
    const t = fakeTransformers(new Set());
    await constructEmbedPipeline(t, 'model-a', {});
    expect(t.calls.map((c) => c.opts.device)).toEqual(['cpu']);
  });

  it('NEVER passes dtype, on any path (the vector-space invariant)', async () => {
    for (const pref of ['gpu', 'cpu']) {
      _resetEmbedDeviceState();
      process.env[EMBED_DEVICE_ENV] = pref;
      const t = fakeTransformers(new Set(['cuda']));
      await constructEmbedPipeline(t, 'model-a', { intraOpNumThreads: 1 });
      expect(t.calls.length).toBeGreaterThan(0);
      for (const c of t.calls) {
        expect(c.opts).not.toHaveProperty('dtype');
        expect(c.opts.session_options).toEqual({ intraOpNumThreads: 1 });
      }
    }
  });
});

describe('applyWorkerDeviceReport — the worker tells the main thread what actually ran', () => {
  beforeEach(() => {
    _resetEmbedDeviceState();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    _resetEmbedDeviceState();
    vi.restoreAllMocks();
  });

  it('records the device and a demotion', () => {
    expect(
      applyWorkerDeviceReport({
        kind: 'device',
        model: 'm',
        device: 'cpu',
        demotion: { from: 'cuda', stage: 'construct', cause: 'boom' },
      }),
    ).toBe(true);
    expect(embedPipelineDevices()).toEqual({ m: 'cpu' });
    expect(embedDeviceDemotion()).toMatchObject({ from: 'cuda', model: 'm', stage: 'construct', cause: 'boom' });
  });

  it('ignores malformed reports instead of trusting them', () => {
    expect(applyWorkerDeviceReport(null)).toBe(false);
    expect(applyWorkerDeviceReport({ kind: 'embed_ok', model: 'm', device: 'cpu' })).toBe(false);
    expect(applyWorkerDeviceReport({ kind: 'device', model: 'm', device: 'tpu' })).toBe(false);
    expect(applyWorkerDeviceReport({ kind: 'device', device: 'cpu' })).toBe(false);
    expect(embedPipelineDevices()).toEqual({});
  });
});

/**
 * The REAL worker script, driven with a fake transformers module (the
 * `workerData.transformersSpecifier` seam), so its fallback is proven without a
 * GPU. Model ids select the fake's behaviour.
 */
describe('local-embedder-worker.script.mjs — the hot path falls back too', () => {
  let dir: string;
  let fake: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'embed-device-worker-'));
    fake = join(dir, 'fake-transformers.mjs');
    writeFileSync(
      fake,
      [
        'export const env = { allowLocalModels: false, allowRemoteModels: true };',
        'export async function pipeline(task, model, opts = {}) {',
        "  if (opts.device === 'cuda' && model === 'fail-construct') {",
        "    throw new Error('OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library');",
        '  }',
        '  const pipe = async (text) => {',
        "    if (opts.device === 'cuda' && model === 'fail-inference') throw new Error('CUDA failure 2: out of memory');",
        "    if (text === 'bad-input') throw new Error('input rejected');",
        // [device code, dtype passed?, text length]
        "    return { data: Float32Array.from([opts.device === 'cuda' ? 1 : 2, 'dtype' in opts ? 1 : 0, text.length]) };",
        '  };',
        '  return pipe;',
        '}',
      ].join('\n'),
    );
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  async function drive(device: string, requests: Array<{ model: string; text: string }>) {
    const worker = new Worker(join(SRC, 'local-embedder-worker.script.mjs'), {
      workerData: { device, transformersSpecifier: pathToFileURL(fake).href },
    });
    const messages: Array<Record<string, unknown>> = [];
    try {
      await new Promise<void>((resolveReady, reject) => {
        worker.once('error', reject);
        worker.on('message', (m: Record<string, unknown>) => {
          messages.push(m);
          if (m.kind === 'ready') resolveReady();
        });
      });
      const results: Array<Record<string, unknown>> = [];
      for (const [i, r] of requests.entries()) {
        const id = i + 1;
        const done = new Promise<Record<string, unknown>>((resolveMsg) => {
          const on = (m: Record<string, unknown>) => {
            if (m.id === id) {
              worker.off('message', on);
              resolveMsg(m);
            }
          };
          worker.on('message', on);
        });
        worker.postMessage({ kind: 'embed', id, text: r.text, model: r.model });
        results.push(await done);
      }
      return { results, devices: messages.filter((m) => m.kind === 'device') };
    } finally {
      await worker.terminate();
    }
  }

  it('builds on cuda when handed cuda and it constructs', async () => {
    const { results, devices } = await drive('cuda', [{ model: 'ok', text: 'hello' }]);
    expect(results[0]).toMatchObject({ kind: 'embed_ok', vector: [1, 0, 5] });
    expect(devices).toEqual([{ kind: 'device', model: 'ok', device: 'cuda', demotion: null }]);
  });

  it('a cuda session that will not construct is rebuilt on cpu, reported, and the switch is sticky', async () => {
    const { results, devices } = await drive('cuda', [
      { model: 'fail-construct', text: 'hello' },
      { model: 'ok', text: 'hi' },
    ]);
    expect(results[0]).toMatchObject({ kind: 'embed_ok', vector: [2, 0, 5] });
    expect(devices[0]).toMatchObject({
      model: 'fail-construct',
      device: 'cpu',
      demotion: { from: 'cuda', stage: 'construct' },
    });
    expect(String((devices[0].demotion as { cause: string }).cause)).toContain('Failed to load shared library');
    // Sticky: a later model goes straight to cpu.
    expect(results[1]).toMatchObject({ kind: 'embed_ok', vector: [2, 0, 2] });
    expect(devices[1]).toMatchObject({ model: 'ok', device: 'cpu', demotion: null });
  });

  it('a forward pass that fails ON THE GPU is retried once on cpu instead of failing the embed', async () => {
    const { results, devices } = await drive('cuda', [{ model: 'fail-inference', text: 'hello' }]);
    expect(results[0]).toMatchObject({ kind: 'embed_ok', vector: [2, 0, 5] });
    expect(devices.at(-1)).toMatchObject({
      model: 'fail-inference',
      device: 'cpu',
      demotion: { from: 'cuda', stage: 'inference', cause: 'CUDA failure 2: out of memory' },
    });
  });

  it('an input error on the GPU is the request’s own — no demotion', async () => {
    const { results, devices } = await drive('cuda', [{ model: 'ok', text: 'bad-input' }]);
    expect(results[0]).toMatchObject({ kind: 'embed_err', error: 'input rejected' });
    expect(devices).toEqual([{ kind: 'device', model: 'ok', device: 'cuda', demotion: null }]);
  });

  it('handed cpu, never attempts cuda', async () => {
    const { results, devices } = await drive('cpu', [{ model: 'fail-construct', text: 'hey' }]);
    expect(results[0]).toMatchObject({ kind: 'embed_ok', vector: [2, 0, 3] });
    expect(devices).toEqual([{ kind: 'device', model: 'fail-construct', device: 'cpu', demotion: null }]);
  });
});
