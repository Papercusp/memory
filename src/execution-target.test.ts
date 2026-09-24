/**
 * execution-target.test.ts — the recurrence guard for the embed-execution detector.
 *
 * The bug this closes is not "embeds are slow". It is that embed latency had no
 * DETECTOR: `/healthz` described the request path (queue depth, warm models,
 * worker breakers) and said nothing about the device, so fourteen work-items
 * filed between 2026-08-27 and 2026-09-06 all searched the scheduler for a cause
 * that lived in the execution target. Proving it required reading
 * `/proc/<pid>/maps` of the live sidecar.
 *
 * So the guard has to protect the DETECTOR's honesty, in both directions:
 *
 *   1. `embedRequestedExecution()` claims every embed path constructs through
 *      the device selector (embed-device.ts) and pins no dtype. That is a claim
 *      ABOUT CODE, and the derived-truth ladder says such a claim must be pinned
 *      or it drifts. The source scan below fails the moment an embedder builds a
 *      pipeline around the selector, or anything passes `dtype:`. (Until
 *      2026-09-24 this pinned the opposite — that NO device was requested.)
 *
 *   2. An unmeasured probe must never render as a confident `false`. A detector
 *      that reports "no GPU bundled" from a measurement it never took recreates
 *      exactly the mistake it exists to prevent, and is strictly worse than no
 *      detector, because the next reader believes it.
 *
 * The source scan carries a POSITIVE CONTROL. An absence proved by a scan that
 * silently matched nothing is not an absence — if the control stops finding
 * `session_options` (which every one of these call sites demonstrably passes),
 * the instrument is broken and the test says so instead of reporting a clean
 * bill.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { beforeEach, describe, expect, it } from 'vitest';

import { _resetEmbedDeviceState, applyWorkerDeviceReport } from './embed-device';
import {
  embedRequestedExecution,
  _resetEmbedExecutionProbe,
  embedExecutionHealth,
  embedExecutionTarget,
  embedGpuProviderAvailable,
  embedProviderLibraries,
  ensureEmbedBackendsProbed,
} from './execution-target';

const SRC = dirname(fileURLToPath(import.meta.url));

/**
 * Every module that constructs a feature-extraction pipeline. Hardcoding the
 * list is deliberate: a NEW embedder file must be added here consciously, and
 * the completeness assertion below fails if one of these stops constructing a
 * pipeline (a rename that would otherwise silently shrink the scan's coverage).
 */
const EMBEDDER_SOURCES = [
  'gemma-embedder.ts',
  'granite-embedder.ts',
  'harrier-embedder.ts',
  'local-embedder-worker.ts',
  'qwen3-embedder.ts',
] as const;

const PIPELINE_CALL = "pipeline('feature-extraction'";

/** The device selector — the one .ts place allowed to call pipeline() directly. */
const SELECTOR_SOURCE = 'embed-device.ts';
/** The worker thread's script — the hot path. */
const WORKER_SCRIPT = 'local-embedder-worker.script.mjs';

/**
 * Extract the options-object text of each `pipeline('feature-extraction', …, { … })`
 * call in one source file, by brace-matching from the call's first `{`.
 */
function pipelineOptionBlocks(source: string): string[] {
  const blocks: string[] = [];
  let from = 0;
  for (;;) {
    const call = source.indexOf(PIPELINE_CALL, from);
    if (call === -1) break;
    from = call + PIPELINE_CALL.length;
    const open = source.indexOf('{', from);
    if (open === -1) break;
    let depth = 0;
    let end = -1;
    for (let i = open; i < source.length; i += 1) {
      if (source[i] === '{') depth += 1;
      else if (source[i] === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) break;
    blocks.push(source.slice(open, end + 1));
    from = end + 1;
  }
  return blocks;
}

describe('embed execution target — the detector the 14 latency filings lacked', () => {
  beforeEach(() => {
    _resetEmbedExecutionProbe();
  });

  describe('every embed path constructs through the device selector (pinned to the sources)', () => {
    // Since 2026-09-24 the embedders DO request a device (embed-device.ts, plan
    // memory-reduction-2026-09-24 D-003). What must hold now: no embedder builds
    // a pipeline around the selector (it would silently skip the GPU choice AND
    // the CPU fallback), and nothing passes a dtype (only the device may move,
    // or new vectors drift from the stored corpus).
    const tsSources = EMBEDDER_SOURCES.map((file) => ({ file, source: readFileSync(join(SRC, file), 'utf8') }));
    // The files that DO call pipeline('feature-extraction', …) directly: the
    // selector itself and the worker script (the hot path — which the pre-2026-09-24
    // version of this scan never covered at all).
    const pipelineOwners = [SELECTOR_SOURCE, WORKER_SCRIPT].map((file) => ({
      file,
      blocks: pipelineOptionBlocks(readFileSync(join(SRC, file), 'utf8')),
    }));

    it('every embedder constructs through constructEmbedPipeline (positive control for coverage)', () => {
      for (const { file, source } of tsSources) {
        expect(source, `${file} — no constructEmbedPipeline( call found`).toContain('constructEmbedPipeline(');
      }
    });

    it('no embedder calls pipeline(\'feature-extraction\') around the selector', () => {
      const offenders = tsSources.filter(({ source }) => source.includes(PIPELINE_CALL)).map(({ file }) => file);
      expect(
        offenders,
        `${offenders.join(', ')} build a pipeline directly — route it through constructEmbedPipeline so the ` +
          'device choice, the CPU fallback and /healthz all see it.',
      ).toEqual([]);
    });

    it('the scan found and can see inside the selector + worker pipeline calls (positive control)', () => {
      // If this fails, every device/dtype assertion below is vacuous — the
      // instrument matched nothing, which reads identically to a clean bill.
      for (const { file, blocks } of pipelineOwners) {
        expect(blocks.length, `${file} — no pipeline('feature-extraction', …) options block found`).toBeGreaterThan(0);
        for (const block of blocks) {
          expect(block, `${file} — extracted block lacks the known session_options key`).toContain('session_options');
        }
      }
    });

    it('the selector and the worker pass a device on every pipeline call', () => {
      for (const { file, blocks } of pipelineOwners) {
        for (const block of blocks) {
          expect(/(^|[^\w])device\s*:/.test(block), `${file} — a pipeline call passes no device: ${block}`).toBe(true);
        }
      }
    });

    it('nothing pins a dtype, matching embedRequestedExecution().dtype === null', () => {
      const offenders = pipelineOwners
        .filter(({ blocks }) => blocks.some((b) => /(^|[^\w])dtype\s*:/.test(b)))
        .map(({ file }) => file);
      expect(
        offenders,
        `${offenders.join(', ')} now pass a dtype to pipeline(). Stored vectors come from the default ` +
          'weights; a different weight format drifts every new vector from the corpus.',
      ).toEqual([]);
      expect(embedRequestedExecution().dtype).toBeNull();
    });
  });

  describe('an unmeasured probe is UNKNOWN, never a confident answer', () => {
    it('reports availability null and probe "pending" before the probe runs', () => {
      const health = embedExecutionHealth();
      expect(health.probe).toBe('pending');
      // The load-bearing assertion: null, NOT false. A false here would read as
      // "measured: no GPU provider" and is the failure mode this module exists
      // to end.
      expect(health.gpuProviderAvailable).toBeNull();
      expect(health.providerLibraries).toBeNull();
      expect(health.defaultBundledBackends).toBeNull();
      expect(embedGpuProviderAvailable()).toBeNull();
      expect(embedProviderLibraries()).toBeNull();
    });

    it('before any pipeline constructs, reports the requested device as UNVERIFIED', () => {
      // A GPU choice can still fall back at construction, so the device the
      // next pipeline will try must not read as a measurement.
      _resetEmbedDeviceState();
      const target = embedExecutionTarget();
      expect(target.verified).toBe(false);
      expect(target.device).toBe(embedRequestedExecution().device);
      expect(target.why).toContain('unverified');
    });

    it('after a pipeline constructs, reports what actually ran as verified', () => {
      _resetEmbedDeviceState();
      applyWorkerDeviceReport({ kind: 'device', model: 'm', device: 'cpu', demotion: null });
      const target = embedExecutionTarget();
      expect(target).toMatchObject({ device: 'cpu', verified: true });
      expect(embedExecutionHealth().pipelines).toEqual({ m: 'cpu' });
      _resetEmbedDeviceState();
    });
  });

  describe('after the probe resolves', () => {
    it('answers from a real measurement and stays internally consistent', async () => {
      await ensureEmbedBackendsProbed();
      const health = embedExecutionHealth();
      expect(health.probe === 'ok' || health.probe === 'failed').toBe(true);

      if (health.probe === 'ok') {
        expect(Array.isArray(health.defaultBundledBackends)).toBe(true);
        expect(health.defaultBundledBackends!.length).toBeGreaterThan(0);
        for (const b of health.defaultBundledBackends!) {
          expect(typeof b.name).toBe('string');
          expect(typeof b.bundled).toBe('boolean');
        }
        expect(health.probeError).toBeNull();
        // Availability is derived from the LIBRARIES ON DISK, never from the
        // packaging metadata — see the next test for why.
        if (health.providerLibraries !== null) {
          expect(health.gpuProviderAvailable).toBe(
            health.providerLibraries.some((f) =>
              /^libonnxruntime_providers_(cuda|tensorrt|rocm|migraphx|dml)\./i.test(f),
            ),
          );
        } else {
          expect(health.gpuProviderAvailable).toBeNull();
        }
      } else {
        // A failed probe must degrade to UNKNOWN with the reason attached —
        // never to an empty list that would render as "no GPU available".
        expect(health.gpuProviderAvailable).toBeNull();
        expect(health.providerLibraries).toBeNull();
        expect(health.defaultBundledBackends).toBeNull();
        expect(health.probeError).toBeTruthy();
      }
    });

    it('NEVER derives availability from listSupportedBackends().bundled (the 2026-09-07 correction)', async () => {
      // MEASURED that day: installing onnxruntime-node@1.24.3 with
      // ONNXRUNTIME_NODE_INSTALL_CUDA=v12 downloads libonnxruntime_providers_cuda.so
      // (~315MB) and CUDA sessions then CONSTRUCT successfully — while
      // listSupportedBackends() STILL reports cuda:{bundled:false}, and the
      // onnxruntime_binding.node addon is byte-identical (384040 bytes) in both
      // trees. So `bundled` is publish-time packaging metadata; deriving
      // availability from it under-reports a working GPU box, which is what the
      // first version of this module did.
      await ensureEmbedBackendsProbed();
      const health = embedExecutionHealth();
      if (health.probe !== 'ok' || health.providerLibraries === null) return;

      const gpuByPackaging = (health.defaultBundledBackends ?? []).some(
        (b) => b.bundled && ['cuda', 'tensorrt', 'rocm', 'migraphx', 'dml'].includes(b.name.toLowerCase()),
      );
      const gpuByDisk = health.providerLibraries.some((f) =>
        /^libonnxruntime_providers_(cuda|tensorrt|rocm|migraphx|dml)\./i.test(f),
      );
      // The reported value must track DISK. Where the two disagree, that is
      // exactly the case the old implementation got wrong.
      expect(health.gpuProviderAvailable).toBe(gpuByDisk);
      if (gpuByPackaging !== gpuByDisk) {
        expect(health.gpuProviderAvailable).not.toBe(gpuByPackaging);
      }
    });

    it('is idempotent and safe to call concurrently', async () => {
      await Promise.all([ensureEmbedBackendsProbed(), ensureEmbedBackendsProbed(), ensureEmbedBackendsProbed()]);
      const first = embedExecutionHealth();
      await ensureEmbedBackendsProbed();
      expect(embedExecutionHealth()).toEqual(first);
    });
  });

  it('reports the ONNX thread caps beside the device that makes them load-bearing', () => {
    const { sessionOptions } = embedExecutionHealth();
    expect(Number.isInteger(sessionOptions.intraOpNumThreads)).toBe(true);
    expect(sessionOptions.intraOpNumThreads).toBeGreaterThanOrEqual(1);
    expect(Number.isInteger(sessionOptions.interOpNumThreads)).toBe(true);
    expect(sessionOptions.interOpNumThreads).toBeGreaterThanOrEqual(1);
  });
});
