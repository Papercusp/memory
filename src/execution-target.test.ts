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
 *   1. `EMBED_REQUESTED_EXECUTION` claims the embed path pins no device/dtype.
 *      That is a claim ABOUT CODE, and the derived-truth ladder says such a claim
 *      must be pinned or it drifts. The source scan below fails the moment an
 *      embedder starts passing `device:`/`dtype:` to `pipeline()` without this
 *      constant being updated with it.
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

import {
  EMBED_REQUESTED_EXECUTION,
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

  describe('EMBED_REQUESTED_EXECUTION is pinned to the embedder sources', () => {
    const scanned = EMBEDDER_SOURCES.map((file) => ({
      file,
      blocks: pipelineOptionBlocks(readFileSync(join(SRC, file), 'utf8')),
    }));

    it('the scan actually found a pipeline call in every embedder (positive control)', () => {
      // If this fails, every "no device requested" assertion below is vacuous —
      // the instrument matched nothing, which reads identically to a clean bill.
      for (const { file, blocks } of scanned) {
        expect(blocks.length, `${file} — no pipeline('feature-extraction', …) options block found`).toBeGreaterThan(0);
      }
    });

    it('the scan can see inside those blocks (positive control)', () => {
      // `session_options` is passed by every one of these call sites. If the
      // brace-matcher regressed to returning empty or wrong slices, this catches
      // it before the device/dtype assertions can report a false absence.
      for (const { file, blocks } of scanned) {
        for (const block of blocks) {
          expect(block, `${file} — extracted block does not contain the known session_options key`).toContain(
            'session_options',
          );
        }
      }
    });

    it('no embedder pins a device, matching EMBED_REQUESTED_EXECUTION.device === null', () => {
      const offenders = scanned
        .filter(({ blocks }) => blocks.some((b) => /(^|[^\w])device\s*:/.test(b)))
        .map(({ file }) => file);
      expect(
        offenders,
        `${offenders.join(', ')} now pass a device to pipeline(). That is a real change to how embeds ` +
          'execute — update EMBED_REQUESTED_EXECUTION.device (and embedExecutionTarget) to report it, ' +
          'rather than leaving /healthz claiming no device is requested.',
      ).toEqual([]);
      expect(EMBED_REQUESTED_EXECUTION.device).toBeNull();
    });

    it('no embedder pins a dtype, matching EMBED_REQUESTED_EXECUTION.dtype === null', () => {
      const offenders = scanned
        .filter(({ blocks }) => blocks.some((b) => /(^|[^\w])dtype\s*:/.test(b)))
        .map(({ file }) => file);
      expect(
        offenders,
        `${offenders.join(', ')} now pass a dtype to pipeline(). Update EMBED_REQUESTED_EXECUTION.dtype ` +
          'so /healthz reports the weight format actually in use.',
      ).toEqual([]);
      expect(EMBED_REQUESTED_EXECUTION.dtype).toBeNull();
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

    it('says so in `why` rather than implying the question was settled', () => {
      expect(embedExecutionTarget().why).toContain('UNKNOWN');
    });

    it('still reports the CPU target, because that part needs no probe', () => {
      // The root cause — nothing requests a device — is static and always
      // knowable, so the detector is useful even with the probe unresolved.
      expect(embedExecutionTarget().device).toBe('cpu');
    });
  });

  describe('after the probe resolves', () => {
    it('answers from a real measurement and stays internally consistent', async () => {
      await ensureEmbedBackendsProbed();
      const health = embedExecutionHealth();
      expect(health.probe === 'ok' || health.probe === 'failed').toBe(true);

      if (health.probe === 'ok') {
        expect(Array.isArray(health.backends)).toBe(true);
        expect(health.backends!.length).toBeGreaterThan(0);
        // Every backend entry is a well-formed pair, so `bundled` can be trusted.
        for (const b of health.backends!) {
          expect(typeof b.name).toBe('string');
          expect(typeof b.bundled).toBe('boolean');
        }
        expect(typeof health.gpuBundled).toBe('boolean');
        expect(health.probeError).toBeNull();
        // The rolled-up boolean must never travel without the names behind it:
        // a bundled CUDA provider and a bundled WebGPU one both render `true`
        // and imply completely different next steps. On this host it is webgpu
        // (cuda reports bundled:false), so a bare `true` read as "CUDA is
        // available" would be exactly the wrong conclusion.
        expect(Array.isArray(health.gpuBackendsBundled)).toBe(true);
        expect(health.gpuBundled).toBe(health.gpuBackendsBundled!.length > 0);
        // The verdict must be derived from the list, not stored independently.
        const derived = health.backends!.some(
          (b) => b.bundled && ['cuda', 'tensorrt', 'webgpu', 'dml', 'coreml', 'rocm'].includes(b.name.toLowerCase()),
        );
        expect(health.gpuBundled).toBe(derived);
      } else {
        // A failed probe must degrade to UNKNOWN with the reason attached —
        // never to an empty list that would render as "no GPU bundled".
        expect(health.gpuBundled).toBeNull();
        expect(health.backends).toBeNull();
        expect(health.probeError).toBeTruthy();
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
