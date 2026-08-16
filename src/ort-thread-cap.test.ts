/**
 * Regression guard for the ONNX-thread-pool CPU-burn class.
 *
 * Two incidents, ONE knob — the second is what the first's fix left open:
 *
 * 1. WI-3792 / EI-9021 (2026-07-10, BIG host). ONNX Runtime defaults its
 *    intra-op thread pool to EVERY core and SPIN-WAITS the idle threads. On the
 *    128-core host each operator process (main host + every sidecar + cluster
 *    worker) that lazily loaded a local-embedder pipeline grew a ~128-thread
 *    busy-spin pool → loadavg 2000-3000, host-wide stutter that starved the
 *    event loop of CPU (EI-9021's 648ms request-loop-lag p95 was a SYMPTOM of
 *    this, not a separate loop bug). Fixed by pinning `intraOpNumThreads: 4`.
 *
 * 2. EI-20493854163389792 (SMALL host) — the same knob, unfixed at the other
 *    end. That `4` was an ABSOLUTE constant, so it capped the 128-core box and
 *    never scaled DOWN. The packaged 0.0.16 desktop on a fresh 8-vCPU Ubuntu
 *    guest measured 415.2% average process CPU (360/418/422/465/411 over 5×1s
 *    pidstat samples) while the FIRST-RUN UI sat idle — ≈4 saturated intra-op
 *    threads plus main, i.e. HALF that machine. `/api/health/deep` reported
 *    loopLag pressure=ok at the same moment, which is the tell that the burn is
 *    on native ORT threads rather than the JS event loop. First run is the worst
 *    case (the whole seeded corpus is unembedded, so the embed-backfill sweep
 *    keeps the pool hot) and the packaged default embeds IN-PROCESS (the embed
 *    sidecar is opt-in), which is why it lands on the operator's own PID.
 *    Fixed by making the cap a SHARE of the host instead of a constant.
 *
 * The cap lives in TWO files that MUST stay in sync (the plain-JS worker can't
 * import the TS module):
 *   - local-embedder-worker.ts         → `resolveIntraOpNumThreads` + constants
 *   - local-embedder-worker.script.mjs → an inlined copy of the same formula
 * The source comments say "keep the two in sync" — a manual invariant with no
 * mechanical guard, i.e. exactly the drift trap that re-arms a meltdown if one
 * copy is reverted, raised, or the two diverge.
 *
 * THIS TEST is that guard. It fails if either copy loses the ceiling (regresses
 * toward the all-cores ONNX default), if the sizing stops being host-relative
 * (regresses to the hardcoded 4 that burned half the 8-vCPU guest), or if the
 * two copies drift apart.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import {
  BACKGROUND_HOST_SHARE_DIVISOR,
  MAX_INTRA_OP_THREADS,
  ORT_SESSION_OPTIONS,
  resolveIntraOpNumThreads,
} from './local-embedder-worker';

// A bounded intra-op pool. The landed ceiling is 4; this bound leaves headroom
// to tune (4→8) while still definitively excluding the unbounded ONNX default
// (0 = "auto" = every physical core — 128 on the incident host). Anything above
// this is treated as a regression toward the meltdown, not a legitimate tune.
const MAX_INTRA_OP = 8;
const MAX_INTER_OP = 4;

/** The guest the EI-20493854163389792 burn was measured on. */
const REPORTED_GUEST_CORES = 8;

/** Parse the cap FORMULA's inputs out of the plain-JS worker script. The intra
 *  value is no longer a literal (it is derived from the host), so the two copies
 *  are compared by their constants + the shared formula rather than by a number
 *  that only holds on one machine. Each is matched independently so a reformat
 *  can't silently pass an unparsed value. */
function readScriptOrtOptions(): {
  maxIntraOpThreads: number;
  backgroundHostShareDivisor: number;
  interOpNumThreads: number;
} {
  const scriptPath = resolve(
    dirname(fileURLToPath(import.meta.url)),
    'local-embedder-worker.script.mjs',
  );
  const src = readFileSync(scriptPath, 'utf8');
  const maxIntra = src.match(/MAX_INTRA_OP_THREADS\s*=\s*(\d+)/);
  const divisor = src.match(/BACKGROUND_HOST_SHARE_DIVISOR\s*=\s*(\d+)/);
  const inter = src.match(/interOpNumThreads\s*:\s*(\d+)/);
  expect(maxIntra, 'script.mjs must set MAX_INTRA_OP_THREADS (WI-3792 ceiling)').not.toBeNull();
  expect(
    divisor,
    'script.mjs must set BACKGROUND_HOST_SHARE_DIVISOR (EI-20493854163389792 host share)',
  ).not.toBeNull();
  expect(inter, 'script.mjs must set interOpNumThreads (WI-3792 cap)').not.toBeNull();
  return {
    maxIntraOpThreads: Number(maxIntra![1]),
    backgroundHostShareDivisor: Number(divisor![1]),
    interOpNumThreads: Number(inter![1]),
  };
}

describe('ONNX Runtime thread-pool cap (WI-3792 big-host + EI-20493854163389792 small-host guard)', () => {
  it('caps the resolved ORT_SESSION_OPTIONS pools to a small bounded value', () => {
    expect(Number.isInteger(ORT_SESSION_OPTIONS.intraOpNumThreads)).toBe(true);
    expect(ORT_SESSION_OPTIONS.intraOpNumThreads).toBeGreaterThanOrEqual(1);
    // The load-bearing assertion: NOT the unbounded all-cores default.
    expect(ORT_SESSION_OPTIONS.intraOpNumThreads).toBeLessThanOrEqual(MAX_INTRA_OP);

    expect(Number.isInteger(ORT_SESSION_OPTIONS.interOpNumThreads)).toBe(true);
    expect(ORT_SESSION_OPTIONS.interOpNumThreads).toBeGreaterThanOrEqual(1);
    expect(ORT_SESSION_OPTIONS.interOpNumThreads).toBeLessThanOrEqual(MAX_INTER_OP);
  });

  it('never exceeds the WI-3792 ceiling, however large the host', () => {
    // The 128-core incident host, and an absurd one for good measure.
    for (const cores of [16, 32, 128, 1024]) {
      expect(resolveIntraOpNumThreads(cores)).toBe(MAX_INTRA_OP_THREADS);
    }
  });

  it('leaves an 8-vCPU guest most of its machine (EI-20493854163389792)', () => {
    // THE regression assertion. The hardcoded 4 that shipped in 0.0.16 gave
    // background embedding HALF of this guest and pegged it behind an idle
    // first-run UI. A revert to any host-independent constant ≥3 fails here.
    const onGuest = resolveIntraOpNumThreads(REPORTED_GUEST_CORES);
    expect(onGuest).toBeGreaterThanOrEqual(1);
    expect(onGuest).toBeLessThanOrEqual(REPORTED_GUEST_CORES / BACKGROUND_HOST_SHARE_DIVISOR);
    expect(onGuest).toBeLessThan(4);
  });

  it('sizes the pool RELATIVE to the host rather than as a constant', () => {
    // A hardcoded value (the exact shape of the 0.0.16 bug) makes every one of
    // these equal, so this fails the moment the sizing stops being relative.
    expect(resolveIntraOpNumThreads(4)).toBeLessThan(resolveIntraOpNumThreads(64));
    expect(resolveIntraOpNumThreads(8)).toBeLessThan(resolveIntraOpNumThreads(64));

    // Monotonic, and never more than the declared share of the host.
    let prev = 0;
    for (const cores of [1, 2, 4, 8, 16, 64, 128]) {
      const got = resolveIntraOpNumThreads(cores);
      expect(got).toBeGreaterThanOrEqual(prev);
      expect(got).toBeLessThanOrEqual(Math.max(1, cores / BACKGROUND_HOST_SHARE_DIVISOR));
      prev = got;
    }
  });

  it('floors at one thread on a tiny or unreadable host', () => {
    // A 0/NaN/negative reading must degrade to one thread, NEVER fall through to
    // ONNX's all-cores default — the failure mode is silent and host-wide.
    for (const cores of [1, 2, 3, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const got = resolveIntraOpNumThreads(cores);
      expect(got).toBeGreaterThanOrEqual(1);
      expect(got).toBeLessThanOrEqual(MAX_INTRA_OP_THREADS);
    }
    expect(resolveIntraOpNumThreads(Number.NaN)).toBe(1);
    expect(resolveIntraOpNumThreads(0)).toBe(1);
  });

  it('keeps the plain-JS worker script cap in sync with the TS export', () => {
    const script = readScriptOrtOptions();
    // The "keep the two in sync" comment invariant, mechanized: a divergence
    // (one file reverted/tuned without the other) re-arms the burn on whichever
    // load path uses the stale copy.
    expect(script.maxIntraOpThreads).toBe(MAX_INTRA_OP_THREADS);
    expect(script.backgroundHostShareDivisor).toBe(BACKGROUND_HOST_SHARE_DIVISOR);
    expect(script.interOpNumThreads).toBe(ORT_SESSION_OPTIONS.interOpNumThreads);
  });

  it('the worker-script cap is itself bounded (independent of the TS copy)', () => {
    const script = readScriptOrtOptions();
    expect(script.maxIntraOpThreads).toBeGreaterThanOrEqual(1);
    expect(script.maxIntraOpThreads).toBeLessThanOrEqual(MAX_INTRA_OP);
    // A divisor of 1 would hand the whole host to background embedding — the
    // 0.0.16 behaviour by another route.
    expect(script.backgroundHostShareDivisor).toBeGreaterThanOrEqual(2);
    expect(script.interOpNumThreads).toBeGreaterThanOrEqual(1);
    expect(script.interOpNumThreads).toBeLessThanOrEqual(MAX_INTER_OP);
  });

  it('the worker script actually APPLIES the share formula (not just declares it)', () => {
    // Guards the shape of the mirror: constants present but unused would pass
    // every assertion above while the worker still spun a hardcoded pool.
    const scriptPath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      'local-embedder-worker.script.mjs',
    );
    const src = readFileSync(scriptPath, 'utf8');
    expect(src).toMatch(/intraOpNumThreads:\s*Math\.max\(/);
    expect(src).toContain('MAX_INTRA_OP_THREADS');
    expect(src).toContain('BACKGROUND_HOST_SHARE_DIVISOR');
    expect(src).toContain('hostParallelism()');
    // The literal that caused the bug must not come back.
    expect(src).not.toMatch(/intraOpNumThreads:\s*\d+/);
  });
});
