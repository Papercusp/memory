/**
 * Regression guard for the WI-3792 host-meltdown class (EI-9021).
 *
 * ROOT CAUSE (WI-3792, 2026-07-10): ONNX Runtime defaults its intra-op thread
 * pool to EVERY core and SPIN-WAITS the idle threads. On the 128-core host each
 * operator process (main host + every sidecar + cluster worker) that lazily
 * loaded a local-embedder pipeline grew a ~128-thread busy-spin pool → hundreds
 * of spinning threads, loadavg 2000-3000, host-wide stutter that starved the
 * event loop of CPU. That CPU starvation is what pushed the request event-loop
 * lag p95 past its SLO budget (EI-9021: 648ms vs the 600ms warn budget) — a
 * SYMPTOM of this root cause, not a separate loop bug.
 *
 * THE FIX (already landed in code): cap the ONNX session's thread pools to a
 * small, bounded pool at every pipeline-load site — embeds are single-request,
 * latency-tolerant background work, so a full-core spin pool per process is pure
 * waste. The cap lives in TWO files that MUST stay in sync (the plain-JS worker
 * can't import the TS module):
 *   - local-embedder-worker.ts        → `ORT_SESSION_OPTIONS` (exported const)
 *   - local-embedder-worker.script.mjs → an inlined copy of the same literal
 * The source comments say "keep the two in sync" — a manual invariant with no
 * mechanical guard, i.e. exactly the drift trap that re-arms the meltdown if one
 * copy is reverted, raised, or the two diverge.
 *
 * THIS TEST is that missing recurrence guard. It fails if either copy loses the
 * cap (regresses toward the all-cores ONNX default) or if the two copies drift
 * apart — so the WI-3792 fix cannot silently rot back into a host meltdown.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import { ORT_SESSION_OPTIONS } from './local-embedder-worker';

// A bounded intra-op pool. The landed value is 4; this ceiling leaves headroom
// to tune (4→8) while still definitively excluding the unbounded ONNX default
// (0 = "auto" = every physical core — 128 on the incident host). Anything above
// this is treated as a regression toward the meltdown, not a legitimate tune.
const MAX_INTRA_OP = 8;
const MAX_INTER_OP = 4;

/** Parse the { intraOpNumThreads, interOpNumThreads } literal out of the plain
 *  JS worker script. Key-order tolerant; each number is matched independently so
 *  a reformat of the literal doesn't silently pass an unparsed value. */
function readScriptOrtOptions(): { intraOpNumThreads: number; interOpNumThreads: number } {
  const scriptPath = resolve(
    dirname(fileURLToPath(import.meta.url)),
    'local-embedder-worker.script.mjs',
  );
  const src = readFileSync(scriptPath, 'utf8');
  const intra = src.match(/intraOpNumThreads\s*:\s*(\d+)/);
  const inter = src.match(/interOpNumThreads\s*:\s*(\d+)/);
  expect(intra, 'script.mjs must set intraOpNumThreads (WI-3792 cap)').not.toBeNull();
  expect(inter, 'script.mjs must set interOpNumThreads (WI-3792 cap)').not.toBeNull();
  return {
    intraOpNumThreads: Number(intra![1]),
    interOpNumThreads: Number(inter![1]),
  };
}

describe('ONNX Runtime thread-pool cap (WI-3792 / EI-9021 host-meltdown guard)', () => {
  it('caps the TS ORT_SESSION_OPTIONS intra/inter-op pools to a small bounded value', () => {
    expect(Number.isInteger(ORT_SESSION_OPTIONS.intraOpNumThreads)).toBe(true);
    expect(ORT_SESSION_OPTIONS.intraOpNumThreads).toBeGreaterThanOrEqual(1);
    // The load-bearing assertion: NOT the unbounded all-cores default.
    expect(ORT_SESSION_OPTIONS.intraOpNumThreads).toBeLessThanOrEqual(MAX_INTRA_OP);

    expect(Number.isInteger(ORT_SESSION_OPTIONS.interOpNumThreads)).toBe(true);
    expect(ORT_SESSION_OPTIONS.interOpNumThreads).toBeGreaterThanOrEqual(1);
    expect(ORT_SESSION_OPTIONS.interOpNumThreads).toBeLessThanOrEqual(MAX_INTER_OP);
  });

  it('keeps the plain-JS worker script cap in sync with the TS export', () => {
    const script = readScriptOrtOptions();
    // The "keep the two in sync" comment invariant, mechanized: a divergence
    // (one file reverted/tuned without the other) re-arms the meltdown on
    // whichever load path uses the stale copy.
    expect(script.intraOpNumThreads).toBe(ORT_SESSION_OPTIONS.intraOpNumThreads);
    expect(script.interOpNumThreads).toBe(ORT_SESSION_OPTIONS.interOpNumThreads);
  });

  it('the worker-script cap is itself bounded (independent of the TS copy)', () => {
    const script = readScriptOrtOptions();
    expect(script.intraOpNumThreads).toBeGreaterThanOrEqual(1);
    expect(script.intraOpNumThreads).toBeLessThanOrEqual(MAX_INTRA_OP);
    expect(script.interOpNumThreads).toBeGreaterThanOrEqual(1);
    expect(script.interOpNumThreads).toBeLessThanOrEqual(MAX_INTER_OP);
  });
});
