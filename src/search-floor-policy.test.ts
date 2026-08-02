/**
 * The floor/fusion pairing — context-injection-audit-2026-07-28 P-001 (D-020).
 *
 * These assertions are COMPILE-TIME. `@ts-expect-error` fails the build when the
 * error it expects stops happening, so if someone ever relaxes `fusionMode` back
 * to optional on the floor branch, THIS FILE goes red — which is the only way a
 * type-level guard can have a regression test at all. There is deliberately no
 * runtime behaviour under test here; `hybrid-backend.test.ts` covers that.
 */
import { describe, expect, it } from 'vitest';
import type { SearchOptions } from './backend';

describe('SearchFloorPolicy', () => {
  it('REFUSES an absolute floor that does not say which fusion shape it applies to', () => {
    // @ts-expect-error minScore without fusionMode is the silent-bypass shape D-020 named.
    const bad: SearchOptions = { scope: 's', minScore: 0.45 };
    expect(bad).toBeTruthy();
  });

  it('REFUSES a relative floor that does not say which fusion shape it applies to', () => {
    // @ts-expect-error minScoreRatio is a floor too — same requirement.
    const bad: SearchOptions = { scope: 's', minScoreRatio: 0.8 };
    expect(bad).toBeTruthy();
  });

  it('ACCEPTS a floor that names its fusion mode', () => {
    const pushLike: SearchOptions = { scope: 's', minScore: 0.45, fusionMode: 'cosine-gated' };
    const recallLike: SearchOptions = { scope: 's', minScore: 0.45, fusionMode: 'floored-union' };
    expect([pushLike.fusionMode, recallLike.fusionMode]).toEqual(['cosine-gated', 'floored-union']);
  });

  it('ACCEPTS an env-tunable floor that resolves to undefined, when the mode is stated', () => {
    // The live push path is exactly this shape (injection.ts: injectMinScore()
    // returns number | undefined). A caller who has named the fusion mode has
    // done the thinking the guard exists to force, floor or no floor.
    const maybeFloor: number | undefined = undefined;
    const opts: SearchOptions = { scope: 's', minScore: maybeFloor, fusionMode: 'cosine-gated' };
    expect(opts.fusionMode).toBe('cosine-gated');
  });

  it('ACCEPTS an unfloored pull-path query with no fusion mode at all', () => {
    // Dedup, classification and memory:search legitimately want unfloored
    // nearest-neighbour behaviour; the guard must not tax them.
    const pull: SearchOptions = { scope: 's', limit: 8 };
    expect(pull.limit).toBe(8);
  });
});
