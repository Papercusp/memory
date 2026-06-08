import { describe, it, expect } from 'vitest';
import { queryLevelPRF, renderFloorSweepMarkdown } from './floor-sweep';
import type { QueryOutcome } from './types';

const pos = (id: string, hit: boolean): QueryOutcome => ({
  queryId: id,
  class: 'lexical-gap',
  expected: ['k1'],
  rankedKeys: hit ? ['k1'] : ['other'],
  rawHits: hit ? 1 : 1,
  ms: 1,
});
const neg = (id: string, returnedSomething: boolean): QueryOutcome => ({
  queryId: id,
  class: 'hard-negative',
  expected: [],
  rankedKeys: [],
  rawHits: returnedSomething ? 1 : 0,
  ms: 1,
});

describe('queryLevelPRF (P-031)', () => {
  it('counts a positive with its answer in top-k as TP, else FN', () => {
    const r = queryLevelPRF([pos('a', true), pos('b', false)]);
    expect(r.tp).toBe(1);
    expect(r.fn).toBe(1);
    expect(r.recall).toBeCloseTo(0.5, 5);
  });

  it('counts a hard-negative that returns anything as FP', () => {
    const r = queryLevelPRF([pos('a', true), neg('n1', true), neg('n2', false)]);
    expect(r.tp).toBe(1);
    expect(r.fp).toBe(1); // n1 returned something
    expect(r.precision).toBeCloseTo(0.5, 5); // 1/(1+1)
  });

  it('perfect floor: all positives hit, all hard-negatives empty → F1 = 1', () => {
    const r = queryLevelPRF([pos('a', true), pos('b', true), neg('n1', false), neg('n2', false)]);
    expect(r.precision).toBe(1);
    expect(r.recall).toBe(1);
    expect(r.f1).toBe(1);
  });

  it('a too-high floor (positives buried) tanks recall even with zero FP', () => {
    const r = queryLevelPRF([pos('a', false), pos('b', false), neg('n1', false)]);
    expect(r.fp).toBe(0);
    expect(r.recall).toBe(0);
    expect(r.f1).toBe(0);
  });

  it('renderFloorSweepMarkdown marks the best floor + tabulates', () => {
    const md = renderFloorSweepMarkdown({
      points: [
        { floor: 0, fpAt5: 1, r10: 0.97, mrr: 0.86, exactIdMrr: 0.99, positivesEmptied: 0,
          prf: { tp: 9, fp: 9, fn: 1, precision: 0.5, recall: 0.9, f1: 0.64 } },
        { floor: 0.45, fpAt5: 0, r10: 0.95, mrr: 0.85, exactIdMrr: 0.98, positivesEmptied: 1,
          prf: { tp: 9, fp: 0, fn: 1, precision: 1, recall: 0.9, f1: 0.95 } },
      ],
      bestFloor: 0.45,
      bestF1: 0.95,
    });
    expect(md).toContain('| floor |');
    expect(md).toContain('0.45 ⭐');
    expect(md).toContain('F1-max floor: 0.45');
  });
});
