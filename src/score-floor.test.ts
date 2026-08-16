import { describe, it, expect } from 'vitest';
import { applyScoreFloor } from './score-floor';
import type { MemoryEntry } from './backend';

const e = (id: string, score?: number): MemoryEntry => ({
  id,
  text: id,
  scope: 's',
  ...(score !== undefined ? { score } : {}),
});

describe('applyScoreFloor', () => {
  it('returns everything when no floor is set', () => {
    const xs = [e('a', 0.9), e('b', 0.1)];
    expect(applyScoreFloor(xs, {}).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('absolute floor drops sub-threshold hits', () => {
    const xs = [e('a', 0.6), e('b', 0.4), e('c', 0.3)];
    expect(applyScoreFloor(xs, { minScore: 0.45 }).map((x) => x.id)).toEqual(['a']);
  });

  it('returns EMPTY for an all-weak (hard-negative) result — the FP fix', () => {
    // hard-negative top score measured ~0.385; everything below the floor
    const xs = [e('a', 0.39), e('b', 0.3), e('c', 0.22)];
    expect(applyScoreFloor(xs, { minScore: 0.45 })).toEqual([]);
  });

  it('keeps a real-hit result above the floor — recall preserved', () => {
    // real-hit classes measured ~0.51–0.58
    const xs = [e('a', 0.55), e('b', 0.5)];
    expect(applyScoreFloor(xs, { minScore: 0.45 }).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('relative floor trims the weak tail below ratio × top', () => {
    const xs = [e('a', 0.8), e('b', 0.6), e('c', 0.3)];
    // top 0.8, ratio 0.6 → floor 0.48 → keep a,b drop c
    expect(applyScoreFloor(xs, { minScoreRatio: 0.6 }).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('combines absolute + relative and takes the stricter', () => {
    const xs = [e('a', 0.9), e('b', 0.5), e('c', 0.46)];
    // abs 0.45, rel 0.6×0.9=0.54 → floor 0.54 → keep a only
    expect(applyScoreFloor(xs, { minScore: 0.45, minScoreRatio: 0.6 }).map((x) => x.id)).toEqual(['a']);
  });

  it('never drops unscored entries (can not judge them)', () => {
    const xs = [e('a', 0.6), e('b')];
    expect(applyScoreFloor(xs, { minScore: 0.45 }).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('does not mutate the input', () => {
    const xs = [e('a', 0.6), e('b', 0.1)];
    const copy = [...xs];
    applyScoreFloor(xs, { minScore: 0.45 });
    expect(xs).toEqual(copy);
  });
});
