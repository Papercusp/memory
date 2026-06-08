import { describe, it, expect } from 'vitest';
import { fuseCosineGated } from './hybrid-fusion';
import type { MemoryEntry } from './backend';

const e = (id: string, score?: number): MemoryEntry => ({
  id,
  text: id,
  scope: 's',
  ...(score !== undefined ? { score } : {}),
});

describe('fuseCosineGated (P-020)', () => {
  it('returns empty when the cosine gate is empty (hard-negative discipline)', () => {
    expect(fuseCosineGated([], [e('x'), e('y')])).toEqual([]);
  });

  it('keeps only cosine-gated entries — lexical-only hits are excluded', () => {
    const out = fuseCosineGated([e('a')], [e('a'), e('b')]); // b is lexical-only
    expect(out.map((x) => x.id)).toEqual(['a']);
  });

  it('lexical re-rank lifts an exact match above a cosine-only paraphrase', () => {
    // cosine order: [para(rank1), exact(rank2)]; lexical: exact is rank1.
    // exact = 1/62 + 1/61 > para = 1/61 → exact wins.
    const out = fuseCosineGated([e('para'), e('exact')], [e('exact')], 60);
    expect(out.map((x) => x.id)).toEqual(['exact', 'para']);
  });

  it('a paraphrase hit (cosine-only, no lexical match) is still returned', () => {
    const out = fuseCosineGated([e('para')], []);
    expect(out.map((x) => x.id)).toEqual(['para']);
    expect(out[0].score).toBeGreaterThan(0);
  });

  it('writes the fused RRF score and orders by it', () => {
    const out = fuseCosineGated([e('a'), e('b')], [e('b')]); // b boosted by lexical
    expect(out.every((x) => typeof x.score === 'number')).toBe(true);
    expect(out.map((x) => x.id)).toEqual(['b', 'a']);
  });
});
