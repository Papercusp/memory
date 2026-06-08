import { describe, it, expect } from 'vitest';
import { fuse, fuseCosineGated } from './hybrid-fusion';
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

describe('fuse — floored-union (P-031)', () => {
  it('admits a strong lexical-ONLY hit (exact-id the cosine leg missed)', () => {
    // cosine has only a paraphrase; lexical has the exact-id target (score 1.0).
    const out = fuse([e('para', 0.55)], [e('exact', 1.0)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id).sort()).toEqual(['exact', 'para']);
  });

  it('does NOT admit a weak lexical-only hit (below minLexScore)', () => {
    const out = fuse([e('para', 0.55)], [e('weak', 0.3)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id)).toEqual(['para']);
  });

  it('hard-negative: empty cosine + only weak lexical overlap → empty', () => {
    const out = fuse([], [e('weak', 0.2)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out).toEqual([]);
  });

  it('a hit in BOTH legs outranks a lexical-only and a cosine-only hit', () => {
    // both: cosRank + lexRank; cosine-only: cosRank; lexical-only(strong): lexRank.
    const out = fuse([e('both', 0.6), e('cosOnly', 0.5)], [e('both', 0.9), e('lexOnly', 0.8)], {
      mode: 'floored-union',
      minLexScore: 0.5,
    });
    expect(out[0].id).toBe('both');
    expect(out.map((x) => x.id).sort()).toEqual(['both', 'cosOnly', 'lexOnly']);
  });

  it('floored-union is the default mode', () => {
    const out = fuse([e('para', 0.55)], [e('exact', 1.0)]); // no mode → floored-union
    expect(out.map((x) => x.id).sort()).toEqual(['exact', 'para']);
  });
});
