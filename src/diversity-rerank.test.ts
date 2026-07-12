import { describe, it, expect } from 'vitest';
import { diversityRerank, lexicalSimilarity, textSimilarity } from './diversity-rerank';
import type { MemoryEntry } from './backend';

const e = (id: string, text: string, score?: number): MemoryEntry => ({
  id,
  text,
  scope: 's',
  ...(score !== undefined ? { score } : {}),
});

describe('diversityRerank', () => {
  it('lambda=1 (default) is an identity no-op — same order, same members', () => {
    const xs = [e('a', 'apple pie recipe', 0.9), e('b', 'banana bread recipe', 0.5), e('c', 'car engine repair', 0.3)];
    const out = diversityRerank(xs, { similarity: textSimilarity });
    expect(out.map((x) => x.id)).toEqual(['a', 'b', 'c']);
    expect(out).not.toBe(xs); // new array
  });

  it('lambda=1 explicit behaves the same as default', () => {
    const xs = [e('a', 'x', 0.9), e('b', 'y', 0.5)];
    expect(diversityRerank(xs, { lambda: 1, similarity: textSimilarity }).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('never drops or adds entries — same length and members regardless of lambda', () => {
    const xs = [e('a', 'the quick brown fox', 0.9), e('b', 'the quick brown fox jumps', 0.85), e('c', 'totally unrelated topic', 0.2)];
    for (const lambda of [0, 0.3, 0.5, 0.7, 1]) {
      const out = diversityRerank(xs, { lambda, similarity: textSimilarity });
      expect(out).toHaveLength(xs.length);
      expect(out.map((x) => x.id).sort()).toEqual(['a', 'b', 'c']);
    }
  });

  it('demotes a near-duplicate below a lower-scored but distinct hit (the redundancy fix)', () => {
    // a and b are near-verbatim paraphrases (high lexical overlap); c is distinct.
    const xs = [
      e('a', 'the deploy pipeline uses a two port model for staging and release', 0.95),
      e('b', 'the deploy pipeline uses a two-port model for staging vs release', 0.94),
      e('c', 'git-sync owns commit and push for the shared tree', 0.6),
    ];
    // Pure relevance (lambda=1) would keep a,b,c — b's near-dup of a wastes the
    // #2 slot. At a diversity-favoring lambda, c should out-rank b for slot 2.
    const out = diversityRerank(xs, { lambda: 0.5, similarity: textSimilarity });
    expect(out[0].id).toBe('a'); // best relevance always picked first
    expect(out[1].id).toBe('c'); // distinct fact beats the near-duplicate for #2
    expect(out.map((x) => x.id).sort()).toEqual(['a', 'b', 'c']);
  });

  it('lambda=0 (pure diversity) still picks the top-scored entry first (empty selected set ⇒ maxSim=0)', () => {
    const xs = [e('a', 'alpha', 0.9), e('b', 'beta', 0.1)];
    const out = diversityRerank(xs, { lambda: 0, similarity: textSimilarity });
    expect(out[0].id).toBe('a');
  });

  it('single-entry and empty inputs are no-ops', () => {
    expect(diversityRerank([], { similarity: textSimilarity })).toEqual([]);
    const one = [e('a', 'solo', 0.5)];
    expect(diversityRerank(one, { lambda: 0, similarity: textSimilarity }).map((x) => x.id)).toEqual(['a']);
  });

  it('unscored entries default to 0 relevance but can still be selected', () => {
    const xs = [e('a', 'apple', 0.9), e('b', 'zebra')]; // b has no score
    const out = diversityRerank(xs, { lambda: 1, similarity: textSimilarity });
    expect(out.map((x) => x.id).sort()).toEqual(['a', 'b']);
  });

  it('does not mutate the input array', () => {
    const xs = [e('a', 'x', 0.9), e('b', 'y', 0.1)];
    const copy = [...xs];
    diversityRerank(xs, { lambda: 0.5, similarity: textSimilarity });
    expect(xs).toEqual(copy);
  });

  it('clamps an out-of-range lambda instead of throwing', () => {
    const xs = [e('a', 'x', 0.9), e('b', 'y', 0.1)];
    expect(() => diversityRerank(xs, { lambda: 5, similarity: textSimilarity })).not.toThrow();
    expect(() => diversityRerank(xs, { lambda: -5, similarity: textSimilarity })).not.toThrow();
  });
});

describe('lexicalSimilarity', () => {
  it('is 1 for identical strings', () => {
    expect(lexicalSimilarity('same text here', 'same text here')).toBe(1);
  });

  it('is near 0 for unrelated strings', () => {
    expect(lexicalSimilarity('quantum physics research', 'banana bread recipe')).toBeLessThan(0.15);
  });

  it('is high for near-verbatim paraphrases', () => {
    const sim = lexicalSimilarity(
      'the deploy pipeline uses a two port model for staging and release',
      'the deploy pipeline uses a two-port model for staging vs release',
    );
    expect(sim).toBeGreaterThan(0.6);
  });

  it('handles empty strings without throwing', () => {
    expect(lexicalSimilarity('', '')).toBe(1);
    expect(lexicalSimilarity('', 'x')).toBe(0);
  });

  it('is symmetric', () => {
    const a = 'foo bar baz';
    const b = 'bar baz qux';
    expect(lexicalSimilarity(a, b)).toBeCloseTo(lexicalSimilarity(b, a), 10);
  });
});
