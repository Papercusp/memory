import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  createTextSimilarity,
  diversityDisabledByEnv,
  diversityRerank,
  lexicalSimilarity,
  textSimilarity,
} from './diversity-rerank';
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

describe('createTextSimilarity (memoized — F-D)', () => {
  const m = (id: string, text: string): MemoryEntry => ({ id, text, scope: 's' });

  it('agrees with the unmemoized textSimilarity on every pair', () => {
    const xs = [
      m('a', 'the deploy pipeline uses a two port model'),
      m('b', 'the deploy pipeline uses a two-port model'),
      m('c', 'git-sync owns commit and push'),
      m('d', ''),
    ];
    const sim = createTextSimilarity();
    for (const x of xs) {
      for (const y of xs) {
        expect(sim(x, y)).toBeCloseTo(textSimilarity(x, y), 10);
      }
    }
  });

  it('produces the SAME ordering as the unmemoized proxy — memoization is perf-only', () => {
    const xs = [
      { ...m('best', 'the deploy pipeline uses a two port model for staging and release'), score: 0.95 },
      { ...m('dup', 'the deploy pipeline uses a two-port model for staging vs release'), score: 0.94 },
      { ...m('distinct', 'git-sync owns commit and push for the shared tree'), score: 0.6 },
    ];
    const plain = diversityRerank(xs, { lambda: 0.5, similarity: textSimilarity });
    const memo = diversityRerank(xs, { lambda: 0.5, similarity: createTextSimilarity() });
    expect(memo.map((x) => x.id)).toEqual(plain.map((x) => x.id));
  });

  it('computes each entry trigram set ONCE — the O(n²) pass must not rebuild them', () => {
    // Guards the reason this exists: on the injection critical path the naive
    // adapter rebuilds both operands' trigram sets on every pairwise call.
    const xs = Array.from({ length: 8 }, (_, i) => ({
      ...m(`e${i}`, `a memory about subsystem ${i} and how it behaves under load`),
      score: 1 - i / 10,
    }));
    let textReads = 0;
    const counted = xs.map(
      (x) =>
        ({
          ...x,
          get text() {
            textReads++;
            return `a memory about subsystem ${x.id} and how it behaves under load`;
          },
        }) as MemoryEntry,
    );
    diversityRerank(counted, { lambda: 0.5, similarity: createTextSimilarity() });
    expect(textReads).toBeLessThanOrEqual(counted.length);
  });
});

describe('diversityDisabledByEnv', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('is off unless PAPERCUSP_MEMORY_MMR is exactly "0"', () => {
    expect(diversityDisabledByEnv()).toBe(false);
    vi.stubEnv('PAPERCUSP_MEMORY_MMR', '1');
    expect(diversityDisabledByEnv()).toBe(false);
    vi.stubEnv('PAPERCUSP_MEMORY_MMR', '0');
    expect(diversityDisabledByEnv()).toBe(true);
  });
});
