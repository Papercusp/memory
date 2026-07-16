import { describe, expect, it } from 'vitest';

import { coalesceEmbedFn, coalesceEmbedFnWithStats } from './embed-coalesce';

/** A controllable embedder: counts calls, resolves on demand. */
function slowEmbedder(vector: number[] = [1, 2, 3]) {
  let calls = 0;
  const releases: Array<() => void> = [];
  const fn = (_text: string): Promise<number[]> => {
    calls += 1;
    return new Promise<number[]>((resolve) => {
      releases.push(() => resolve(vector));
    });
  };
  return {
    fn,
    calls: () => calls,
    releaseAll: () => {
      for (const r of releases.splice(0)) r();
    },
  };
}

describe('coalesceEmbedFn', () => {
  it('coalesces concurrent same-text calls onto ONE underlying embed (the injection 3-leg case)', async () => {
    const under = slowEmbedder([7, 8]);
    const { embed, stats } = coalesceEmbedFnWithStats(under.fn);

    const p1 = embed('same query');
    const p2 = embed('same query');
    const p3 = embed('same query');
    expect(under.calls()).toBe(1);

    under.releaseAll();
    const [v1, v2, v3] = await Promise.all([p1, p2, p3]);
    expect(v1).toEqual([7, 8]);
    expect(v2).toEqual([7, 8]);
    expect(v3).toEqual([7, 8]);
    expect(stats.misses).toBe(1);
    expect(stats.coalesced).toBe(2);
  });

  it('does NOT coalesce different texts', async () => {
    const under = slowEmbedder();
    const embed = coalesceEmbedFn(under.fn);
    void embed('alpha');
    void embed('beta');
    expect(under.calls()).toBe(2);
    under.releaseAll();
  });

  it('serves a fresh cache hit without re-embedding, and expires it past ttlMs', async () => {
    let clock = 1_000;
    const under = slowEmbedder([4]);
    const { embed, stats } = coalesceEmbedFnWithStats(under.fn, {
      ttlMs: 100,
      now: () => clock,
    });

    const p = embed('q');
    under.releaseAll();
    await expect(p).resolves.toEqual([4]);

    clock += 50; // fresh
    await expect(embed('q')).resolves.toEqual([4]);
    expect(under.calls()).toBe(1);
    expect(stats.hits).toBe(1);

    clock += 100; // expired
    const p2 = embed('q');
    expect(under.calls()).toBe(2);
    under.releaseAll();
    await expect(p2).resolves.toEqual([4]);
  });

  it('propagates a rejection to every coalesced waiter and never caches it', async () => {
    let calls = 0;
    let reject!: (err: Error) => void;
    let resolve!: (v: number[]) => void;
    const fn = (): Promise<number[]> => {
      calls += 1;
      return new Promise<number[]>((res, rej) => {
        resolve = res;
        reject = rej;
      });
    };
    const embed = coalesceEmbedFn(fn);

    const p1 = embed('q');
    const p2 = embed('q');
    reject(new Error('embedder down'));
    await expect(p1).rejects.toThrow('embedder down');
    await expect(p2).rejects.toThrow('embedder down');

    // Next call retries the real embedder (the failure did not stick).
    const p3 = embed('q');
    expect(calls).toBe(2);
    resolve([9]);
    await expect(p3).resolves.toEqual([9]);
  });

  it('does not cache an empty vector', async () => {
    let calls = 0;
    const embed = coalesceEmbedFn(async () => {
      calls += 1;
      return [];
    });
    await expect(embed('q')).resolves.toEqual([]);
    await expect(embed('q')).resolves.toEqual([]);
    expect(calls).toBe(2);
  });

  it('evicts the least-recently-used entry past maxEntries', async () => {
    let calls = 0;
    const embed = coalesceEmbedFn(
      async (text: string) => {
        calls += 1;
        return [text.length];
      },
      { maxEntries: 2 },
    );
    await embed('a'); // cache: a
    await embed('bb'); // cache: a, bb
    await embed('a'); // hit — refreshes a's recency (cache: bb, a)
    await embed('ccc'); // evicts bb (cache: a, ccc)
    expect(calls).toBe(3);
    await embed('a'); // still cached
    expect(calls).toBe(3);
    await embed('bb'); // was evicted — re-embeds
    expect(calls).toBe(4);
  });
});
