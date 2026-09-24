import { describe, expect, it, vi } from 'vitest';

import { coalesceEmbedFn, coalesceEmbedFnWithStats, normalizeEmbeddingText } from './embed-coalesce';

it('normalizeEmbeddingText canonicalizes Unicode, case, and incidental whitespace', () => {
  expect(normalizeEmbeddingText('  Héllo\t WORLD  ')).toBe('héllo world');
  expect(normalizeEmbeddingText('ｅxample')).toBe('example');
});

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
  it('cancels one waiter without cancelling a sibling that still needs the vector', async () => {
    let upstream!: AbortSignal;
    let resolve!: (vector: number[]) => void;
    const fn = vi.fn((_text: string, signal?: AbortSignal) => {
      upstream = signal!;
      return new Promise<number[]>((res) => { resolve = res; });
    });
    const embed = coalesceEmbedFn(fn);
    const first = new AbortController();
    const cancelled = embed('q', first.signal);
    const observed = expect(cancelled).rejects.toThrow('obsolete');
    const sibling = embed('q');
    first.abort(new Error('obsolete'));
    await observed;
    expect(upstream.aborted).toBe(false);
    resolve([7]);
    await expect(sibling).resolves.toEqual([7]);
    expect(fn).toHaveBeenCalledOnce();
  });

  it('aborts upstream only when the last consumer leaves', async () => {
    let upstream!: AbortSignal;
    const embed = coalesceEmbedFn((_text, signal) => {
      upstream = signal!;
      return new Promise((_resolve, reject) => signal!.addEventListener('abort', () => reject(signal!.reason)));
    });
    const a = new AbortController();
    const b = new AbortController();
    const pa = embed('q', a.signal);
    const pb = embed('q', b.signal);
    const observed = Promise.allSettled([pa, pb]);
    a.abort();
    expect(upstream.aborted).toBe(false);
    b.abort();
    expect(upstream.aborted).toBe(true);
    expect((await observed).map((r) => r.status)).toEqual(['rejected', 'rejected']);
  });

  it('does not start work or return a cache hit for a pre-aborted consumer', async () => {
    const fn = vi.fn(async () => [1]);
    const embed = coalesceEmbedFn(fn);
    const controller = new AbortController();
    controller.abort(new Error('obsolete'));
    await expect(embed('q', controller.signal)).rejects.toThrow('obsolete');
    expect(fn).not.toHaveBeenCalled();
    await embed('q');
    await expect(embed('q', controller.signal)).rejects.toThrow('obsolete');
    expect(fn).toHaveBeenCalledOnce();
  });

  it('a late cancelled generation cannot cache a vector or remove its replacement', async () => {
    const releases: Array<(v: number[]) => void> = [];
    const fn = vi.fn(() => new Promise<number[]>((resolve) => { releases.push(resolve); }));
    const { embed, stats } = coalesceEmbedFnWithStats(fn);
    const controller = new AbortController();
    const old = embed('q', controller.signal);
    const observed = expect(old).rejects.toThrow('obsolete');
    controller.abort(new Error('obsolete'));
    await observed;
    const current = embed('q');
    releases[0]([1]); // Models a native implementation that cannot cancel.
    await Promise.resolve();
    await Promise.resolve();
    expect(stats.size()).toBe(0);
    const sibling = embed('q');
    expect(fn).toHaveBeenCalledTimes(2);
    releases[1]([2]);
    await expect(current).resolves.toEqual([2]);
    await expect(sibling).resolves.toEqual([2]);
    await expect(embed('q')).resolves.toEqual([2]);
  });

  it('removes abort listeners after success and failure', async () => {
    for (const fail of [false, true]) {
      const controller = new AbortController();
      const remove = vi.spyOn(controller.signal, 'removeEventListener');
      const embed = coalesceEmbedFn(async () => {
        if (fail) throw new Error('failed');
        return [1];
      });
      await Promise.allSettled([embed('q', controller.signal)]);
      expect(remove).toHaveBeenCalledWith('abort', expect.any(Function));
    }
  });

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

  it('coalesces equivalent casing/whitespace variants on one canonical identity', async () => {
    let calls = 0;
    const embed = coalesceEmbedFn(async (text) => {
      calls += 1;
      return [text.length];
    });
    await expect(embed('  Same\tQuery ')).resolves.toEqual([13]);
    await expect(embed('same query')).resolves.toEqual([13]);
    expect(calls).toBe(1);
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
