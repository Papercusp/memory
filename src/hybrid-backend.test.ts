import { describe, it, expect, vi } from 'vitest';
import { HybridBackend } from './hybrid-backend';
import type { MemoryBackend, MemoryEntry } from './backend';

const e = (id: string, score?: number): MemoryEntry => ({
  id,
  text: id,
  scope: 's',
  ...(score !== undefined ? { score } : {}),
});

function fakeBackend(name: string, searchResult: MemoryEntry[], over: Partial<MemoryBackend> = {}): MemoryBackend {
  return {
    name,
    available: async () => ({ ok: true }),
    remember: async () => ({ ids: [] }),
    search: async () => searchResult,
    list: async () => [],
    get: async () => null,
    forget: async () => {},
    update: async () => {},
    ...over,
  } as MemoryBackend;
}

describe('HybridBackend (P-020)', () => {
  it('fuses: an exact-id hit (in both legs) ranks above a paraphrase (cosine-only)', async () => {
    const cosine = fakeBackend('cosine', [e('para', 0.6), e('exact', 0.5)]);
    const lexical = fakeBackend('lexical', [e('exact', 9)]);
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.search('q', { scope: 's', limit: 6, minScore: 0.45 });
    expect(out.map((x) => x.id)).toEqual(['exact', 'para']);
  });

  it('returns empty when the cosine gate is empty (hard-negative — FP discipline)', async () => {
    const cosine = fakeBackend('cosine', []); // floored upstream
    const lexical = fakeBackend('lexical', [e('noise', 1)]); // lexical token-overlap noise
    const hy = new HybridBackend(lexical, cosine);
    expect(await hy.search('kubernetes', { scope: 's', minScore: 0.45 })).toEqual([]);
  });

  it('preserves a paraphrase (cosine finds it, lexical misses)', async () => {
    const cosine = fakeBackend('cosine', [e('para', 0.55)]);
    const lexical = fakeBackend('lexical', []);
    const hy = new HybridBackend(lexical, cosine);
    expect((await hy.search('reworded', { scope: 's' })).map((x) => x.id)).toEqual(['para']);
  });

  it('degrades to cosine-only when the lexical leg throws', async () => {
    const cosine = fakeBackend('cosine', [e('a', 0.6), e('b', 0.5)]);
    const lexical = fakeBackend('lexical', [], { search: async () => { throw new Error('claude dir missing'); } });
    const hy = new HybridBackend(lexical, cosine);
    expect((await hy.search('q', { scope: 's' })).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('writes the canonical id from the cosine leg; forget/update target it', async () => {
    const remember = vi.fn(async () => ({ ids: ['1'] }));
    const forget = vi.fn(async () => {});
    const cosine = fakeBackend('cosine', [], { remember, forget });
    const hy = new HybridBackend(fakeBackend('lexical', []), cosine);
    const out = await hy.remember('fact', { scope: 's' });
    await hy.forget('1');
    expect(remember).toHaveBeenCalledOnce();
    expect(out.ids).toEqual(['1']); // canonical (cosine) ids are returned
    expect(forget).toHaveBeenCalledWith('1');
  });

  it('write-throughs the projection into the lexical leg (best-effort), still returns cosine ids', async () => {
    const cosineRemember = vi.fn(async () => ({ ids: ['c1'] }));
    const lexicalRemember = vi.fn(async () => ({ ids: ['l1'] }));
    const cosine = fakeBackend('cosine', [], { remember: cosineRemember });
    const lexical = fakeBackend('lexical', [], { remember: lexicalRemember });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.remember('the marker token kestrel-7', { scope: 's' });
    expect(cosineRemember).toHaveBeenCalledOnce();
    expect(lexicalRemember).toHaveBeenCalledOnce(); // projected to the native surface
    expect(out.ids).toEqual(['c1']); // canonical store owns ids
  });

  it('a throwing lexical projection is non-fatal — the canonical write still succeeds', async () => {
    const cosine = fakeBackend('cosine', [], { remember: async () => ({ ids: ['c1'] }) });
    const lexical = fakeBackend('lexical', [], { remember: async () => { throw new Error('no ~/.claude dir'); } });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.remember('fact', { scope: 's' });
    expect(out.ids).toEqual(['c1']);
  });

  it('passes the FP floor through to the cosine leg', async () => {
    const search = vi.fn(async () => [e('a', 0.6)]);
    const cosine = fakeBackend('cosine', [], { search });
    const hy = new HybridBackend(fakeBackend('lexical', []), cosine);
    await hy.search('q', { scope: 's', limit: 6, minScore: 0.42 });
    expect(search.mock.calls[0]?.[1]).toMatchObject({ minScore: 0.42 });
  });

  it('cross-backend by construction: a remember via one client is recallable via another (P-022 / D-002)', async () => {
    // One shared canonical (cosine) store; two HybridBackends over it model two
    // different MCP clients (e.g. claude-su + codex-su) hitting the same operator.
    const store: MemoryEntry[] = [];
    const canonical: MemoryBackend = {
      name: 'canonical',
      available: async () => ({ ok: true }),
      remember: async (text, opts) => {
        const id = `c${store.length}`;
        store.push({ id, text, scope: opts.scope as string, score: 0.7 });
        return { ids: [id] };
      },
      search: async (q) => store.filter((e) => e.text.includes(q)),
      list: async () => store,
      get: async (id) => store.find((e) => e.id === id) ?? null,
      forget: async () => {},
      update: async () => {},
    };
    const clientA = new HybridBackend(fakeBackend('lexical', []), canonical);
    const clientB = new HybridBackend(fakeBackend('lexical', []), canonical);

    await clientA.remember('the user prefers nuqs', { scope: 's' });
    // Client B (a different client instance) recalls A's write from the one store.
    const recalled = await clientB.search('nuqs', { scope: 's', minScore: 0.45 });
    expect(recalled.map((r) => r.text)).toEqual(['the user prefers nuqs']);
  });
});
