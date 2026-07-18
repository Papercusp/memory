import { describe, it, expect, vi } from 'vitest';
import { HybridBackend } from './hybrid-backend';
import { NoopBackend, NOOP_DISABLED_REASON } from './noop-backend';
import { MemoryUnavailableError } from './backend';
import type { MemoryBackend, MemoryEntry, RememberOptions, SearchOptions } from './backend';

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

  it('hard-negative: cosine floored empty + weak lexical overlap → empty (dual gate, floored-union default)', async () => {
    const cosine = fakeBackend('cosine', []); // floored upstream
    // Realistic hard-negative lexical: generic token overlap scores BELOW the
    // identifier bar (0.5), so the union does not admit it.
    const lexical = fakeBackend('lexical', [e('weak-overlap', 0.2)]);
    const hy = new HybridBackend(lexical, cosine);
    expect(await hy.search('kubernetes', { scope: 's', minScore: 0.45 })).toEqual([]);
  });

  it('floored-union: a STRONG lexical-only identifier hit IS admitted (captures exact-id the cosine leg missed)', async () => {
    // The cosine leg missed the exact-id target entirely (only a paraphrase);
    // the lexical leg matched it on the identifier token (score ≥ 0.5) → admitted.
    const cosine = fakeBackend('cosine', [e('para', 0.55)]);
    const lexical = fakeBackend('lexical', [e('CODEX_HOME', 1.0)]);
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.search('CODEX_HOME rotation', { scope: 's', limit: 6, minScore: 0.45 });
    expect(out.map((x) => x.id).sort()).toEqual(['CODEX_HOME', 'para']);
  });

  it('floored-union: a WEAK lexical-only hit (below minLexScore) is NOT admitted', async () => {
    const cosine = fakeBackend('cosine', [e('para', 0.55)]);
    const lexical = fakeBackend('lexical', [e('weak', 0.3)]);
    const hy = new HybridBackend(lexical, cosine);
    // Per-call minLexScore override (0.5) — 0.3 is below it → excluded.
    const out = await hy.search('q', { scope: 's', minScore: 0.45, minLexScore: 0.5 });
    expect(out.map((x) => x.id)).toEqual(['para']); // weak lexical-only excluded
  });

  it('per-call minLexScore override beats the constructor/default bar', async () => {
    const cosine = fakeBackend('cosine', [e('para', 0.55)]);
    const lexical = fakeBackend('lexical', [e('mid', 0.35)]);
    // Default bar 0.30 would admit 0.35; a per-call 0.50 override rejects it.
    const hy = new HybridBackend(lexical, cosine);
    expect((await hy.search('q', { scope: 's' })).map((x) => x.id).sort()).toEqual(['mid', 'para']);
    expect((await hy.search('q', { scope: 's', minLexScore: 0.5 })).map((x) => x.id)).toEqual(['para']);
  });

  it('cosine-gated mode: a lexical-only hit is NEVER admitted, even when strong', async () => {
    const cosine = fakeBackend('cosine', []); // floored empty
    const lexical = fakeBackend('lexical', [e('CODEX_HOME', 1.0)]);
    const hy = new HybridBackend(lexical, cosine, { fusionMode: 'cosine-gated' });
    expect(await hy.search('CODEX_HOME', { scope: 's', minScore: 0.45 })).toEqual([]);
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

  // EI-2777: the lexical-leg call must survive a SYNCHRONOUS throw, not just an async
  // rejection. The async-reject case (above) passed even before the fix because a
  // trailing `.catch()` handles a rejected promise — but a sync throw from the method
  // ACCESS (an undefined/misconfigured leg, or a leg missing `.search`) escapes `.catch()`
  // and surfaced as the deterministic, search-only "reading 'search'" tool error (while
  // remember()'s try/catch swallowed the identical fault). These two would FAIL on the
  // old `.catch()` code and pass on the try/catch.
  it('degrades to cosine-only when the lexical leg is undefined (EI-2777)', async () => {
    const cosine = fakeBackend('cosine', [e('a', 0.6), e('b', 0.5)]);
    const hy = new HybridBackend(undefined as unknown as MemoryBackend, cosine);
    expect((await hy.search('q', { scope: 's' })).map((x) => x.id)).toEqual(['a', 'b']);
  });

  it('degrades to cosine-only when the lexical leg.search throws synchronously (EI-2777)', async () => {
    const cosine = fakeBackend('cosine', [e('a', 0.6), e('b', 0.5)]);
    const lexical = fakeBackend('lexical', [], {
      search: (() => { throw new Error('sync boom — not a rejected promise'); }) as never,
    });
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

  it('write-through stamps link_id = canonical id onto the lexical projection (for cross-leg dedup)', async () => {
    const cosine = fakeBackend('cosine', [], { remember: async () => ({ ids: ['canon-9'] }) });
    const lexRemember = vi.fn(async (_text: string, _opts: RememberOptions) => ({ ids: ['l'] }));
    const lexical = fakeBackend('lexical', [], { remember: lexRemember });
    const hy = new HybridBackend(lexical, cosine);
    await hy.remember('a fact', { scope: 's', metadata: { kind: 'x' } });
    // lexical write carries the original metadata PLUS link_id = the canonical id.
    expect(lexRemember.mock.calls[0]?.[1]?.metadata).toMatchObject({ kind: 'x', link_id: 'canon-9' });
  });

  it('passes the FP floor through to the cosine leg', async () => {
    const search = vi.fn(async (_q: string, _opts: SearchOptions) => [e('a', 0.6)]);
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

  // ── GAP 6: available() is a PROBE, not a writability guarantee ──────────────
  // available() forwards to cosine.available() (hybrid-backend.ts:69). A green
  // probe is NOT a promise that remember succeeds: the cosine leg can report
  // available={ok:true} (e.g. PG reachable, schema present) yet its embedder is
  // dead so remember THROWS. This pins the GAP-1 root at the backend layer — the
  // documented contract that available() does not gate writes. (No defect: the
  // probe is non-throwing by design; callers must still handle a write throw.)
  it('available() GREEN does NOT guarantee remember succeeds (false-green probe)', async () => {
    const cosine = fakeBackend('cosine', [], {
      available: async () => ({ ok: true }),
      remember: async () => {
        throw new MemoryUnavailableError('embedder_dead');
      },
    });
    const hy = new HybridBackend(fakeBackend('lexical', []), cosine);
    // The probe reports GREEN...
    expect(await hy.available()).toEqual({ ok: true });
    // ...yet the very next write THROWS — green probe ≠ writable backend.
    await expect(hy.remember('fact', { scope: 's' })).rejects.toBeInstanceOf(
      MemoryUnavailableError,
    );
  });

  it('available() forwards the cosine leg verbatim (ok:false + reason passes through)', async () => {
    const cosine = fakeBackend('cosine', [], {
      available: async () => ({ ok: false, reason: 'pg_unreachable' }),
    });
    const hy = new HybridBackend(fakeBackend('lexical', []), cosine);
    // available() tracks the cosine leg (the write target / source of truth);
    // the lexical leg's health is irrelevant to the probe.
    expect(await hy.available()).toEqual({ ok: false, reason: 'pg_unreachable' });
  });

  // ── GAP 12: NoopBackend as the cosine leg ──────────────────────────────────
  // A hybrid whose cosine leg is the deliberate "no store": available is
  // {ok:false}, search returns [] (NOT a throw — the dedup/search path never
  // probes available, so the Noop's empty search degrades cleanly), but remember
  // THROWS MemoryUnavailableError (no caller is lied to about a fact storing).
  describe('NoopBackend as the cosine leg (memory OFF)', () => {
    it('available() reports {ok:false} with the disabled reason', async () => {
      const hy = new HybridBackend(fakeBackend('lexical', []), new NoopBackend());
      expect(await hy.available()).toEqual({ ok: false, reason: NOOP_DISABLED_REASON });
    });

    it('search returns [] (does NOT throw — bypasses the availability probe)', async () => {
      const hy = new HybridBackend(fakeBackend('lexical', []), new NoopBackend());
      // floored-union default: cosine is [] but NOT cosine-gated, so the lexical
      // leg still runs; with an empty lexical leg the fused result is [].
      await expect(hy.search('anything', { scope: 's', minScore: 0.45 })).resolves.toEqual([]);
    });

    it('remember THROWS MemoryUnavailableError — the write is NOT silently dropped', async () => {
      const hy = new HybridBackend(fakeBackend('lexical', []), new NoopBackend());
      await expect(hy.remember('fact', { scope: 's' })).rejects.toBeInstanceOf(
        MemoryUnavailableError,
      );
    });
  });

  // ── GAP 13: per-call fusionMode override + cosine-gated early-return ────────
  // A hybrid constructed floored-union, then CALLED with {fusionMode:'cosine-gated'}:
  // the per-call override beats the constructor default (hybrid-backend.ts:112),
  // and with an empty cosine set the cosine-gated branch early-returns []
  // (hybrid-backend.ts:118) BEFORE the lexical leg can admit anything — so a
  // strong lexical-only identifier hit is NOT admitted (override wins).
  it('per-call fusionMode:cosine-gated beats a floored-union constructor + floors a strong lexical-only hit to []', async () => {
    const cosine = fakeBackend('cosine', []); // floored empty
    const lexical = fakeBackend('lexical', [e('CODEX_HOME', 1.0)]); // strong identifier hit
    // Constructor says floored-union (which WOULD admit the strong lexical hit)...
    const hy = new HybridBackend(lexical, cosine, { fusionMode: 'floored-union' });
    // ...but the per-call override flips to cosine-gated → empty cosine early-returns [].
    expect(
      await hy.search('CODEX_HOME', { scope: 's', minScore: 0.45, fusionMode: 'cosine-gated' }),
    ).toEqual([]);
    // Control: with NO override the constructor's floored-union DOES admit it.
    expect(
      (await hy.search('CODEX_HOME', { scope: 's', minScore: 0.45 })).map((x) => x.id),
    ).toEqual(['CODEX_HOME']);
  });

  it('cosine-gated early-return short-circuits the lexical leg entirely (it is never searched)', async () => {
    const cosine = fakeBackend('cosine', []); // floored empty
    const lexSearch = vi.fn(async () => [e('CODEX_HOME', 1.0)]);
    const lexical = fakeBackend('lexical', [], { search: lexSearch });
    const hy = new HybridBackend(lexical, cosine, { fusionMode: 'cosine-gated' });
    expect(await hy.search('CODEX_HOME', { scope: 's', minScore: 0.45 })).toEqual([]);
    // The early-return fires BEFORE the lexical search — no wasted lexical probe.
    expect(lexSearch).not.toHaveBeenCalled();
  });

  /**
   * Recurrence guard: in floored-union mode the two legs MUST run CONCURRENTLY.
   *
   * They are independent (two rankings of the same rows — neither's input depends on the
   * other's output), so awaiting them in sequence puts the lexical leg's whole cost on the
   * critical path: total = cosine + lexical instead of max(cosine, lexical). That was free
   * to ignore while the lexical leg was a local file scan, and became the DOMINANT cost the
   * moment the leg became a PG query — most of why `hybrid-pg` benched at p50 1182ms against
   * `hybrid`'s 821ms (memory-pg-lexical-own-injection-2026-07-13 P-006 run 8).
   *
   * A future refactor that innocently re-writes this back to `await cosine; await lexical;`
   * would silently restore the regression with every test still green — so pin the overlap
   * itself: the lexical leg must be STARTED before the cosine leg has resolved.
   */
  it('floored-union runs BOTH legs concurrently (the lexical leg is not on the critical path)', async () => {
    let cosineResolved = false;
    let lexicalStartedBeforeCosineResolved = false;

    const cosine = fakeBackend('cosine', [], {
      search: vi.fn(async () => {
        // Yield to the microtask queue so a concurrently-started lexical leg can run.
        await new Promise((r) => setTimeout(r, 10));
        cosineResolved = true;
        return [e('cosine-hit', 0.9)];
      }),
    });
    const lexical = fakeBackend('lexical', [], {
      search: vi.fn(async () => {
        lexicalStartedBeforeCosineResolved = !cosineResolved;
        return [e('CODEX_HOME', 1.0)];
      }),
    });

    const hy = new HybridBackend(lexical, cosine, { fusionMode: 'floored-union' });
    const hits = await hy.search('CODEX_HOME', { scope: 's' });

    // The overlap itself — the thing a re-serializing refactor would break.
    expect(lexicalStartedBeforeCosineResolved).toBe(true);
    // ...and both legs still actually contribute to the fused result.
    expect(hits.map((h) => h.text)).toEqual(expect.arrayContaining(['CODEX_HOME', 'cosine-hit']));
  });
});

describe('HybridBackend.embedQuery (EI-12992 shared-embed delegation)', () => {
  it('exposes the COSINE leg capability and delegates to it — never the lexical leg', async () => {
    const cosineEmbed = vi.fn(async () => [0.1, 0.2]);
    const lexEmbed = vi.fn(async () => [9, 9]);
    const hy = new HybridBackend(
      fakeBackend('lexical', [], { embedQuery: lexEmbed }),
      fakeBackend('cosine', [], { embedQuery: cosineEmbed }),
    );
    expect(await hy.embedQuery?.('q')).toEqual([0.1, 0.2]);
    expect(cosineEmbed).toHaveBeenCalledWith('q');
    // The vector MUST come from the leg that CONSUMES it: search() forwards
    // opts.vector to the cosine leg, so a lexical-leg vector would be compared
    // across embedding spaces and return plausible garbage instead of failing.
    expect(lexEmbed).not.toHaveBeenCalled();
  });

  it('is UNDEFINED when the cosine leg cannot embed — the capability feature-test stays honest', () => {
    const hy = new HybridBackend(fakeBackend('lexical', []), fakeBackend('cosine', []));
    expect(hy.embedQuery).toBeUndefined();
  });

  it('preserves the cosine leg\'s `this` binding (delegation is bound, not a bare fn ref)', async () => {
    const cosine = fakeBackend('cosine', []) as MemoryBackend & { model?: string };
    cosine.model = 'bge-small';
    cosine.embedQuery = async function (this: { model?: string }) {
      return this.model === 'bge-small' ? [1] : null;
    };
    const hy = new HybridBackend(fakeBackend('lexical', []), cosine);
    expect(await hy.embedQuery?.('q')).toEqual([1]);
  });

  it('forwards a precomputed vector to the cosine leg, and never leaks it to the lexical leg', async () => {
    const cosineSearch = vi.fn(async (_q: string, _o: SearchOptions) => [e('a', 0.6)]);
    const lexSearch = vi.fn(async (_q: string, _o: SearchOptions) => []);
    const hy = new HybridBackend(
      fakeBackend('lexical', [], { search: lexSearch }),
      fakeBackend('cosine', [], { search: cosineSearch }),
    );
    await hy.search('q', { scope: 's', limit: 6, minScore: 0.45, vector: [0.3, 0.4] });
    expect(cosineSearch.mock.calls[0]?.[1]).toMatchObject({ vector: [0.3, 0.4] });
    // The lexical leg is embed-free by construction — it gets scope+limit only.
    expect(lexSearch.mock.calls[0]?.[1]).not.toHaveProperty('vector');
  });

  it('REGRESSION (production shape): 3 pulls over one query embed ONCE, not 3×', async () => {
    // Mirrors buildMemoryContextBlock's fan-out (user/harness/hive pools over the
    // SAME query text) against a cosine leg that embeds per-search unless handed a
    // vector — exactly Mem0Backend's `opts.vector ?? await this.embedQuery(query)`.
    // Before this delegation the wrapper hid embedQuery, so `hybrid-pg` — the
    // PRODUCTION backend — silently paid all 3 embeds and the fix was a no-op.
    let embeds = 0;
    const cosine = fakeBackend('cosine', [], {
      embedQuery: async () => {
        embeds += 1;
        return [0.5];
      },
      search: async (_q: string, o: SearchOptions) => {
        if (!o.vector) embeds += 1; // the leg had to embed for itself
        return [e('a', 0.6)];
      },
    });
    const hy = new HybridBackend(fakeBackend('lexical', []), cosine);

    // Deliberately NO assertion on `shared` here: the embed COUNT below must be
    // what discriminates, so a regression fails with "expected 3 to be 1" (the
    // actual defect) rather than tripping an earlier assertion that never
    // exercises the fan-out. The `?.` + conditional spread mirror injection.ts's
    // real feature-test/fallback, so this stays a faithful simulation on both sides.
    const shared = await hy.embedQuery?.('the same query text');
    for (const scope of ['user', 'harness:papercusp', 'hive:pc']) {
      await hy.search('the same query text', {
        scope,
        limit: 3,
        minScore: 0.45,
        ...(shared ? { vector: shared } : {}),
      });
    }
    expect(embeds).toBe(1);
  });
});

describe('HybridBackend searchLexical (WI-4214 embed-free fallback)', () => {
  it('prefers the cosine leg\'s lexical capability (canonical store covers ALL memories)', async () => {
    const cosineLex = vi.fn(async () => [e('canonical-hit', 0.5)]);
    const cosine = fakeBackend('cosine', [], { searchLexical: cosineLex } as Partial<MemoryBackend>);
    const lexSearch = vi.fn(async () => [e('projected-hit', 1)]);
    const lexical = fakeBackend('lexical', [], { search: lexSearch });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.searchLexical('q', { scope: 's', limit: 3 });
    expect(out.map((x) => x.id)).toEqual(['canonical-hit']);
    expect(cosineLex).toHaveBeenCalledWith('q', { scope: 's', limit: 3 });
    expect(lexSearch).not.toHaveBeenCalled();
  });

  it('falls back to the native lexical leg when the cosine leg lacks the capability', async () => {
    const cosine = fakeBackend('cosine', []);
    const lexical = fakeBackend('lexical', [e('projected-hit', 1)]);
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.searchLexical('q', { scope: 's', limit: 3 });
    expect(out.map((x) => x.id)).toEqual(['projected-hit']);
  });

  it('falls back to the native lexical leg when the cosine lexical THROWS (degraded path stays serving)', async () => {
    const cosine = fakeBackend('cosine', [], {
      searchLexical: vi.fn(async () => {
        throw new Error('pg down');
      }),
    } as Partial<MemoryBackend>);
    const lexical = fakeBackend('lexical', [e('projected-hit', 1)]);
    const hy = new HybridBackend(lexical, cosine);
    expect((await hy.searchLexical('q', { scope: 's' })).map((x) => x.id)).toEqual(['projected-hit']);
  });
});

describe('HybridBackend.invalidateEntry (temporal-lite lifecycle delegation)', () => {
  it('delegates to the COSINE leg (the canonical store owns lifecycle) with the supersededBy option', async () => {
    const inv = vi.fn(async () => true);
    const cosine = fakeBackend('cosine', [], { invalidateEntry: inv });
    const hybrid = new HybridBackend(fakeBackend('lex', []), cosine);
    await expect(hybrid.invalidateEntry('old', { supersededBy: 'new' })).resolves.toBe(true);
    expect(inv).toHaveBeenCalledWith('old', { supersededBy: 'new' });
  });

  it('a cosine leg WITHOUT the capability throws — a false would read as "not found" upstream', () => {
    const hybrid = new HybridBackend(fakeBackend('lex', []), fakeBackend('cosine', []));
    expect(() => hybrid.invalidateEntry('old')).toThrow(/validity-window/);
  });
});
