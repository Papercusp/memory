import { describe, it, expect, vi, afterEach } from 'vitest';
import { Mem0Backend, extractAddedIds, extractStoredEventCount } from './mem0-backend';
import { MemoryUnavailableError } from './backend';
import type { MemoryClient } from './mem0-client';

type Row = { id: string; memory: string; score: number };

const UUID_A = '11111111-1111-1111-1111-111111111111';
const UUID_B = '22222222-2222-2222-2222-222222222222';
const UUID_C = '33333333-3333-3333-3333-333333333333';

/** Minimal MemoryClient stub: every method is a vi.fn; override the ones a
 *  given test exercises. Defaults make unrelated methods safe no-ops. */
function stubClient(over: Partial<Record<keyof MemoryClient, unknown>> = {}): MemoryClient {
  return {
    add: vi.fn(async () => ({ results: [] })),
    delete: vi.fn(async () => ({})),
    get: vi.fn(async () => null),
    getAll: vi.fn(async () => ({ results: [] })),
    search: vi.fn(async () => ({ results: [] })),
    update: vi.fn(async () => ({})),
    ...over,
  } as unknown as MemoryClient;
}

/** A fake mem0 client that returns canned rows per scope and records every
 *  search's options (so we can assert the scope filter). */
function fakeClient(rowsByScope: Record<string, Row[]>, calls: Record<string, unknown>[]): MemoryClient {
  return {
    add: vi.fn(),
    get: vi.fn(),
    getAll: vi.fn(async () => ({ results: [] })),
    update: vi.fn(),
    search: vi.fn(async (_query: string, opts: Record<string, unknown>) => {
      calls.push(opts);
      const scope = (opts.filters as { user_id?: string } | undefined)?.user_id;
      return { results: typeof scope === 'string' ? rowsByScope[scope] ?? [] : [] };
    }),
  } as unknown as MemoryClient;
}

describe('Mem0Backend.search — scope invariant (P-003) + relevance floor (P-001)', () => {
  it('fans out ONE user_id-scoped query per scope — never an unscoped full-table search', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient(
      { userA: [{ id: '1', memory: 'a', score: 0.9 }], 'harness:x': [{ id: '2', memory: 'b', score: 0.8 }] },
      calls,
    );
    const be = new Mem0Backend({ getClient: async () => client });

    const out = await be.search('q', { scope: ['userA', 'harness:x'], limit: 6 });

    expect(calls).toHaveLength(2);
    // every call carries a non-empty user_id filter — the pool is bounded by scope
    const scopes = calls.map((c) => (c.filters as { user_id?: string }).user_id);
    expect(scopes.every((s) => typeof s === 'string' && s.length > 0)).toBe(true);
    expect([...scopes].sort()).toEqual(['harness:x', 'userA']);
    expect(out.map((e) => e.id).sort()).toEqual(['1', '2']);
  });

  it('an empty-string scope yields no pulls (not an unscoped scan)', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient({}, calls);
    const be = new Mem0Backend({ getClient: async () => client });
    // scopesOf drops empty strings, so an all-empty scope produces zero queries
    // and an empty result — never a global full-table search.
    const out = await be.search('q', { scope: [''] as unknown as string[] });
    expect(calls).toHaveLength(0);
    expect(out).toEqual([]);
  });

  it('floors a hard-negative-shaped result to EMPTY (P-001 end-to-end)', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient({ userA: [{ id: '1', memory: 'x', score: 0.39 }, { id: '2', memory: 'y', score: 0.3 }] }, calls);
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.search('off-topic', { scope: 'userA', limit: 6, minScore: 0.45 });
    expect(out).toEqual([]);
  });

  it('keeps real hits above the floor (recall preserved)', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient({ userA: [{ id: '1', memory: 'x', score: 0.55 }, { id: '2', memory: 'y', score: 0.2 }] }, calls);
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.search('on-topic', { scope: 'userA', limit: 6, minScore: 0.45 });
    expect(out.map((e) => e.id)).toEqual(['1']);
  });
});

// ---------------------------------------------------------------------------
// GAP 2 — remember(): verbatim→infer:false (EI-178), metadata+kind merge,
// {ids, storedEvents} wiring via extractAddedIds/extractStoredEventCount.
// ---------------------------------------------------------------------------

describe('Mem0Backend.remember — verbatim/infer mapping, metadata+kind, {ids,storedEvents}', () => {
  it('maps verbatim:true → infer:false (EI-178 — skip LLM extraction)', async () => {
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, event: 'ADD' }] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    await be.remember('raw fact', { scope: 'userA', verbatim: true });

    expect(add).toHaveBeenCalledTimes(1);
    const [content, opts] = add.mock.calls[0];
    expect(content).toBe('raw fact');
    expect((opts as Record<string, unknown>).infer).toBe(false);
    expect((opts as Record<string, unknown>).userId).toBe('userA');
  });

  it('OMITS infer entirely when verbatim is falsy (keeps mem0 LLM extraction on)', async () => {
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, event: 'ADD' }] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    await be.remember('a fact', { scope: 'userA' });

    const opts = add.mock.calls[0][1] as Record<string, unknown>;
    // not `infer: true` — the key must be absent so mem0's default (infer) holds.
    expect('infer' in opts).toBe(false);
  });

  it('merges opts.metadata and sets metadata.kind from opts.kind', async () => {
    const add = vi.fn(async () => ({ results: [] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    await be.remember('fact', {
      scope: 'userA',
      kind: 'preference',
      metadata: { source: 'chat', anchor: 'x' },
    });

    const opts = add.mock.calls[0][1] as Record<string, unknown>;
    expect(opts.metadata).toEqual({ source: 'chat', anchor: 'x', kind: 'preference' });
  });

  it('kind overrides any kind already present in opts.metadata', async () => {
    const add = vi.fn(async () => ({ results: [] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    await be.remember('fact', { scope: 'userA', kind: 'identity', metadata: { kind: 'STALE' } });

    const opts = add.mock.calls[0][1] as Record<string, unknown>;
    expect((opts.metadata as Record<string, unknown>).kind).toBe('identity');
  });

  it('omits metadata.kind when opts.kind is undefined', async () => {
    const add = vi.fn(async () => ({ results: [] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    await be.remember('fact', { scope: 'userA', metadata: { source: 'chat' } });

    const opts = add.mock.calls[0][1] as Record<string, unknown>;
    expect(opts.metadata).toEqual({ source: 'chat' });
    expect('kind' in (opts.metadata as object)).toBe(false);
  });

  it('returns {ids,storedEvents} from a mixed ADD/UPDATE/NONE result', async () => {
    const add = vi.fn(async () => ({
      results: [
        { id: UUID_A, event: 'ADD' },
        { id: UUID_B, event: 'UPDATE' },
        { id: UUID_C, event: 'NONE' },
      ],
    }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    const out = await be.remember('fact', { scope: 'userA' });

    // ids = ADD rows only; storedEvents = ADD + UPDATE (NONE excluded).
    expect(out.ids).toEqual([UUID_A]);
    expect(out.storedEvents).toBe(2);
  });

  it('honest failure: a swallowed extraction ({results:[]}) yields ids:[] storedEvents:0', async () => {
    const add = vi.fn(async () => ({ results: [] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });

    const out = await be.remember('fact', { scope: 'userA' });

    expect(out).toEqual({ ids: [], storedEvents: 0 });
  });

  it('throws MemoryUnavailableError when the client is null', async () => {
    const be = new Mem0Backend({ getClient: async () => null });
    await expect(be.remember('x', { scope: 'userA' })).rejects.toBeInstanceOf(MemoryUnavailableError);
  });
});

// ---------------------------------------------------------------------------
// GAP 2 (helpers) — extractAddedIds / extractStoredEventCount, incl. the
// infer:false nested-metadata.event wire shape and the UUID guard.
// ---------------------------------------------------------------------------

describe('extractAddedIds / extractStoredEventCount', () => {
  it('extractAddedIds returns only valid-UUID ADD ids, case-insensitive event', () => {
    const result = {
      results: [
        { id: UUID_A, event: 'ADD' },
        { id: UUID_B, event: 'add' }, // lowercase still ADD after upper()
        { id: UUID_C, event: 'UPDATE' },
        { id: 'not-a-uuid', event: 'ADD' }, // rejected by UUID_RE
      ],
    };
    expect(extractAddedIds(result)).toEqual([UUID_A, UUID_B]);
  });

  it('reads the infer:false nested metadata.event wire shape', () => {
    const result = {
      results: [
        { id: UUID_A, metadata: { event: 'ADD' } },
        { id: UUID_B, metadata: { event: 'UPDATE' } },
      ],
    };
    expect(extractAddedIds(result)).toEqual([UUID_A]);
    expect(extractStoredEventCount(result)).toBe(2);
  });

  it('non-array / missing results → empty / zero (no throw)', () => {
    expect(extractAddedIds(null)).toEqual([]);
    expect(extractAddedIds({})).toEqual([]);
    expect(extractAddedIds({ results: 'nope' })).toEqual([]);
    expect(extractStoredEventCount(null)).toBe(0);
    expect(extractStoredEventCount({ results: [{ event: 'NONE' }, { event: 'DELETE' }] })).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// GAP 3 — forget / update / list / get / available.
// ---------------------------------------------------------------------------

describe('Mem0Backend.forget', () => {
  it('delegates to client.delete(id)', async () => {
    const del = vi.fn(async () => ({}));
    const be = new Mem0Backend({ getClient: async () => stubClient({ delete: del }) });
    await be.forget(UUID_A);
    expect(del).toHaveBeenCalledWith(UUID_A);
  });

  it('throws MemoryUnavailableError when client is null', async () => {
    const be = new Mem0Backend({ getClient: async () => null });
    await expect(be.forget(UUID_A)).rejects.toBeInstanceOf(MemoryUnavailableError);
  });
});

describe('Mem0Backend.update', () => {
  it('a metadata patch merges via the canonical-store updatePayload (vec-safe, no client.update)', async () => {
    const update = vi.fn();
    const updatePayload = vi.fn(async () => true);
    const be = new Mem0Backend({ getClient: async () => stubClient({ update }), updatePayload });
    await be.update(UUID_A, { metadata: { kind: 'reference', workspace_id: 'papercusp-workspace' } });
    expect(updatePayload).toHaveBeenCalledWith(UUID_A, { kind: 'reference', workspace_id: 'papercusp-workspace' });
    // The metadata path never touches mem0's text-only update.
    expect(update).not.toHaveBeenCalled();
  });

  it('a metadata patch on an unknown id throws the not-found contract', async () => {
    const updatePayload = vi.fn(async () => false); // no row matched
    const be = new Mem0Backend({ getClient: async () => stubClient({}), updatePayload });
    await expect(be.update(UUID_A, { metadata: { kind: 'x' } })).rejects.toThrow(/not found/i);
  });

  it('text + metadata both apply (metadata merge, then the text update)', async () => {
    const update = vi.fn(async () => ({}));
    const updatePayload = vi.fn(async () => true);
    const be = new Mem0Backend({ getClient: async () => stubClient({ update }), updatePayload });
    await be.update(UUID_A, { text: 'new', metadata: { kind: 'project' } });
    expect(updatePayload).toHaveBeenCalledWith(UUID_A, { kind: 'project' });
    expect(update).toHaveBeenCalledWith(UUID_A, 'new');
  });

  it('a text patch delegates to client.update(id, text)', async () => {
    const update = vi.fn(async () => ({}));
    const be = new Mem0Backend({ getClient: async () => stubClient({ update }) });
    await be.update(UUID_A, { text: 'replacement' });
    expect(update).toHaveBeenCalledWith(UUID_A, 'replacement');
  });

  it('an empty patch is a no-op — never resolves the client', async () => {
    const update = vi.fn();
    const getClient = vi.fn(async () => stubClient({ update }));
    const be = new Mem0Backend({ getClient });
    await expect(be.update(UUID_A, {})).resolves.toBeUndefined();
    expect(update).not.toHaveBeenCalled();
    // text===undefined returns BEFORE resolving the client.
    expect(getClient).not.toHaveBeenCalled();
  });
});

describe('Mem0Backend.list', () => {
  it('fans out one getAll per scope with a user_id filter + LIST_TOP_K cap', async () => {
    const rowsByScope: Record<string, Array<{ id: string; memory: string; metadata?: Record<string, unknown> }>> = {
      userA: [{ id: '1', memory: 'a' }],
      'harness:x': [{ id: '2', memory: 'b' }],
    };
    const calls: Record<string, unknown>[] = [];
    const getAll = vi.fn(async (opts: Record<string, unknown>) => {
      calls.push(opts);
      const scope = (opts.filters as { user_id?: string }).user_id ?? '';
      return { results: rowsByScope[scope] ?? [] };
    });
    const be = new Mem0Backend({ getClient: async () => stubClient({ getAll }) });

    const out = await be.list({ scope: ['userA', 'harness:x'] });

    expect(calls).toHaveLength(2);
    expect(calls.every((c) => (c.topK as number) === 5000)).toBe(true);
    expect(calls.map((c) => (c.filters as { user_id: string }).user_id).sort()).toEqual(['harness:x', 'userA']);
    expect(out.map((e) => e.id).sort()).toEqual(['1', '2']);
    // scope is stamped from the pool the row came from.
    expect(out.find((e) => e.id === '1')?.scope).toBe('userA');
  });

  it('filters by kind when opts.kind is set (reads metadata.kind)', async () => {
    const getAll = vi.fn(async () => ({
      results: [
        { id: '1', memory: 'p', metadata: { kind: 'preference' } },
        { id: '2', memory: 'i', metadata: { kind: 'identity' } },
        { id: '3', memory: 'u', metadata: {} },
      ],
    }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ getAll }) });

    const out = await be.list({ scope: 'userA', kind: 'preference' });
    expect(out.map((e) => e.id)).toEqual(['1']);
  });

  it('de-dupes a row that appears in two scopes (mergeById)', async () => {
    const getAll = vi.fn(async () => ({ results: [{ id: 'dup', memory: 'x' }] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ getAll }) });
    const out = await be.list({ scope: ['userA', 'userB'] });
    expect(out.map((e) => e.id)).toEqual(['dup']);
  });
});

describe('Mem0Backend.get', () => {
  it('derives scope from row.user_id', async () => {
    const get = vi.fn(async () => ({ id: UUID_A, memory: 'hi', user_id: 'harness:y', metadata: { kind: 'project' } }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ get }) });

    const out = await be.get(UUID_A);
    expect(out).toEqual({ id: UUID_A, text: 'hi', kind: 'project', scope: 'harness:y', metadata: { kind: 'project' } });
  });

  it('falls back to empty scope when user_id is absent', async () => {
    const get = vi.fn(async () => ({ id: UUID_A, memory: 'hi' }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ get }) });
    const out = await be.get(UUID_A);
    expect(out?.scope).toBe('');
  });

  it('returns null when the row does not exist', async () => {
    const get = vi.fn(async () => null);
    const be = new Mem0Backend({ getClient: async () => stubClient({ get }) });
    expect(await be.get(UUID_A)).toBeNull();
  });
});

describe('Mem0Backend.available', () => {
  it('client present → {ok:true}', async () => {
    const be = new Mem0Backend({ getClient: async () => stubClient() });
    expect(await be.available()).toEqual({ ok: true });
  });

  it('client null → {ok:false, reason:mem0_unavailable}', async () => {
    const be = new Mem0Backend({ getClient: async () => null });
    expect(await be.available()).toEqual({ ok: false, reason: 'mem0_unavailable' });
  });

  it('a rejected getClient is swallowed → {ok:false, mem0_unavailable} (non-throwing probe)', async () => {
    const be = new Mem0Backend({ getClient: async () => { throw new Error('boom'); } });
    expect(await be.available()).toEqual({ ok: false, reason: 'mem0_unavailable' });
  });
});

// ---------------------------------------------------------------------------
// GAP 11 — rememberConversation(): KEEPS LLM extraction (no infer:false),
// passes the message array through, honest storedEvents=0 on empty results.
// ---------------------------------------------------------------------------

describe('Mem0Backend.rememberConversation', () => {
  it('passes the message array to client.add and NEVER sets infer:false (keeps extraction)', async () => {
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, event: 'ADD' }] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });
    const messages = [
      { role: 'user', content: 'I prefer dark mode' },
      { role: 'assistant', content: 'noted' },
    ];

    await be.rememberConversation(messages, { scope: 'userA', kind: 'preference', metadata: { src: 'op' } });

    const [content, opts] = add.mock.calls[0];
    expect(content).toEqual(messages);
    expect(Array.isArray(content)).toBe(true);
    const o = opts as Record<string, unknown>;
    expect('infer' in o).toBe(false); // extraction stays ON
    expect(o.userId).toBe('userA');
    expect(o.metadata).toEqual({ src: 'op', kind: 'preference' });
  });

  it('returns {ids,storedEvents} from the result', async () => {
    const add = vi.fn(async () => ({
      results: [{ id: UUID_A, event: 'ADD' }, { id: UUID_B, event: 'UPDATE' }],
    }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });
    const out = await be.rememberConversation([{ role: 'user', content: 'hi' }], { scope: 'userA' });
    expect(out).toEqual({ ids: [UUID_A], storedEvents: 2 });
  });

  it('honest failure: empty results → ids:[] storedEvents:0', async () => {
    const add = vi.fn(async () => ({ results: [] }));
    const be = new Mem0Backend({ getClient: async () => stubClient({ add }) });
    const out = await be.rememberConversation([{ role: 'user', content: 'hi' }], { scope: 'userA' });
    expect(out).toEqual({ ids: [], storedEvents: 0 });
  });

  it('throws MemoryUnavailableError when client is null', async () => {
    const be = new Mem0Backend({ getClient: async () => null });
    await expect(
      be.rememberConversation([{ role: 'user', content: 'hi' }], { scope: 'userA' }),
    ).rejects.toBeInstanceOf(MemoryUnavailableError);
  });
});

// ---------------------------------------------------------------------------
// Memory decay (recency ranking bias) — search-time re-ordering only.
// The OSS equivalent of mem0 platform's Memory Decay (absent from mem0ai
// ≤3.0.3 OSS). Scores stay RAW so threshold consumers keep calibration.
// ---------------------------------------------------------------------------
describe('Mem0Backend.search — memory decay (recency ranking bias)', () => {
  const DAY = 86_400_000;
  const iso = (agoMs: number): string => new Date(Date.now() - agoMs).toISOString();

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('re-orders a near-tie toward the fresher memory, scores unchanged', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient(
      {
        userA: [
          { id: 'old', memory: 'x', score: 0.8, updatedAt: iso(60 * DAY) } as Row,
          { id: 'fresh', memory: 'y', score: 0.76, updatedAt: iso(0) } as Row,
        ],
      },
      calls,
    );
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.search('q', { scope: 'userA', limit: 6 });
    // sort keys (floor .9, half-life 30d): old .8×.925=.74 < fresh .76×1=.76
    expect(out.map((e) => e.id)).toEqual(['fresh', 'old']);
    // reported scores are the RAW mem0 scores — decay never rewrites them
    expect(out.map((e) => e.score)).toEqual([0.76, 0.8]);
  });

  it('is bounded: a clearly stronger old match stays ahead of a weak fresh one', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient(
      {
        userA: [
          { id: 'old', memory: 'x', score: 0.9, updatedAt: iso(365 * DAY) } as Row,
          { id: 'fresh', memory: 'y', score: 0.7, updatedAt: iso(0) } as Row,
        ],
      },
      calls,
    );
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.search('q', { scope: 'userA', limit: 6 });
    // even fully decayed, old's key is ≥ .9×.9=.81 > fresh's .7
    expect(out.map((e) => e.id)).toEqual(['old', 'fresh']);
  });

  it('PAPERCUSP_MEMORY_DECAY=0 disables the re-ordering', async () => {
    vi.stubEnv('PAPERCUSP_MEMORY_DECAY', '0');
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient(
      {
        userA: [
          { id: 'old', memory: 'x', score: 0.8, updatedAt: iso(60 * DAY) } as Row,
          { id: 'fresh', memory: 'y', score: 0.76, updatedAt: iso(0) } as Row,
        ],
      },
      calls,
    );
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.search('q', { scope: 'userA', limit: 6 });
    expect(out.map((e) => e.id)).toEqual(['old', 'fresh']);
  });

  it('rows without timestamps rank by raw score (factor 1)', async () => {
    const calls: Record<string, unknown>[] = [];
    const client = fakeClient(
      {
        userA: [
          { id: 'a', memory: 'x', score: 0.8 },
          { id: 'b', memory: 'y', score: 0.76 },
        ],
      },
      calls,
    );
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.search('q', { scope: 'userA', limit: 6 });
    expect(out.map((e) => e.id)).toEqual(['a', 'b']);
  });
});

// ---------------------------------------------------------------------------
// Entity-linking parity for verbatim writes: mem0's infer:false branch skips
// its Phase-7 entity linking, so remember(verbatim) calls the same private
// per-memory linker mem0's update() uses — feature-detected, best-effort.
// ---------------------------------------------------------------------------
describe('Mem0Backend.remember — entity-linking parity on verbatim writes', () => {
  it('links entities for each ADD id with the scope as user_id', async () => {
    const linker = vi.fn(async () => {});
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, metadata: { event: 'ADD' } }] }));
    const client = stubClient({ add, _linkEntitiesForMemory: linker } as Record<string, unknown>);
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.remember('WI-1234 shipped', { scope: 'userA', verbatim: true });
    expect(out.ids).toEqual([UUID_A]);
    expect(linker).toHaveBeenCalledTimes(1);
    expect(linker).toHaveBeenCalledWith(UUID_A, 'WI-1234 shipped', { user_id: 'userA' });
  });

  it('does NOT invoke the linker on the non-verbatim (infer) path — mem0 links internally there', async () => {
    const linker = vi.fn(async () => {});
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, event: 'ADD' }] }));
    const client = stubClient({ add, _linkEntitiesForMemory: linker } as Record<string, unknown>);
    const be = new Mem0Backend({ getClient: async () => client });
    await be.remember('fact', { scope: 'userA' });
    expect(linker).not.toHaveBeenCalled();
  });

  it('a linker failure never fails the write', async () => {
    const linker = vi.fn(async () => {
      throw new Error('entity store down');
    });
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, metadata: { event: 'ADD' } }] }));
    const client = stubClient({ add, _linkEntitiesForMemory: linker } as Record<string, unknown>);
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.remember('fact', { scope: 'userA', verbatim: true });
    expect(out.ids).toEqual([UUID_A]);
  });

  it('no linker on the client (mem0 upgrade) → silent no-op, write succeeds', async () => {
    const add = vi.fn(async () => ({ results: [{ id: UUID_A, metadata: { event: 'ADD' } }] }));
    const client = stubClient({ add });
    const be = new Mem0Backend({ getClient: async () => client });
    const out = await be.remember('fact', { scope: 'userA', verbatim: true });
    expect(out.ids).toEqual([UUID_A]);
  });
});
