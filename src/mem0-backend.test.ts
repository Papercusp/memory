import { describe, it, expect, vi } from 'vitest';
import { Mem0Backend } from './mem0-backend';
import type { MemoryClient } from './mem0-client';

type Row = { id: string; memory: string; score: number };

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
