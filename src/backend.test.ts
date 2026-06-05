/**
 * Tests for the neutral MemoryBackend seam: the NoopBackend contract,
 * the Mem0Backend shape-mapping (scope→user_id, kind→metadata.kind,
 * fan-out merge, ADD-id extraction), and the registry/selector flip
 * (generalize-memory-backend-swappable-2026-06-05 P-002/P-005).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

import {
  MemoryUnavailableError,
  scopesOf,
  type MemoryBackend,
  type MemoryEntry,
} from './backend';
import { Mem0Backend, extractAddedIds, extractStoredEventCount } from './mem0-backend';
import { NoopBackend, NOOP_DISABLED_REASON } from './noop-backend';
import {
  getMemoryBackend,
  registerMemoryBackend,
  registeredMemoryBackends,
  _resetMemoryBackendsForTest,
} from './backend-registry';
import { configureMemory } from './config';
import type { MemoryClient } from './mem0-client';

const UUID_A = '11111111-1111-1111-1111-111111111111';
const UUID_B = '22222222-2222-2222-2222-222222222222';

function fakeClient(overrides: Partial<MemoryClient> = {}): MemoryClient {
  return {
    add: vi.fn(async () => ({ results: [{ id: UUID_A, event: 'ADD' }] })),
    delete: vi.fn(async () => ({})),
    get: vi.fn(async () => null),
    getAll: vi.fn(async () => ({ results: [] })),
    search: vi.fn(async () => ({ results: [] })),
    update: vi.fn(async () => ({})),
    ...overrides,
  } as MemoryClient;
}

function stubHost(extra: Record<string, unknown> = {}): void {
  configureMemory({
    getAdminUrl: () => 'postgres://stub',
    getCredentials: async () => ({}),
    resolveEmbedder: async () => ({ mode: 'disabled' as const }),
    buildEmbedderForMode: async () => async () => [],
    ...extra,
  });
}

describe('scopesOf', () => {
  it('normalizes a single scope to an array', () => {
    expect(scopesOf('user-1')).toEqual(['user-1']);
  });
  it('de-dupes and drops empties', () => {
    expect(scopesOf(['a', 'a', '', 'b'])).toEqual(['a', 'b']);
  });
});

describe('extractAddedIds', () => {
  it('keeps only ADD events with UUID ids', () => {
    expect(
      extractAddedIds({
        results: [
          { id: UUID_A, event: 'ADD' },
          { id: UUID_B, event: 'UPDATE' },
          { id: 'not-a-uuid', event: 'ADD' },
          { id: UUID_B, event: 'NONE' },
        ],
      }),
    ).toEqual([UUID_A]);
  });
  it('accepts the infer:false nested metadata.event shape', () => {
    expect(
      extractAddedIds({
        results: [
          { id: UUID_A, memory: 'raw', metadata: { event: 'ADD' } },
          { id: UUID_B, memory: 'raw2', metadata: { event: 'NONE' } },
        ],
      }),
    ).toEqual([UUID_A]);
  });

  it('tolerates malformed results', () => {
    expect(extractAddedIds(null)).toEqual([]);
    expect(extractAddedIds(undefined)).toEqual([]);
    expect(extractAddedIds({})).toEqual([]);
    expect(extractAddedIds({ results: null })).toEqual([]);
    expect(extractAddedIds({ results: 'nope' })).toEqual([]);
    expect(extractAddedIds('garbage')).toEqual([]);
  });

  it('is case-insensitive on event names', () => {
    expect(
      extractAddedIds({
        results: [
          { id: UUID_A, event: 'add' },
          { id: UUID_B, event: 'Add' },
        ],
      }),
    ).toEqual([UUID_A, UUID_B]);
  });

  it('skips rows with missing id', () => {
    expect(
      extractAddedIds({ results: [{ event: 'ADD' }, { id: UUID_A, event: 'ADD' }] }),
    ).toEqual([UUID_A]);
  });
});

describe('extractStoredEventCount', () => {
  it('counts ADD and UPDATE events, ignores NONE/DELETE', () => {
    expect(
      extractStoredEventCount({
        results: [
          { id: UUID_A, event: 'ADD' },
          { id: UUID_B, event: 'UPDATE' },
          { id: UUID_A, event: 'NONE' },
          { id: UUID_B, event: 'DELETE' },
        ],
      }),
    ).toBe(2);
  });

  it('is case-insensitive on event names', () => {
    expect(extractStoredEventCount({ results: [{ event: 'add' }, { event: 'Update' }] })).toBe(2);
  });

  it('returns 0 for a swallowed-extraction-failure shape (empty results) and malformed input', () => {
    expect(extractStoredEventCount({ results: [] })).toBe(0);
    expect(extractStoredEventCount(null)).toBe(0);
    expect(extractStoredEventCount(undefined)).toBe(0);
    expect(extractStoredEventCount({})).toBe(0);
    expect(extractStoredEventCount({ results: 'nope' })).toBe(0);
  });
});

describe('NoopBackend', () => {
  const noop = new NoopBackend();

  it('reports deliberately unavailable', async () => {
    expect(await noop.available()).toEqual({ ok: false, reason: NOOP_DISABLED_REASON });
  });
  it('reads come back empty, never throwing', async () => {
    expect(await noop.search('q', { scope: 'u' })).toEqual([]);
    expect(await noop.list({ scope: 'u' })).toEqual([]);
    expect(await noop.get('id')).toBeNull();
  });
  it('writes throw MemoryUnavailableError with the stable reason', async () => {
    await expect(noop.remember('x', { scope: 'u' })).rejects.toThrow(MemoryUnavailableError);
    await expect(noop.forget('id')).rejects.toMatchObject({ reason: NOOP_DISABLED_REASON });
    await expect(noop.update('id', { text: 'y' })).rejects.toThrow(MemoryUnavailableError);
  });
  it('does not claim the optional capabilities', () => {
    const asBackend: MemoryBackend = noop;
    expect(asBackend.rememberConversation).toBeUndefined();
    expect(asBackend.invalidate).toBeUndefined();
  });
});

describe('Mem0Backend', () => {
  it('available() reflects the client probe', async () => {
    const up = new Mem0Backend({ getClient: async () => fakeClient() });
    expect(await up.available()).toEqual({ ok: true });
    const down = new Mem0Backend({ getClient: async () => null });
    expect(await down.available()).toEqual({ ok: false, reason: 'mem0_unavailable' });
  });

  it('throws MemoryUnavailableError from every verb when the client is null', async () => {
    const b = new Mem0Backend({ getClient: async () => null });
    await expect(b.remember('x', { scope: 'u' })).rejects.toThrow(MemoryUnavailableError);
    await expect(b.search('q', { scope: 'u' })).rejects.toThrow(MemoryUnavailableError);
    await expect(b.list({ scope: 'u' })).rejects.toThrow(MemoryUnavailableError);
    await expect(b.get('id')).rejects.toThrow(MemoryUnavailableError);
    await expect(b.forget('id')).rejects.toThrow(MemoryUnavailableError);
    await expect(b.update('id', { text: 'y' })).rejects.toThrow(MemoryUnavailableError);
  });

  it('remember maps scope→userId and kind→metadata.kind, returns ADD ids', async () => {
    const client = fakeClient({
      add: vi.fn(async () => ({
        results: [
          { id: UUID_A, event: 'ADD' },
          { id: UUID_B, event: 'UPDATE' },
        ],
      })),
    });
    const b = new Mem0Backend({ getClient: async () => client });
    const r = await b.remember('fact', { scope: 'harness:papercup', kind: 'project', metadata: { a: 1 } });
    // ids = ADD rows only; storedEvents counts the UPDATE merge too (EI-25).
    expect(r).toEqual({ ids: [UUID_A], storedEvents: 2 });
    expect(client.add).toHaveBeenCalledWith('fact', {
      userId: 'harness:papercup',
      metadata: { a: 1, kind: 'project' },
    });
  });

  it('remember verbatim maps to infer:false and reads the nested metadata.event shape', async () => {
    // mem0's infer:false branch nests the event under metadata
    // (returnedMemories.push({ id, memory, metadata: { event: 'ADD' } })) —
    // verified against mem0ai 3.0.3 dist (D-008).
    const client = fakeClient({
      add: vi.fn(async () => ({
        results: [{ id: UUID_A, memory: 'fact', metadata: { event: 'ADD' } }],
      })),
    });
    const b = new Mem0Backend({ getClient: async () => client });
    const r = await b.remember('fact', { scope: 'u', verbatim: true });
    expect(r).toEqual({ ids: [UUID_A], storedEvents: 1 });
    expect(client.add).toHaveBeenCalledWith('fact', {
      userId: 'u',
      metadata: {},
      infer: false,
    });
  });

  it('remember without verbatim does NOT pass infer (mem0 default extraction path)', async () => {
    const client = fakeClient();
    const b = new Mem0Backend({ getClient: async () => client });
    await b.remember('fact', { scope: 'u' });
    const callOpts = (client.add as ReturnType<typeof vi.fn>).mock.calls[0][1] as Record<string, unknown>;
    expect('infer' in callOpts).toBe(false);
  });

  it('search fans out per scope, maps rows to neutral entries, merges + sorts', async () => {
    const client = fakeClient({
      search: vi.fn(async (_q: string, opts: Record<string, unknown>) => {
        const scope = (opts.filters as { user_id: string }).user_id;
        if (scope === 'user-1') {
          return {
            results: [
              { id: UUID_A, memory: 'shared hit', score: 0.4, metadata: { kind: 'preference' } },
            ],
          };
        }
        return {
          results: [
            { id: UUID_A, memory: 'shared hit', score: 0.9, metadata: { kind: 'preference' } },
            { id: UUID_B, memory: 'harness hit', score: 0.5, metadata: {} },
          ],
        };
      }),
    });
    const b = new Mem0Backend({ getClient: async () => client });
    const hits = await b.search('q', { scope: ['user-1', 'harness:papercup'], limit: 7 });

    expect(client.search).toHaveBeenCalledTimes(2);
    expect(client.search).toHaveBeenCalledWith('q', { filters: { user_id: 'user-1' }, limit: 7 });
    expect(client.search).toHaveBeenCalledWith('q', { filters: { user_id: 'harness:papercup' }, limit: 7 });

    // de-duped by id keeping the higher-scored hit; sorted desc
    expect(hits.map((h) => h.id)).toEqual([UUID_A, UUID_B]);
    expect(hits[0]).toMatchObject({
      id: UUID_A,
      text: 'shared hit',
      kind: 'preference',
      scope: 'harness:papercup',
      score: 0.9,
    });
    expect(hits[1]).toMatchObject({ id: UUID_B, text: 'harness hit', scope: 'harness:papercup' });
    expect(hits[1].kind).toBeUndefined();
  });

  it('list fans out getAll per scope and filters by kind', async () => {
    const client = fakeClient({
      getAll: vi.fn(async (opts: Record<string, unknown>) => {
        const scope = (opts.filters as { user_id: string }).user_id;
        return scope === 'user-1'
          ? { results: [{ id: UUID_A, memory: 'pref', metadata: { kind: 'preference' } }] }
          : { results: [{ id: UUID_B, memory: 'proj', metadata: { kind: 'project' } }] };
      }),
    });
    const b = new Mem0Backend({ getClient: async () => client });

    const all = await b.list({ scope: ['user-1', 'harness:x'] });
    expect(all).toHaveLength(2);
    expect(client.getAll).toHaveBeenCalledWith({ filters: { user_id: 'user-1' }, topK: 5000 });

    const onlyProjects = await b.list({ scope: ['user-1', 'harness:x'], kind: 'project' });
    expect(onlyProjects.map((e) => e.id)).toEqual([UUID_B]);
    expect(onlyProjects[0]).toMatchObject({ text: 'proj', kind: 'project', scope: 'harness:x' });
  });

  it('get maps a row (user_id → scope) and passes null through', async () => {
    const client = fakeClient({
      get: vi.fn(async () => ({ id: UUID_A, memory: 'hello', user_id: 'user-1', metadata: { kind: 'identity' } })),
    });
    const b = new Mem0Backend({ getClient: async () => client });
    expect(await b.get(UUID_A)).toMatchObject({ id: UUID_A, text: 'hello', scope: 'user-1', kind: 'identity' });

    const empty = new Mem0Backend({ getClient: async () => fakeClient() });
    expect(await empty.get('missing')).toBeNull();
  });

  it('forget delegates to delete', async () => {
    const client = fakeClient();
    const b = new Mem0Backend({ getClient: async () => client });
    await b.forget(UUID_A);
    expect(client.delete).toHaveBeenCalledWith(UUID_A);
  });

  it('update applies text, rejects metadata patches, no-ops on empty patch', async () => {
    const client = fakeClient();
    const b = new Mem0Backend({ getClient: async () => client });

    await b.update(UUID_A, { text: 'new text' });
    expect(client.update).toHaveBeenCalledWith(UUID_A, 'new text');

    await expect(b.update(UUID_A, { metadata: { x: 1 } })).rejects.toThrow(/metadata/);

    vi.mocked(client.update as ReturnType<typeof vi.fn>).mockClear();
    await b.update(UUID_A, {});
    expect(client.update).not.toHaveBeenCalled();
  });

  it('rememberConversation passes the window through to add', async () => {
    const client = fakeClient();
    const b = new Mem0Backend({ getClient: async () => client });
    const messages = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ];
    const r = await b.rememberConversation(messages, { scope: 'user-1', metadata: { turn_at: 5 } });
    expect(r).toEqual({ ids: [UUID_A], storedEvents: 1 });
    expect(client.add).toHaveBeenCalledWith(messages, {
      userId: 'user-1',
      metadata: { turn_at: 5 },
    });
  });
});

describe('backend registry / selector', () => {
  beforeEach(() => {
    _resetMemoryBackendsForTest();
  });

  it('defaults to mem0', () => {
    stubHost(); // no backend choice
    expect(getMemoryBackend()).toBeInstanceOf(Mem0Backend);
    expect(getMemoryBackend().name).toBe('mem0');
  });

  it('flips to noop via configureMemory({ backend }) — no handler changes', () => {
    stubHost({ backend: 'noop' });
    expect(getMemoryBackend()).toBeInstanceOf(NoopBackend);
    // flip back
    stubHost({ backend: 'mem0' });
    expect(getMemoryBackend()).toBeInstanceOf(Mem0Backend);
  });

  it('serves a direct instance unchanged', () => {
    const custom: MemoryBackend = new NoopBackend();
    stubHost({ backend: custom });
    expect(getMemoryBackend()).toBe(custom);
  });

  it('caches the per-name instance', () => {
    stubHost({ backend: 'noop' });
    expect(getMemoryBackend()).toBe(getMemoryBackend());
  });

  it('an unknown name throws loud, listing the registered set', () => {
    stubHost({ backend: 'claude-file' });
    expect(() => getMemoryBackend()).toThrow(/unknown memory backend 'claude-file'.*mem0, noop/);
  });

  it('out-of-lib backends register by name and become selectable', async () => {
    const entries: MemoryEntry[] = [
      { id: 'x', text: 'from custom', scope: 'user-1' },
    ];
    class CustomBackend extends NoopBackend {
      override async search(): Promise<MemoryEntry[]> {
        return entries;
      }
    }
    registerMemoryBackend('claude-file', () => new CustomBackend());
    stubHost({ backend: 'claude-file' });
    expect(registeredMemoryBackends()).toContain('claude-file');
    expect(await getMemoryBackend().search('q', { scope: 'user-1' })).toEqual(entries);
  });
});
