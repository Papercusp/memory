import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { configureMemory } from './config';
import { getMemoryClient } from './mem0-client';

vi.mock('./mem0-client', () => ({ getMemoryClient: vi.fn() }));
vi.mock('pg', () => ({ Client: FakeClient, default: { Client: FakeClient } }));

type Payload = { data: string; user_id?: string; entityType?: string; linkedMemoryIds?: string[] };
let records: Map<string, Payload>;
let vectors: Record<string, Set<string>>;
let snapshot: { records: Map<string, Payload>; vectors: Record<string, Set<string>> } | undefined;
let failCleanup: boolean;
let failLink: Set<string>;
let failEmbedding: boolean;
let swallow: boolean;
let noOp: boolean;
let onLink: ((id: string) => void) | undefined;
let linked: string[];
let connections: number;
let ended: number;

function copyRecords() {
  return new Map([...records].map(([id, payload]) => [id, structuredClone(payload)]));
}
function copyVectors() {
  return Object.fromEntries(Object.entries(vectors).map(([table, ids]) => [table, new Set(ids)]));
}
function project(id: string) {
  const payload = records.get(id)!;
  return { id, payload, data: payload.data, user_id: payload.user_id ?? null,
    linked: payload.linkedMemoryIds ?? [] };
}
function junk(id = 'junk', mids = ['m1'], user = 'u') {
  records.set(id, { data: 'just before end of', user_id: user, entityType: 'COMPOUND', linkedMemoryIds: mids });
  for (const ids of Object.values(vectors)) ids.add(id);
}
function memory(id: string, user = 'u', data = 'Harrier stores the memory') {
  records.set(id, { data, user_id: user });
}

// The same fake-pg seam as reembed.test.ts, with transactional state so the
// assertions concern retained data and retry eligibility, not query spelling.
class FakeClient {
  async connect() { connections += 1; }
  async end() { ended += 1; }
  async query(sql: string, params: unknown[] = []) {
    const q = sql.replace(/\s+/g, ' ').trim();
    if (q === 'BEGIN') { snapshot = { records: copyRecords(), vectors: copyVectors() }; return { rows: [] }; }
    if (q === 'COMMIT') { snapshot = undefined; return { rows: [] }; }
    if (q === 'ROLLBACK') {
      if (snapshot) { records = snapshot.records; vectors = snapshot.vectors; }
      snapshot = undefined;
      return { rows: [] };
    }
    if (q.startsWith('SELECT')) {
      let ids: string[];
      if (q.includes('FOR UPDATE')) ids = params[0] as string[];
      else if (q.includes("entityType' = 'COMPOUND'")) {
        ids = [...records].filter(([, p]) => p.entityType === 'COMPOUND'
          && (!params[0] || p.user_id === params[0])).map(([id]) => id);
      } else ids = Array.isArray(params[0]) ? params[0] as string[] : [params[0] as string];
      const rows = ids.filter(id => records.has(id)).map(project)
        .filter(row => !q.includes("NOT (payload ? 'entityType')") || !row.payload.entityType);
      return { rows, rowCount: rows.length };
    }
    if (q.startsWith('UPDATE')) {
      const id = params[0] as string;
      const payload = records.get(id);
      if (!payload) return { rows: [], rowCount: 0 };
      // A touched queue row follows older rows (ORDER BY updated_at, id).
      records.delete(id);
      records.set(id, { ...payload, linkedMemoryIds: JSON.parse(params[1] as string) });
      return { rows: [], rowCount: 1 };
    }
    if (q.startsWith('DELETE')) {
      const ids = Array.isArray(params[0]) ? params[0] as string[] : [params[0] as string];
      const table = Object.keys(vectors).find(name => q.includes(name));
      if (table) {
        if (table === 'memory_vec_gemma' && failCleanup) { failCleanup = false; throw new Error('cleanup failed'); }
        for (const id of ids) vectors[table].delete(id);
      } else for (const id of ids) records.delete(id);
      return { rows: [], rowCount: ids.length };
    }
    throw new Error(`Unexpected fixture query: ${q}`);
  }
}

function makeMem0() {
  return {
    embedder: { embed: async () => { if (failEmbedding) throw new Error('embed failed'); return [1]; } },
    async getEntityStore() {
      return { insert: async (_vecs: number[][], ids: string[], payloads: Payload[]) => {
        ids.forEach((id, i) => {
          records.set(id, structuredClone(payloads[i]));
          for (const set of Object.values(vectors)) set.add(id);
        });
      } };
    },
    async _linkEntitiesForMemory(id: string, _text: string, filters: Record<string, string>) {
      linked.push(id);
      if (noOp) return;
      try {
        if (failLink.has(id)) throw new Error('link failed');
        const vec = await this.embedder.embed();
        const store = await this.getEntityStore();
        await store.insert([vec], [`clean-${id}`], [{ data: 'Harrier', entityType: 'PROPER',
          user_id: filters.user_id, linkedMemoryIds: [id] }]);
        onLink?.(id);
      } catch (error) {
        // The real mem0 private linker swallows embed/store failures.
        if (!swallow) throw error;
      }
    },
  };
}

let relinkEntities: typeof import('./relink-entities').relinkEntities;
beforeEach(async () => {
  records = new Map();
  vectors = { memory_vec_harrier: new Set(), memory_vec_gemma: new Set() };
  snapshot = undefined;
  failCleanup = failEmbedding = swallow = noOp = false;
  failLink = new Set(); linked = []; connections = ended = 0; onLink = undefined;
  configureMemory({ getAdminUrl: () => 'postgres://u:p@localhost:5432/db',
    getCredentials: async () => ({}), resolveEmbedder: async () => ({ mode: 'disabled' as const }),
    schema: 'harness_shared' });
  vi.mocked(getMemoryClient).mockReset().mockResolvedValue(makeMem0() as never);
  ({ relinkEntities } = await import('./relink-entities'));
  junk(); memory('m1');
});
afterEach(() => vi.unstubAllEnvs());

describe('relinkEntities — loss-safe maintenance and bounded retries', () => {
  it.each(['missing client', 'missing linker', 'client rejects'])('preserves rows and both vectors on %s', async mode => {
    if (mode === 'missing client') vi.mocked(getMemoryClient).mockResolvedValue(null);
    if (mode === 'missing linker') vi.mocked(getMemoryClient).mockResolvedValue({} as never);
    if (mode === 'client rejects') vi.mocked(getMemoryClient).mockRejectedValue(new Error('unavailable'));
    const before = { records: copyRecords(), vectors: copyVectors() };
    await expect(relinkEntities()).rejects.toThrow();
    expect(records).toEqual(before.records); expect(vectors).toEqual(before.vectors);
    expect(ended).toBe(1);
  });

  it('keeps the original data until replacement succeeds', async () => {
    onLink = () => expect(records.has('junk')).toBe(true);
    expect(await relinkEntities()).toMatchObject({ memoriesRelinked: 1, errors: 0, junkRemaining: 0 });
    expect(records.has('clean-m1')).toBe(true); expect(records.has('junk')).toBe(false);
  });

  it('retains a failed memory and makes a later successful retry possible', async () => {
    failLink.add('m1');
    expect(await relinkEntities()).toMatchObject({ errors: 1, memoriesRelinked: 0, junkRemaining: 1 });
    expect(records.get('junk')?.linkedMemoryIds).toEqual(['m1']);
    for (const set of Object.values(vectors)) expect(set.has('junk')).toBe(true);
    failLink.clear();
    expect(await relinkEntities()).toMatchObject({ errors: 0, memoriesRelinked: 1, junkRemaining: 0 });
  });

  it('does not equate a swallowed embedding failure with success', async () => {
    failEmbedding = swallow = true;
    expect(await relinkEntities()).toMatchObject({ errors: 1, memoriesRelinked: 0, junkRemaining: 1 });
    expect(records.get('junk')?.linkedMemoryIds).toEqual(['m1']);
  });

  it('preserves the retry marker when a void private linker performs no verified write', async () => {
    noOp = true;
    expect(await relinkEntities()).toMatchObject({ errors: 1, memoriesRelinked: 0, junkRemaining: 1 });
    expect(records.has('junk')).toBe(true);
  });

  it('drains shared fanout across bounded calls instead of repeating the successful prefix', async () => {
    junk('junk', ['m1', 'm2', 'm3']); memory('m2'); memory('m3');
    for (let i = 0; i < 3; i++) {
      expect(await relinkEntities({ maxMemories: 1 })).toMatchObject({ affectedMemories: 1, memoriesRelinked: 1 });
    }
    expect(linked).toEqual(['m1', 'm2', 'm3']); expect(records.has('junk')).toBe(false);
  });

  it('does not starve another entity behind a persistently failing singleton', async () => {
    junk('later', ['m2']); memory('m2'); failLink.add('m1');
    await relinkEntities({ maxMemories: 1 });
    expect(await relinkEntities({ maxMemories: 1 })).toMatchObject({ memoriesRelinked: 1, errors: 0 });
    expect(records.has('clean-m2')).toBe(true); expect(records.has('junk')).toBe(true);
  });

  it('retains failed and concurrently added links while removing only successful work', async () => {
    junk('junk', ['m1', 'm2']); memory('m2'); failLink.add('m2');
    onLink = () => { records.get('junk')!.linkedMemoryIds!.push('m3'); };
    expect(await relinkEntities({ maxMemories: 2 })).toMatchObject({ memoriesRelinked: 1, errors: 1 });
    expect(new Set(records.get('junk')?.linkedMemoryIds)).toEqual(new Set(['m2', 'm3']));
  });

  it('rolls back vector cleanup if retiring the canonical row fails partway', async () => {
    failCleanup = true;
    expect(await relinkEntities()).toMatchObject({ errors: 1, memoriesRelinked: 0, junkRemaining: 1 });
    expect(records.has('junk')).toBe(true);
    for (const set of Object.values(vectors)) expect(set.has('junk')).toBe(true);
  });

  it('does not retire work against memory text changed during linking', async () => {
    onLink = () => { records.get('m1')!.data = 'changed text'; };
    expect(await relinkEntities()).toMatchObject({ errors: 1, memoriesRelinked: 0, junkRemaining: 1 });
    expect(records.has('junk')).toBe(true);
  });

  it('reports missing memory data and preserves its retry marker', async () => {
    records.delete('m1');
    expect(await relinkEntities()).toMatchObject({ errors: 1, memoriesRelinked: 0, junkRemaining: 1 });
    expect(records.has('junk')).toBe(true);
  });

  it('keeps dry-run, tenant scope, PROPER and QUOTED rows intact', async () => {
    junk('foreign', ['foreign-m'], 'other'); memory('foreign-m', 'other');
    records.set('proper', { data: 'Harrier', entityType: 'PROPER', linkedMemoryIds: ['m1'] });
    records.set('quoted', { data: 'the folder', entityType: 'QUOTED', linkedMemoryIds: ['m1'] });
    const before = copyRecords();
    expect(await relinkEntities({ dryRun: true, userId: 'u' })).toMatchObject({ dryRun: true, affectedMemories: 1 });
    expect(records).toEqual(before); expect(getMemoryClient).not.toHaveBeenCalled();
    await relinkEntities({ userId: 'u' });
    for (const id of ['foreign', 'proper', 'quoted']) expect(records.get(id)).toEqual(before.get(id));
    expect(await relinkEntities({ userId: 'u' })).toMatchObject({ affectedMemories: 0, memoriesRelinked: 0 });
  });

  it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])('rejects invalid maxMemories %s before opening PG', async value => {
    await expect(relinkEntities({ maxMemories: value })).rejects.toThrow('relink_invalid_max_memories');
    expect(connections).toBe(0);
  });
});
