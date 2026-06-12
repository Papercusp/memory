/**
 * CanonicalVectorStore store-kind segregation (EI-366).
 *
 * mem0 creates its entity-linking store as a second CanonicalVectorStore
 * distinguished only by a `*_entities` collectionName suffix. The store
 * shares the physical tables across instances, so search()/list() must
 * partition rows by payload shape: entity payloads always carry
 * `entityType`, memory payloads never do. Without this, COMPOUND/PROPER
 * fragments pollute memory recall (84% of the live store pre-fix).
 *
 * The pool is swapped for a query-capturing fake — these tests pin the
 * SQL the store emits, not PG behavior (the live audit query is the
 * integration proof).
 */
import { describe, it, expect, vi } from 'vitest';
import { CanonicalVectorStore, type CanonicalStoreConfig } from './canonical-store';

type CapturedQuery = { sql: string; params: unknown[] };

function makeStore(collectionName: string): { store: CanonicalVectorStore; queries: CapturedQuery[] } {
  const queries: CapturedQuery[] = [];
  const fakePool = {
    query: vi.fn(async (sql: string, params: unknown[] = []) => {
      queries.push({ sql, params });
      return { rows: [{ n: 0 }], rowCount: 0 };
    }),
    on: () => {},
  };
  const cfg: CanonicalStoreConfig = {
    host: 'localhost',
    port: 5432,
    user: 'u',
    password: 'p',
    dbname: 'db',
    schema: 'harness_shared',
    collectionName,
    vecTable: 'memory_vec_local',
    embeddingModelDims: 3,
  };
  const store = new CanonicalVectorStore(cfg);
  (store as unknown as { pool: unknown }).pool = fakePool;
  return { store, queries };
}

const VEC = [0.1, 0.2, 0.3];

describe('CanonicalVectorStore store-kind segregation', () => {
  it('memory-kind search excludes entity rows', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    expect(queries).toHaveLength(1);
    expect(queries[0].sql).toContain("NOT (c.payload ? 'entityType')");
    // scope filter still applies alongside the kind clause
    expect(queries[0].sql).toContain("c.payload->>'user_id' = $3");
    expect(queries[0].params).toContain('scope-a');
  });

  it('entity-kind search (collection *_entities) sees ONLY entity rows', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    expect(queries[0].sql).toContain("c.payload ? 'entityType'");
    expect(queries[0].sql).not.toContain('NOT (');
  });

  it('memory-kind list excludes entity rows in BOTH the page and the count', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.list({ user_id: 'scope-a' }, 10);
    expect(queries).toHaveLength(2); // page + count
    for (const q of queries) {
      expect(q.sql).toContain("NOT (payload ? 'entityType')");
    }
  });

  it('entity-kind list sees only entity rows', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.list(undefined, 10);
    expect(queries).toHaveLength(2);
    for (const q of queries) {
      expect(q.sql).toContain("payload ? 'entityType'");
      expect(q.sql).not.toContain('NOT (');
    }
  });

  it('search with no filters still carries the kind clause', async () => {
    const { store, queries } = makeStore('operator_memory_openai');
    await store.search(VEC, 5);
    expect(queries[0].sql).toContain("WHERE NOT (c.payload ? 'entityType')");
  });

  it('insert is kind-agnostic — payloads pass through untouched', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.insert([VEC], ['id-1'], [{ data: 'x', entityType: 'COMPOUND', linkedMemoryIds: ['m1'] }]);
    const canonical = queries.find((q) => q.sql.includes('memory_canonical'));
    expect(canonical).toBeDefined();
    expect(JSON.parse(canonical!.params[1] as string)).toMatchObject({ entityType: 'COMPOUND' });
  });
});
