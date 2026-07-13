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
import {
  CanonicalVectorStore,
  isLowQualityCompoundEntity,
  lexicalTokens,
  splitTemporalControls,
  foldValidity,
  type CanonicalStoreConfig,
} from './canonical-store';

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

  it('insert stores a well-formed entity payload untouched', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.insert(
      [VEC],
      ['id-1'],
      [{ data: 'harrier embedder sidecar', entityType: 'COMPOUND', linkedMemoryIds: ['m1'] }],
    );
    const canonical = queries.find((q) => q.sql.includes('memory_canonical'));
    expect(canonical).toBeDefined();
    expect(JSON.parse(canonical!.params[1] as string)).toMatchObject({ entityType: 'COMPOUND' });
  });
});

describe('EI-10183 entity-quality gate — isLowQualityCompoundEntity', () => {
  // Real fragments observed in the live store (regex fallback + nlp residue).
  it.each([
    'so the re',
    'just before end of',
    'flaked mid',
    'left the one',
    'embed job stalled and',
    'nothing else pending',
    'nothing else drained',
    'the folder', // single generic head after stripping the article
    'of the', // pure function words
    'x', // too short
    '', // empty
  ])('rejects junk fragment %j', (frag) => {
    expect(isLowQualityCompoundEntity(frag)).toBe(true);
  });

  // Genuine noun phrases (incl. a leading article and hyphenated heads) survive.
  it.each([
    'the one-liner in the folder',
    'The harrier embedder sidecar heartbeat',
    'harrier embedder sidecar',
    'in-memory cache',
    'release trigger routine',
  ])('keeps real phrase %j', (phrase) => {
    expect(isLowQualityCompoundEntity(phrase)).toBe(false);
  });
});

describe('EI-10183 entity-quality gate — insert filtering', () => {
  const junk = { data: 'just before end of', entityType: 'COMPOUND', linkedMemoryIds: ['m1'] };
  const good = { data: 'harrier embedder sidecar', entityType: 'COMPOUND', linkedMemoryIds: ['m1'] };

  it('drops a junk COMPOUND entity — no canonical or vec write', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.insert([VEC], ['id-junk'], [junk]);
    expect(queries).toHaveLength(0);
  });

  it('still writes a good COMPOUND entity', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.insert([VEC], ['id-good'], [good]);
    expect(queries.some((q) => q.sql.includes('memory_canonical'))).toBe(true);
  });

  it('never filters PROPER/QUOTED entities (gate is COMPOUND-only)', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.insert([VEC], ['id-p'], [{ data: 'of the', entityType: 'PROPER', linkedMemoryIds: [] }]);
    expect(queries.some((q) => q.sql.includes('memory_canonical'))).toBe(true);
  });

  it('never filters a MEMORY-kind store even if the text looks fragmentary', async () => {
    const { store, queries } = makeStore('operator_memory_local'); // not *_entities
    await store.insert([VEC], ['id-m'], [{ data: 'just before end of' }]);
    expect(queries.some((q) => q.sql.includes('memory_canonical'))).toBe(true);
  });

  it('kill-switch PAPERCUSP_MEMORY_ENTITY_FILTER=off passes junk through', async () => {
    const prev = process.env.PAPERCUSP_MEMORY_ENTITY_FILTER;
    process.env.PAPERCUSP_MEMORY_ENTITY_FILTER = 'off';
    try {
      const { store, queries } = makeStore('operator_memory_local_entities');
      await store.insert([VEC], ['id-junk'], [junk]);
      expect(queries.some((q) => q.sql.includes('memory_canonical'))).toBe(true);
    } finally {
      if (prev === undefined) delete process.env.PAPERCUSP_MEMORY_ENTITY_FILTER;
      else process.env.PAPERCUSP_MEMORY_ENTITY_FILTER = prev;
    }
  });
});

describe('CanonicalVectorStore archived-state exclusion (P-016)', () => {
  it('search carries the archived exclusion condition alongside the kind clause', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    expect(queries[0].sql).toContain("c.state != 'archived'");
    // both conditions are present
    expect(queries[0].sql).toContain("NOT (c.payload ? 'entityType')");
  });

  it('list carries the archived exclusion in BOTH the page and count queries', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.list({ user_id: 'scope-a' }, 10);
    expect(queries).toHaveLength(2);
    for (const q of queries) {
      expect(q.sql).toContain("state != 'archived'");
    }
  });

  it('search with no filters still excludes archived rows', async () => {
    const { store, queries } = makeStore('operator_memory_openai');
    await store.search(VEC, 5);
    expect(queries[0].sql).toContain("c.state != 'archived'");
  });
});

describe('CanonicalVectorStore insert guards (GAP 9)', () => {
  it('rejects a wrong-DIMENSION vector and emits NO query (cfg dims=3)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    // embeddingModelDims is 3 (see makeStore); a 5-dim vector is corrupt and must
    // be refused BEFORE any INSERT runs — a wrong-width row would poison the
    // pgvector column for the whole model table.
    await expect(
      store.insert([[0.1, 0.2, 0.3, 0.4, 0.5]], ['id-1'], [{ user_id: 'u' }]),
    ).rejects.toThrow(/dim 5 !== expected 3/);
    // The throw is the WHOLE effect: no canonical upsert, no vec insert.
    expect(queries).toHaveLength(0);
  });

  it('rejects a length-MISMATCH between vectors/ids/payloads and emits NO query', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    // Two vectors but one id — a caller bug that would otherwise mis-pair rows.
    await expect(store.insert([VEC, VEC], ['only-one-id'], [{ user_id: 'u' }])).rejects.toThrow(
      /length mismatch/,
    );
    expect(queries).toHaveLength(0);
  });

  it('a correct-dim insert DOES emit both the canonical upsert and the vec insert', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.insert([VEC], ['id-1'], [{ user_id: 'u' }]);
    expect(queries.some((q) => q.sql.includes('memory_canonical'))).toBe(true);
    expect(queries.some((q) => q.sql.includes('memory_vec_local'))).toBe(true);
  });
});

describe('CanonicalVectorStore.deleteCol — the shared-table data-loss guard (GAP 9)', () => {
  it('REFUSES to delete with no userId scope: warns and emits NO delete', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // No setUserId() — an unscoped deleteCol would DELETE the whole shared
    // memory_canonical table (every user + harness). The guard must refuse.
    await store.deleteCol();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('refusing to wipe'));
    expect(queries.filter((q) => /DELETE/i.test(q.sql))).toHaveLength(0);
    warn.mockRestore();
  });

  it('after setUserId, scopes the delete to that user_id ONLY (no shared-table wipe)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await store.setUserId('scope-a');
    await store.deleteCol();
    // No refusal warning fired — the scope is set.
    expect(warn).not.toHaveBeenCalled();
    const deletes = queries.filter((q) => /DELETE/i.test(q.sql));
    expect(deletes).toHaveLength(1);
    // The delete is user_id-scoped — it can NEVER touch another tenant's rows.
    expect(deletes[0].sql).toContain("payload->>'user_id' = $1");
    expect(deletes[0].params).toEqual(['scope-a']);
    // And it is NOT an unscoped table wipe.
    expect(deletes[0].sql).not.toMatch(/DELETE FROM \S+\s*$/);
    warn.mockRestore();
  });
});

describe('CanonicalVectorStore lexicalSearch (WI-4214 embed-free fallback)', () => {
  /** Fake pool whose query resolves the given rows (payloads included). */
  function makeLexStore(rows: Array<{ id: string; payload: Record<string, unknown> }>) {
    const queries: CapturedQuery[] = [];
    const fakePool = {
      query: vi.fn(async (sql: string, params: unknown[] = []) => {
        queries.push({ sql, params });
        return { rows, rowCount: rows.length };
      }),
      on: () => {},
    };
    const { store } = makeStore('operator_memory_local');
    (store as unknown as { pool: unknown }).pool = fakePool;
    return { store, queries };
  }

  it('emits a token-ILIKE query over the canonical text with kind + archived + scope guards — and NO vec-table join', async () => {
    const { store, queries } = makeLexStore([]);
    await store.lexicalSearch('embed sidecar concurrency', 5, { user_id: 'scope-a' });
    expect(queries).toHaveLength(1);
    const q = queries[0];
    expect(q.sql).toContain("NOT (payload ? 'entityType')");
    expect(q.sql).toContain("state != 'archived'");
    expect(q.sql).toContain("payload->>'user_id' = $1");
    expect(q.sql).toContain("payload->>'data' ILIKE");
    // P-002: candidates come from all three weighted fields, one shared
    // param per token.
    expect(q.sql).toContain("payload->>'name' ILIKE");
    expect(q.sql).toContain("payload->>'description' ILIKE");
    expect(q.sql).not.toContain('memory_vec'); // never touches an embedding table
    expect(q.params).toContain('scope-a');
    expect(q.params).toContain('%embed%');
    expect(q.params).toContain('%sidecar%');
    expect(q.params).toContain('%concurrency%');
    // One param per token (not one per field) — the field conds reuse it.
    expect(q.params.filter((p) => p === '%embed%')).toHaveLength(1);
  });

  it('escapes the LIKE single-char wildcard in tokens (user_id matches literally)', async () => {
    const { store, queries } = makeLexStore([]);
    await store.lexicalSearch('user_id', 5);
    expect(queries[0].params).toContain('%user\\_id%');
  });

  it('a query with no usable tokens returns [] WITHOUT querying', async () => {
    const { store, queries } = makeLexStore([]);
    expect(await store.lexicalSearch('a b ??', 5)).toEqual([]);
    expect(queries).toHaveLength(0);
  });

  it('scores by field-weighted token overlap (normalized by tokens×3) and orders descending', async () => {
    const { store } = makeLexStore([
      { id: 'one-token', payload: { data: 'the embed pipeline', user_id: 'u' } },
      { id: 'both-tokens', payload: { data: 'embed sidecar is warm', user_id: 'u' } },
    ]);
    const out = await store.lexicalSearch('embed sidecar', 5, { user_id: 'u' });
    expect(out.map((r) => r.id)).toEqual(['both-tokens', 'one-token']);
    // Data-only hits weigh ×1 each, normalized by tokens×3 (claude-file parity).
    expect(out[0].score).toBeCloseTo(2 / 6);
    expect(out[1].score).toBeCloseTo(1 / 6);
  });

  it('weights name hits over description hits over data hits (P-002 parity)', async () => {
    const { store } = makeLexStore([
      { id: 'data-hit', payload: { data: 'embed embed embed everywhere', user_id: 'u' } },
      { id: 'name-hit', payload: { name: 'embed pipeline', data: 'unrelated', user_id: 'u' } },
      { id: 'desc-hit', payload: { description: 'about the embed path', data: 'unrelated', user_id: 'u' } },
    ]);
    const out = await store.lexicalSearch('embed', 5, { user_id: 'u' });
    expect(out.map((r) => r.id)).toEqual(['name-hit', 'desc-hit', 'data-hit']);
    expect(out[0].score).toBeCloseTo(3 / 3); // name ×3
    expect(out[1].score).toBeCloseTo(2 / 3); // description ×2
    expect(out[2].score).toBeCloseTo(1 / 3); // data ×1 — repeats don't stack
  });

  it('a token counts its BEST field only (name hit shadows data hit for the same token)', async () => {
    const { store } = makeLexStore([
      { id: 'both-fields', payload: { name: 'embed', data: 'embed sidecar', user_id: 'u' } },
    ]);
    const out = await store.lexicalSearch('embed sidecar', 5, { user_id: 'u' });
    // 'embed': name ×3 (data hit shadowed); 'sidecar': data ×1 → 4/(2×3).
    expect(out[0].score).toBeCloseTo(4 / 6);
  });

  it('slices to topK after scoring', async () => {
    const { store } = makeLexStore([
      { id: 'a', payload: { data: 'embed one' } },
      { id: 'b', payload: { data: 'embed two' } },
      { id: 'c', payload: { data: 'embed three' } },
    ]);
    expect(await store.lexicalSearch('embed', 2)).toHaveLength(2);
  });
});

describe('lexicalTokens', () => {
  it('lowercases, drops 1-char tokens, dedupes, caps at 12', () => {
    // min-len 2 (P-002 claude-file parity): 'of' survives, 'a'/'x' drop.
    expect(lexicalTokens('The EMBED embed of a x!')).toEqual(['the', 'embed', 'of']);
    const many = lexicalTokens(
      'alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november',
    );
    expect(many).toHaveLength(12);
  });

  it('keeps 2-char identifier tokens (pg, ui) — the P-002 min-len change', () => {
    expect(lexicalTokens('pg ui x')).toEqual(['pg', 'ui']);
  });

  it('emits compound identifiers WHOLE plus their subtokens (P-002 lexical-gap parity)', () => {
    expect(lexicalTokens('PAPERCUSP_MEMORY_TIMEOUT op-deadline')).toEqual([
      'papercusp_memory_timeout',
      'op-deadline',
      'papercusp',
      'memory',
      'timeout',
      'op',
      'deadline',
    ]);
  });

  it('whole tokens win the cap over subtoken fragments', () => {
    // 12 whole tokens fill the cap; the compound's subtokens must not evict any.
    const q = 'a_b c_d e_f g_h i_j k_l m_n o_p q_r s_t u_v w_x';
    const out = lexicalTokens(q);
    expect(out).toHaveLength(12);
    expect(out.every((t) => t.includes('_'))).toBe(true);
  });
});

describe('CanonicalVectorStore temporal-lite validity windows (P-002, migration 578)', () => {
  /** Fake pool with controllable rows + rowCount (validity columns included). */
  function makeRowStore(
    collectionName: string,
    rows: Array<Record<string, unknown>>,
    rowCount = rows.length,
  ) {
    const queries: CapturedQuery[] = [];
    const fakePool = {
      query: vi.fn(async (sql: string, params: unknown[] = []) => {
        queries.push({ sql, params });
        return { rows: rows.length > 0 ? rows : [{ n: 0 }], rowCount };
      }),
      on: () => {},
    };
    const { store } = makeStore(collectionName);
    (store as unknown as { pool: unknown }).pool = fakePool;
    return { store, queries };
  }

  it('search EXCLUDES closed-window rows by default and selects the validity columns', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    const q = queries[0];
    expect(q.sql).toContain('(c.invalid_at IS NULL OR c.invalid_at > now())');
    expect(q.sql).toContain('c.valid_at, c.invalid_at, c.superseded_by');
  });

  it('as_of / include_superseded are TEMPORAL controls, never payload-equality filters', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, {
      user_id: 'scope-a',
      as_of: '2026-07-01T00:00:00Z',
      include_superseded: false,
    });
    const q = queries[0];
    // Left in the filter loop these would silently match nothing.
    expect(q.sql).not.toContain("payload->>'as_of'");
    expect(q.sql).not.toContain("payload->>'include_superseded'");
    // Point-in-time window: valid_at (NULL ⇒ created_at) <= as_of < invalid_at.
    expect(q.sql).toContain('COALESCE(c.valid_at, c.created_at) <= $4::timestamptz');
    expect(q.sql).toContain('c.invalid_at > $4::timestamptz');
    expect(q.params).toContain('2026-07-01T00:00:00.000Z');
  });

  it('include_superseded drops the validity clause entirely', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a', include_superseded: true });
    expect(queries[0].sql).not.toContain('invalid_at IS NULL OR');
  });

  it('entity-kind search gets NO validity clause (mem0 lifecycle exempt) but temporal keys are still stripped', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.search(VEC, 5, { user_id: 'scope-a', as_of: '2026-07-01T00:00:00Z' });
    const q = queries[0];
    // No validity CLAUSE (the column list still selects the fields — unused).
    expect(q.sql).not.toContain('invalid_at IS NULL');
    expect(q.sql).not.toContain('COALESCE(c.valid_at');
    expect(q.sql).not.toContain("payload->>'as_of'");
  });

  it('list applies the default exclusion to the page AND the count', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.list({ user_id: 'scope-a' }, 10);
    expect(queries).toHaveLength(2);
    for (const q of queries) {
      expect(q.sql).toContain('(invalid_at IS NULL OR invalid_at > now())');
    }
  });

  it('lexicalSearch (the degraded fallback) excludes closed-window rows too', async () => {
    const { store, queries } = makeRowStore('operator_memory_local', []);
    await store.lexicalSearch('embed sidecar', 5, { user_id: 'scope-a' });
    expect(queries[0].sql).toContain('(invalid_at IS NULL OR invalid_at > now())');
  });

  it('a closed-window row surfaces validity { status: superseded } in its result payload', async () => {
    const { store } = makeRowStore('operator_memory_local', [
      {
        id: 'old-fact',
        payload: { data: 'embed default is gemma', user_id: 'u' },
        valid_at: null,
        invalid_at: '2026-07-01T00:00:00Z',
        superseded_by: 'new-fact',
      },
    ]);
    const out = await store.lexicalSearch('embed gemma', 5, {
      user_id: 'u',
      include_superseded: true,
    });
    expect(out[0].payload.validity).toEqual({
      valid_at: null,
      invalid_at: '2026-07-01T00:00:00Z',
      superseded_by: 'new-fact',
      status: 'superseded',
    });
  });

  it('all-NULL validity rows keep their payload byte-identical (no validity key attached)', async () => {
    const payload = { data: 'embed default is harrier', user_id: 'u' };
    const { store } = makeRowStore('operator_memory_local', [
      { id: 'current-fact', payload, valid_at: null, invalid_at: null, superseded_by: null },
    ]);
    const out = await store.lexicalSearch('embed harrier', 5, { user_id: 'u' });
    expect(out[0].payload).toBe(payload);
  });

  it('invalidate emits a VEC-SAFE column UPDATE guarded to OPEN memory rows (first-wins)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    const closed = await store.invalidate('old-fact', {
      supersededBy: 'new-fact',
      at: '2026-07-12T00:00:00Z',
    });
    const q = queries[0];
    expect(q.sql).toContain('SET invalid_at = COALESCE($2::timestamptz, now())');
    expect(q.sql).toContain('superseded_by = $3::uuid');
    expect(q.sql).toContain("NOT (payload ? 'entityType')");
    expect(q.sql).toContain('AND invalid_at IS NULL');
    expect(q.sql).not.toContain('memory_vec'); // never touches an embedding table
    expect(q.params).toEqual(['old-fact', '2026-07-12T00:00:00Z', 'new-fact']);
    // makeStore's fake reports rowCount 0 — already-closed (or unknown) id → false.
    expect(closed).toBe(false);
  });

  it('invalidate returns true when an open row was closed', async () => {
    const { store } = makeRowStore('operator_memory_local', [], 1);
    expect(await store.invalidate('open-fact')).toBe(true);
  });

  it('insert strips the read-side validity key from stored payloads (echo defense)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.insert(
      [VEC],
      ['id-1'],
      [{ data: 'x', validity: { status: 'current' } }],
    );
    const stored = String(queries[0].params[1]);
    expect(stored).toContain('"data":"x"');
    expect(stored).not.toContain('validity');
  });

  it('updatePayload strips validity from the patch (echo defense)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.updatePayload('id-1', { kind: 'note', validity: { status: 'superseded' } });
    const patch = JSON.parse(String(queries[0].params[1]));
    expect(patch).toEqual({ kind: 'note' });
  });
});

describe('splitTemporalControls', () => {
  it('defaults: no filters → no asOf, includeSuperseded false, rest undefined', () => {
    expect(splitTemporalControls(undefined)).toEqual({
      temporal: { includeSuperseded: false },
      rest: undefined,
    });
  });

  it('extracts as_of (ISO-normalized) and include_superseded, leaving the rest intact', () => {
    const { temporal, rest } = splitTemporalControls({
      user_id: 'u',
      as_of: '2026-07-01T00:00:00Z',
      include_superseded: 'true',
    });
    expect(temporal).toEqual({ asOf: '2026-07-01T00:00:00.000Z', includeSuperseded: true });
    expect(rest).toEqual({ user_id: 'u' });
  });

  it('accepts the truthy forms true/1/"1"/"true" and nothing else', () => {
    for (const v of [true, 1, '1', 'true']) {
      expect(splitTemporalControls({ include_superseded: v }).temporal.includeSuperseded).toBe(true);
    }
    for (const v of [false, 0, '0', 'false', 'yes', undefined]) {
      expect(splitTemporalControls({ include_superseded: v }).temporal.includeSuperseded).toBe(false);
    }
  });

  it('an unparseable as_of is dropped (never a NaN timestamp in SQL)', () => {
    expect(splitTemporalControls({ as_of: 'not-a-date' }).temporal.asOf).toBeUndefined();
  });
});

describe('foldValidity', () => {
  const OPEN = { valid_at: null, invalid_at: null, superseded_by: null };

  it('returns the payload UNCHANGED (same reference) for a trivial row with no as_of', () => {
    const payload = { data: 'x' };
    expect(foldValidity(payload, OPEN, { includeSuperseded: false })).toBe(payload);
  });

  it('a closed window folds status superseded against now()', () => {
    const out = foldValidity(
      { data: 'x' },
      { valid_at: null, invalid_at: '2020-01-01T00:00:00Z', superseded_by: 'y' },
      { includeSuperseded: true },
    );
    expect((out.validity as { status: string }).status).toBe('superseded');
  });

  it('status is computed against as_of when given: current before the close, superseded after', () => {
    const row = { valid_at: null, invalid_at: '2026-08-01T00:00:00Z', superseded_by: null };
    const before = foldValidity({}, row, { asOf: '2026-07-01T00:00:00.000Z', includeSuperseded: true });
    const after = foldValidity({}, row, { asOf: '2026-09-01T00:00:00.000Z', includeSuperseded: true });
    expect((before.validity as { status: string }).status).toBe('current');
    expect((after.validity as { status: string }).status).toBe('superseded');
  });

  it('an as_of read attaches validity even to all-NULL rows (the caller asked for time context)', () => {
    const out = foldValidity({ data: 'x' }, OPEN, { asOf: '2026-07-01T00:00:00.000Z', includeSuperseded: false });
    expect(out.validity).toEqual({
      valid_at: null,
      invalid_at: null,
      superseded_by: null,
      status: 'current',
    });
  });
});
