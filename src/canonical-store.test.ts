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
  LEXICAL_QUERY_CONCURRENCY,
  isLowQualityCompoundEntity,
  lexicalTokens,
  splitTemporalControls,
  foldValidity,
  type CanonicalStoreConfig,
} from './canonical-store';

type CapturedQuery = { sql: string; params: unknown[] };

/** BEGIN / COMMIT / ROLLBACK / SET LOCAL — transaction scaffolding, not data. */
const CONTROL_SQL = /^\s*(BEGIN|COMMIT|ROLLBACK|SET LOCAL)\b/i;

function makeStore(collectionName: string): {
  store: CanonicalVectorStore;
  /** Data queries only — the SQL these tests pin. */
  queries: CapturedQuery[];
  /** Every statement incl. the transaction scaffolding search() wraps itself in. */
  allQueries: CapturedQuery[];
} {
  const queries: CapturedQuery[] = [];
  const allQueries: CapturedQuery[] = [];
  const record = async (sql: string, params: unknown[] = []) => {
    allQueries.push({ sql, params });
    if (!CONTROL_SQL.test(sql)) queries.push({ sql, params });
    return { rows: [{ n: 0 }], rowCount: 0 };
  };
  const fakePool = {
    query: vi.fn(record),
    // search() checks out a dedicated connection so it can SET LOCAL the
    // HNSW iterative-scan GUC inside a transaction (EI-19386910150607131).
    connect: vi.fn(async () => ({ query: vi.fn(record), release: () => {} })),
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
  return { store, queries, allQueries };
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
    expect(queries[0].sql).toContain("WHERE v.row_kind = 'memory'");
    expect(queries[0].sql).toContain("NOT (c.payload ? 'entityType')");
  });

  /**
   * P-004 (plan memory-vector-entity-index-split-2026-09-02). The discriminator
   * has to sit on the `memory_vec_*` side for migration 1093's partial
   * `_hnsw_memory_idx` to be selectable at all: a predicate on the JOINED
   * canonical row cannot be applied until after the approximate index scan has
   * already chosen its candidates. Measured on the live store, moving it took a
   * scoped `LIMIT 12` from 16,621 index rows / 104,633 buffers (full index) to
   * 39 / 807 (partial index).
   *
   * These are SQL-shape assertions because the cost of getting it wrong is
   * silent: the query keeps returning correct rows either way — it just quietly
   * orders over ~91% entity vectors again.
   */
  it('memory search pushes the discriminator onto the VEC side (partial-index predicate)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    expect(queries[0].sql).toContain("v.row_kind = 'memory'");
  });

  it('entity search pushes its own vec-side discriminator', async () => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.search(VEC, 5);
    expect(queries[0].sql).toContain("v.row_kind = 'entity'");
  });

  /**
   * The vec-side clause buys the index; the canonical-side clause is the
   * correctness backstop, and it is free (measured identical plans and buffer
   * counts with and without it). `payload` remains the source of truth for what
   * a row IS, so if the trigger-maintained mirror ever drifted, a false
   * `row_kind = 'memory'` is caught here instead of leaking an entity row into
   * memory recall. Dropping either one is a real regression, so pin both.
   */
  it('retains the canonical-side predicate as a drift backstop alongside the vec-side one', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    expect(queries[0].sql).toContain("v.row_kind = 'memory'");
    expect(queries[0].sql).toContain("NOT (c.payload ? 'entityType')");
  });

  /**
   * Guards the other direction: `row_kind` exists on `memory_canonical` too, so
   * "simplifying" this to `c.row_kind` would typecheck, keep every result
   * correct, and silently give the partial index back.
   */
  it('does not express the vec-side discriminator against the canonical alias', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, { user_id: 'scope-a' });
    expect(queries[0].sql).not.toContain("c.row_kind = 'memory'");
  });

  /** list()/count read memory_canonical alone — there is no `v` to qualify. */
  it('does not leak a vec-side predicate into the non-joining list/count reads', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.list({ user_id: 'scope-a' }, 10);
    expect(queries).toHaveLength(2);
    for (const q of queries) expect(q.sql).not.toContain('row_kind');
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
    'discovering files/scripts', // sentence-leading action residue
    "the fleet's release gate", // possessive sentence-local description
    'a genuine removal reds', // vague modifier + nominalized/plural residue
    'a TEST against the tree', // internal clause marker
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
    "user's guide", // short possessive lexical phrase is still reusable
  ])('keeps real phrase %j', (phrase) => {
    expect(isLowQualityCompoundEntity(phrase)).toBe(false);
  });
});

describe('EI-10183 entity-quality gate — insert filtering', () => {
  const junk = { data: 'just before end of', entityType: 'COMPOUND', linkedMemoryIds: ['m1'] };
  const good = { data: 'harrier embedder sidecar', entityType: 'COMPOUND', linkedMemoryIds: ['m1'] };

  it.each([
    'discovering files/scripts',
    "the fleet's release gate",
    'a genuine removal reds',
    'a TEST against the tree',
  ])('drops observed sentence residue %j before either table write', async (data) => {
    const { store, queries } = makeStore('operator_memory_local_entities');
    await store.insert([VEC], [`id-${data.slice(0, 4)}`], [{ data, entityType: 'COMPOUND' }]);
    expect(queries).toHaveLength(0);
  });

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

describe('CanonicalVectorStore update', () => {
  it('merges text payload over the existing row so arbitrary metadata survives', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.update('id-1', VEC, {
      data: 'replacement text',
      hash: 'replacement-hash',
      updatedAt: '2026-08-21T12:00:00.000Z',
    });

    expect(queries).toHaveLength(2);
    const canonical = queries[0];
    expect(canonical.sql).toContain('UPDATE harness_shared.memory_canonical');
    expect(canonical.sql).toContain('SET payload = payload || $2::jsonb');
    expect(canonical.sql).toContain('WHERE id = $1');
    expect(JSON.parse(String(canonical.params[1]))).toEqual({
      data: 'replacement text',
      hash: 'replacement-hash',
      updatedAt: '2026-08-21T12:00:00.000Z',
    });
    // The separate vector upsert still tracks the re-embedded replacement text.
    expect(queries[1].sql).toContain('memory_vec_local');
  });

  it('rejects a wrong-DIMENSION vector before touching either table', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await expect(
      store.update('id-1', [0.1, 0.2, 0.3, 0.4], { data: 'replacement text' }),
    ).rejects.toThrow(/dim 4 !== expected 3/);
    expect(queries).toHaveLength(0);
  });

  it('does not persist the read-side validity fold during a text update', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.update('id-1', VEC, {
      data: 'replacement text',
      validity: { status: 'superseded' },
    });
    expect(JSON.parse(String(queries[0].params[1]))).toEqual({ data: 'replacement text' });
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
  /** Fake pool whose query resolves the given rows. `lex_raw` is the score Postgres
   *  computes (the CASE-sum) — the leg only normalizes it, so tests supply it. */
  function makeLexStore(rows: Array<{ id: string; payload: Record<string, unknown>; lex_raw?: number }>) {
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
    // WI-6966: the three weighted fields are PROJECTED ONCE each in the fenced
    // subquery and matched/scored via their aliases — never re-extracted per token
    // (each `payload->>'…'` detoasts the whole jsonb; per-token repetition was 28x
    // the buffers). P-002: all three fields score, one shared param per token.
    expect(q.sql).toContain("payload->>'name' AS nm");
    expect(q.sql).toContain("payload->>'description' AS ds");
    expect(q.sql).toContain("payload->>'data' AS dt");
    expect(q.sql).toContain('dt ILIKE');
    expect(q.sql).toContain('nm ILIKE');
    expect(q.sql).toContain('ds ILIKE');
    // Exactly one extraction per field, however many tokens the query has.
    expect(q.sql.match(/payload->>'name'/g)).toHaveLength(1);
    expect(q.sql.match(/payload->>'description'/g)).toHaveLength(1);
    expect(q.sql.match(/payload->>'data'/g)).toHaveLength(1);
    // The optimisation fence that stops the subquery being flattened back (which
    // would re-inline the per-token extraction and silently restore the old cost).
    expect(q.sql).toContain('OFFSET 0');
    // EI-21855287664853515: rank carries ids/scores only; the JSONB payload is
    // joined after the top-K LIMIT so broad matches cannot retain every payload
    // in the ranking tuples.
    expect(q.sql).toContain('WITH ranked AS MATERIALIZED');
    expect(q.sql).toContain('JOIN harness_shared.memory_canonical c ON c.id = ranked.id');
    expect(q.sql).toContain('SELECT ranked.id, c.payload');
    expect(q.sql).not.toContain('SELECT id, payload, valid_at');
    expect(q.sql).not.toContain('memory_vec'); // never touches an embedding table
    expect(q.params).toContain('scope-a');
    expect(q.params).toContain('%embed%');
    expect(q.params).toContain('%sidecar%');
    expect(q.params).toContain('%concurrency%');
    // One param per token (not one per field) — the field CASEs reuse it.
    expect(q.params.filter((p) => p === '%embed%')).toHaveLength(1);
  });

  it('scores IN SQL with the field weights, and RANKS BY THAT SCORE — never by recency', async () => {
    // ⚠ The regression this pins (plan memory-pg-lexical-own-injection-2026-07-13):
    // the leg used to pull `ORDER BY created_at DESC LIMIT max(topK*5, 50)` and score
    // the survivors in JS — truncating the pool BEFORE scoring, so the best lexical
    // match was silently dropped whenever it lost a RECENCY race to rows that merely
    // shared a common word. Ranking must be by score over the WHOLE matching set. The
    // scoring SEMANTICS are verified against real Postgres in
    // packages/operator-core/lib/memory/canonical-lexical-search.integration.test.ts.
    const { store, queries } = makeLexStore([]);
    await store.lexicalSearch('embed sidecar', 5, { user_id: 'u' });
    const { sql, params } = queries[0];
    // The weights live in SQL: name ×3 > description ×2 > data ×1, per token.
    expect(sql).toContain('CASE WHEN nm ILIKE');
    expect(sql).toContain('THEN 3 WHEN');
    expect(sql).toContain('THEN 2 WHEN');
    expect(sql).toContain('THEN 1 ELSE 0 END');
    // Ordered by that score; only scoring rows come back.
    expect(sql).toContain('ORDER BY lex_raw DESC');
    // ⚠ WI-6966 — there must be NO outer `lex_raw > 0`. It is provably redundant
    // given the token match predicate (a row scores > 0 iff ≥1 token hits ≥1 field,
    // which is exactly that OR), and the planner pushes it back down into the SAME
    // scan filter, so re-adding it re-evaluates the whole score chain per row on top
    // of the match chain — measured 761 ms vs 430 ms at 24 tokens. Rows that score 0
    // are already excluded by the match predicate below.
    expect(sql).not.toContain('lex_raw > 0');
    expect(sql).toMatch(/WHERE \(nm ILIKE/);
    // NOT a recency-ranked candidate pull, and no candidate cap above topK.
    expect(sql).not.toContain('ORDER BY created_at DESC');
    expect(params.at(-1)).toBe(5); // the only LIMIT is topK itself
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

  it('normalizes the SQL score by tokens×3 (claude-file parity)', async () => {
    // Postgres computes lex_raw and returns the rows already ranked; the leg's
    // remaining job is the 0..1 normalization. (Weights/ordering: see the real-PG
    // integration test named in the comment above.)
    const { store } = makeLexStore([
      { id: 'name-and-data', payload: { data: 'x', user_id: 'u' }, lex_raw: 4 },
      { id: 'data-only', payload: { data: 'y', user_id: 'u' }, lex_raw: 1 },
    ]);
    const out = await store.lexicalSearch('embed sidecar', 5, { user_id: 'u' });
    expect(out.map((r) => r.id)).toEqual(['name-and-data', 'data-only']);
    // 2 tokens → denominator 2×3.
    expect(out[0].score).toBeCloseTo(4 / 6);
    expect(out[1].score).toBeCloseTo(1 / 6);
  });

  it("passes topK to SQL as the LIMIT (the slice is the database's, not ours)", async () => {
    const { store, queries } = makeLexStore([]);
    await store.lexicalSearch('embed', 2);
    expect(queries[0].sql).toMatch(/LIMIT \$\d+/);
    expect(queries[0].params.at(-1)).toBe(2);
  });

  it('does not admit more than four payload-bearing lexical queries at once', async () => {
    let active = 0;
    let peak = 0;
    const queries: CapturedQuery[] = [];
    const fakePool = {
      query: vi.fn(async (sql: string, params: unknown[] = []) => {
        queries.push({ sql, params });
        active += 1;
        peak = Math.max(peak, active);
        await new Promise<void>((resolve) => setTimeout(resolve, 5));
        active -= 1;
        return { rows: [], rowCount: 0 };
      }),
      on: () => {},
    };
    const { store } = makeStore('operator_memory_local');
    (store as unknown as { pool: unknown }).pool = fakePool;

    await Promise.all(
      Array.from({ length: 10 }, (_, i) =>
        store.lexicalSearch(`query-${i}`, 1, { user_id: `scope-${i}` }),
      ),
    );

    expect(queries).toHaveLength(10);
    expect(peak).toBeLessThanOrEqual(LEXICAL_QUERY_CONCURRENCY);
  });
});

describe('lexicalTokens', () => {
  it('lowercases, drops 1-char tokens, dedupes, caps at 32', () => {
    // min-len 2 (P-002 claude-file parity): 'of' survives, 'a'/'x' drop.
    expect(lexicalTokens('The EMBED embed of a x!')).toEqual(['the', 'embed', 'of']);
    const many = lexicalTokens(Array.from({ length: 50 }, (_, i) => `word${i}`).join(' '));
    expect(many).toHaveLength(32);
  });

  it('does NOT truncate a real natural-language query — the cap is a safety bound, not a relevance knob', () => {
    // ⚠ THE REGRESSION THIS PINS (plan memory-pg-lexical-own-injection-2026-07-13 P-006):
    // the cap was 12 while real recall queries run ~20 tokens (gold-set v1: p50 10, p95 24,
    // max 28). It silently DISCARDED the tail of every long query and then normalized the
    // score by the truncated token count — costing lexical-gap (paraphrase) MRR 0.432 vs
    // 0.546 uncapped, the entire hybrid-pg-vs-hybrid regression. Invisible on exact-identifier
    // queries (short → MRR 1.000 either way), which is what disguised it as a ranking bug.
    // A 28-token query must survive INTACT; if this fails, someone lowered the cap.
    const realQuery =
      'pushing simulated keypresses at a live text-mode console wired into the fleet common ' +
      'backend could accidentally kick off a real worker the rest of the team relies on';
    const tokens = lexicalTokens(realQuery);
    expect(tokens.length).toBeGreaterThanOrEqual(24); // nothing dropped at p95 length
    expect(tokens).toContain('keypresses'); // …including the discriminative tail
    expect(tokens).toContain('worker');
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
    // 32 whole compounds fill the cap; their subtokens must not evict any.
    const q = Array.from({ length: 32 }, (_, i) => `a${i}_b${i}`).join(' ');
    const out = lexicalTokens(q);
    expect(out).toHaveLength(32);
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

  it('as_of remains authoritative when include_superseded is also true', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    await store.search(VEC, 5, {
      user_id: 'scope-a',
      as_of: '2026-07-01T00:00:00Z',
      include_superseded: true,
    });
    expect(queries[0].sql).toContain('COALESCE(c.valid_at, c.created_at) <= $4::timestamptz');
    expect(queries[0].sql).toContain('c.invalid_at > $4::timestamptz');
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

  it('an unparseable as_of fails closed instead of silently becoming a current read', () => {
    expect(() => splitTemporalControls({ as_of: 'not-a-date' })).toThrow('as_of must be a parseable timestamp');
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

/**
 * HNSW post-filter under-retrieval (EI-19386910150607131).
 *
 * The scope predicate lives on the JOINED canonical table, so it can only be
 * applied AFTER the approximate index scan has picked its candidates. Without
 * `hnsw.iterative_scan` the scan stops at ef_search (default 40) candidates
 * and a scoped pull silently returns FEWER rows than LIMIT — measured on the
 * live store: 0 rows for LIMIT 12 against 18,510 eligible rows.
 *
 * These pin the mechanism (the GUC is set, in a transaction, BEFORE the
 * search, and is not required for correctness on an older pgvector). The
 * retrieval-count proof itself is the live-store measurement on the item.
 */
describe('CanonicalVectorStore HNSW iterative scan', () => {
  it('sets hnsw.iterative_scan in a transaction BEFORE the search runs', async () => {
    const { store, allQueries } = makeStore('operator_memory_local');
    await store.search(VEC, 12, { user_id: 'harness:papercusp' });

    const searchIdx = allQueries.findIndex((q) => q.sql.includes('ORDER BY v.vector'));
    expect(searchIdx).toBeGreaterThanOrEqual(0);
    // The SET that matters is the one guarding THIS search, not the one-off
    // support probe that ran before it.
    const setIdx = allQueries.findIndex(
      (q, i) => i < searchIdx && /SET LOCAL hnsw\.iterative_scan/i.test(q.sql) && i > 0,
    );
    expect(setIdx).toBeGreaterThanOrEqual(0);
    // A SET LOCAL issued after the query would be a silent no-op.
    expect(setIdx).toBeLessThan(searchIdx);
    // ...and it must be inside a transaction, or SET LOCAL does nothing.
    expect(allQueries.slice(0, setIdx).some((q) => /^\s*BEGIN/i.test(q.sql))).toBe(true);
    expect(allQueries.slice(searchIdx).some((q) => /^\s*COMMIT/i.test(q.sql))).toBe(true);
  });

  it('still returns rows when the server has no iterative_scan (pgvector < 0.8)', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    const pool = (store as unknown as { pool: { connect: unknown } }).pool;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Old pgvector rejects the GUC outright; search must fail OPEN, not throw.
    pool.connect = vi.fn(async () => ({
      query: vi.fn(async (sql: string) => {
        if (/SET LOCAL hnsw\.iterative_scan/i.test(sql)) {
          throw new Error('unrecognized configuration parameter "hnsw.iterative_scan"');
        }
        return { rows: [], rowCount: 0 };
      }),
      release: () => {},
    }));

    // Resolves rather than throwing — the GUC is an optimization, not a
    // correctness requirement, so an old server degrades to the old behavior.
    await expect(store.search(VEC, 12, { user_id: 'harness:papercusp' })).resolves.toBeInstanceOf(
      Array,
    );
    // The server capability answer is permanent, so a second search reuses
    // the negative probe rather than repeatedly probing a known-old server.
    await expect(store.search(VEC, 12, { user_id: 'harness:papercusp' })).resolves.toBeInstanceOf(
      Array,
    );
    expect(pool.connect).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledTimes(1);
    // ...and the search itself still went out, on the plain pool path.
    expect(queries.filter((q) => q.sql.includes('ORDER BY v.vector'))).toHaveLength(2);
    warn.mockRestore();
  });

  it('does not permanently cache a transient probe failure', async () => {
    const { store, queries } = makeStore('operator_memory_local');
    const pool = (store as unknown as { pool: { connect: unknown } }).pool;
    let probeBegins = 0;
    pool.connect = vi.fn(async () => ({
      query: vi.fn(async (sql: string) => {
        if (/^BEGIN$/i.test(sql.trim())) probeBegins += 1;
        if (/SET LOCAL hnsw\.iterative_scan/i.test(sql) && probeBegins === 1) {
          throw Object.assign(new Error('connection reset by peer'), { code: '08006' });
        }
        return { rows: [], rowCount: 0 };
      }),
      release: () => {},
    }));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(store.search(VEC, 12, { user_id: 'harness:papercusp' })).resolves.toBeInstanceOf(
      Array,
    );
    await expect(store.search(VEC, 12, { user_id: 'harness:papercusp' })).resolves.toBeInstanceOf(
      Array,
    );

    expect(probeBegins).toBe(2);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('transiently'));
    warn.mockRestore();
  });

  it('rolls back rather than leaking an aborted connection when the search throws', async () => {
    const { store } = makeStore('operator_memory_local');
    const seen: string[] = [];
    let released = false;
    const pool = (store as unknown as { pool: { connect: unknown } }).pool;
    pool.connect = vi.fn(async () => ({
      query: vi.fn(async (sql: string) => {
        seen.push(sql.trim());
        if (sql.includes('ORDER BY v.vector')) throw new Error('boom');
        return { rows: [], rowCount: 0 };
      }),
      release: () => {
        released = true;
      },
    }));

    await expect(store.search(VEC, 12, { user_id: 'harness:papercusp' })).rejects.toThrow('boom');
    expect(seen.some((s) => /^ROLLBACK/i.test(s))).toBe(true);
    expect(released).toBe(true);
  });
});
