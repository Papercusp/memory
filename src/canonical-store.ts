/**
 * CanonicalVectorStore — mem0 VectorStore impl backed by our own
 * `memory_canonical` + `memory_vec_<model>` tables (migration 081).
 *
 * One canonical row per fact (text + metadata + scope keys). Each
 * embedder mode (openai, local) gets its own vec table joined by
 * `memory_id`. The text doesn't move when you switch modes — only
 * which vec table is read for recall changes. Re-embedding into the
 * other model is an INSERT into the other vec table.
 *
 * This replaces the mem0 PGVector adapter's "one table per collection,
 * with duplicated payload across collections" model. The interface
 * stays unchanged so every existing caller (memory:remember/search/
 * list/forget/update, the user-memory GET handler, injection.ts)
 * works without changes — they go through mem0's Memory class, which
 * delegates to whichever VectorStore is registered.
 *
 * Registration happens at construction time in mem0-client.ts via a
 * runtime patch of mem0's VectorStoreFactory.create, since the OSS
 * factory uses a hard-coded switch (no plugin hook).
 *
 * Filter semantics: mem0 calls insert/search/list/etc. with `filters`
 * shaped { user_id: string, agent_id?, run_id?, ... }. We post-filter
 * each value against `payload->>'<key>'`. mem0 also processes
 * AND/OR/NOT into $or/$not shapes but its PGVector adapter ignores
 * those — we do the same here (any non-string value other than known
 * payload-key strings is silently skipped). See the memory-harness-
 * scope-2026-05-24 plan for the audit that established this.
 *
 * Store-kind segregation (EI-366): mem0's Memory class creates a SECOND
 * vector store for entity linking via `getEntityStore()`, distinguished
 * only by a `<collection>_entities` collectionName. This store ignores
 * collectionName for table selection (canonical rows are shared across
 * embedder modes by design), which used to dump entity fragments
 * ({ data, entityType, linkedMemoryIds }) into the same pool as real
 * memories — 84% of the store was COMPOUND/PROPER junk surfacing in
 * recall. The discriminator is payload shape: mem0 entity payloads
 * ALWAYS carry `entityType`; memory payloads never do. search()/list()
 * filter on it per store kind, so both kinds share the physical tables
 * but never each other's result sets (and the pre-fix junk rows are
 * segregated retroactively, no backfill needed).
 */

import { Pool as PgPool } from 'pg';

interface VectorStoreResult {
  id: string;
  payload: Record<string, unknown>;
  score?: number;
}

interface SearchFilters {
  user_id?: string;
  agent_id?: string;
  run_id?: string;
  [key: string]: unknown;
}

export interface CanonicalStoreConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  dbname: string;
  /** Schema holding `memory_canonical` + the vec tables (host-defined). */
  schema: string;
  /** Not used for table selection (the canonical row carries scope in
   *  payload), but a `*_entities` suffix marks this instance as mem0's
   *  ENTITY store — its search/list see only entity rows, every other
   *  instance sees only memory rows. */
  collectionName?: string;
  /** Which model's vec table this instance reads/writes. */
  vecTable: 'memory_vec_openai' | 'memory_vec_local' | 'memory_vec_gemma' | 'memory_vec_harrier';
  /** Sanity check — refuses to insert vectors with the wrong length. */
  embeddingModelDims: number;
}

function safeKey(k: string): string {
  return k.replace(/[^a-zA-Z0-9_]/g, '');
}

/** Cap on query tokens for the lexical fallback — bounds the OR-chain. */
const LEXICAL_MAX_TOKENS = 8;

/**
 * Tokenize a query for the lexical fallback: lowercase, split on anything
 * outside [a-z0-9_-], drop tokens under 3 chars (stopword-ish noise), dedupe,
 * cap. Exported for `lexicalSearch` scoring parity and its tests.
 */
export function lexicalTokens(query: string): string[] {
  return [
    ...new Set(
      query
        .toLowerCase()
        .split(/[^a-z0-9_-]+/)
        .filter((t) => t.length >= 3),
    ),
  ].slice(0, LEXICAL_MAX_TOKENS);
}

function toVectorLiteral(v: number[]): string {
  return `[${v.join(',')}]`;
}

/**
 * The store-kind discriminator clause. Entity rows are the ones mem0's
 * entity linking writes — their payload always carries `entityType`
 * (see Memory._linkEntitiesForMemory in mem0ai/oss); real memory
 * payloads never do.
 */
function storeKindCond(alias: string, kind: 'memory' | 'entity'): string {
  const has = `${alias}payload ? 'entityType'`;
  return kind === 'entity' ? has : `NOT (${has})`;
}

/**
 * Temporal-lite validity (memory-temporal-lite-validity-windows-2026-07-11
 * P-002/P-006, migration 578). mem0 forwards our tool-layer read options as
 * FILTER keys, but `as_of` / `include_superseded` are TEMPORAL controls, not
 * payload-equality filters — split them out before the payload post-filter
 * loop (left in, they'd silently match nothing: no payload carries them).
 */
interface TemporalControls {
  /** Point-in-time read: valid_at (NULL ⇒ created_at) <= as_of < invalid_at. */
  asOf?: string;
  /** Opt-in: include rows whose validity window has closed. */
  includeSuperseded: boolean;
}

function splitTemporalControls(filters?: SearchFilters): {
  temporal: TemporalControls;
  rest: SearchFilters | undefined;
} {
  if (!filters) return { temporal: { includeSuperseded: false }, rest: undefined };
  const { as_of, include_superseded, ...rest } = filters as {
    as_of?: unknown;
    include_superseded?: unknown;
  } & SearchFilters;
  const asOfMs = typeof as_of === 'string' || typeof as_of === 'number' ? new Date(as_of).getTime() : NaN;
  return {
    temporal: {
      ...(Number.isFinite(asOfMs) ? { asOf: new Date(asOfMs).toISOString() } : {}),
      includeSuperseded:
        include_superseded === true ||
        include_superseded === 1 ||
        include_superseded === '1' ||
        include_superseded === 'true',
    },
    rest,
  };
}

/**
 * The default current-rows clause (memory kind only — entity rows are mem0's
 * lifecycle, exempt by design). With `asOf` it becomes the point-in-time
 * window; `includeSuperseded` drops it entirely. Returns the SQL condition
 * (may push a param) or null for no condition.
 */
function validityCond(
  alias: string,
  temporal: TemporalControls,
  params: unknown[],
  nextIdx: () => number,
): string | null {
  if (temporal.asOf !== undefined) {
    const i = nextIdx();
    params.push(temporal.asOf);
    return `COALESCE(${alias}valid_at, ${alias}created_at) <= $${i}::timestamptz AND (${alias}invalid_at IS NULL OR ${alias}invalid_at > $${i}::timestamptz)`;
  }
  if (temporal.includeSuperseded) return null;
  return `(${alias}invalid_at IS NULL OR ${alias}invalid_at > now())`;
}

/**
 * Fold the validity window into a result row's payload so it survives mem0's
 * payload→metadata mapping (unknown payload keys land in result metadata —
 * the same ride `kind` takes). Attached ONLY when the row carries a
 * non-trivial window (or a point-in-time read asked): the 9.8k pre-migration
 * rows are all-NULL ⇒ trivially 'current', and attaching nothing keeps their
 * result shape byte-identical.
 */
function foldValidity(
  payload: Record<string, unknown>,
  row: { valid_at?: unknown; invalid_at?: unknown; superseded_by?: unknown },
  temporal: TemporalControls,
): Record<string, unknown> {
  const validAt = row.valid_at ?? null;
  const invalidAt = row.invalid_at ?? null;
  const supersededBy = row.superseded_by ?? null;
  if (validAt === null && invalidAt === null && supersededBy === null && temporal.asOf === undefined) {
    return payload;
  }
  const refMs = temporal.asOf !== undefined ? new Date(temporal.asOf).getTime() : Date.now();
  const invalidMs = invalidAt !== null ? new Date(String(invalidAt)).getTime() : null;
  return {
    ...payload,
    validity: {
      valid_at: validAt,
      invalid_at: invalidAt,
      superseded_by: supersededBy,
      status: invalidMs !== null && invalidMs <= refMs ? 'superseded' : 'current',
    },
  };
}

export class CanonicalVectorStore {
  private cfg: CanonicalStoreConfig;
  /** 'entity' when mem0 constructed this instance as its entity store. */
  private readonly storeKind: 'memory' | 'entity';
  private userId = '';
  // A small Pool, not a single Client: concurrent callers (parallel
  // memory writes, the bench's concurrent seeding) interleave queries,
  // which a lone pg.Client only tolerates via a deprecated internal
  // queue (removed in pg@9). Connection errors surface per query and
  // retry naturally — no poison-cache to manage.
  private pool: PgPool | null = null;

  constructor(config: CanonicalStoreConfig) {
    this.cfg = config;
    this.storeKind = config.collectionName?.endsWith('_entities') ? 'entity' : 'memory';
  }

  private async getClient(): Promise<PgPool> {
    if (!this.pool) {
      this.pool = new PgPool({
        host: this.cfg.host,
        port: this.cfg.port,
        user: this.cfg.user,
        password: this.cfg.password,
        database: this.cfg.dbname,
        max: 5,
      });
      // Don't let a dropped idle connection crash the process — the pool
      // replaces it on the next query.
      this.pool.on('error', () => {});
    }
    return this.pool;
  }

  async initialize(): Promise<void> {
    // Schema lives in migration 081 (libs/papercusp/libs/db/sql/),
    // applied at embedded-PG boot. Nothing to do per-instance.
  }

  /**
   * Close the cached PG pool. The mem0 client is torn down + rebuilt on a
   * TTL (1h) and on `invalidateMemoryClient()`; each rebuild constructs a fresh
   * store via the patched VectorStoreFactory. Without this, the prior store's
   * pool is orphaned — a slow connection leak against embedded PG over a
   * long-running operator. Idempotent and tolerant of a never-connected store.
   */
  async dispose(): Promise<void> {
    const p = this.pool;
    this.pool = null;
    if (!p) return;
    try {
      await p.end();
    } catch {
      /* already closed, or never connected — nothing to release */
    }
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, unknown>[],
  ): Promise<void> {
    if (vectors.length !== ids.length || ids.length !== payloads.length) {
      throw new Error('CanonicalVectorStore.insert: vectors/ids/payloads length mismatch');
    }
    const client = await this.getClient();
    const vecTable = `${this.cfg.schema}.${this.cfg.vecTable}`;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const vec = vectors[i];
      const payload = payloads[i] ?? {};
      if (vec.length !== this.cfg.embeddingModelDims) {
        throw new Error(
          `CanonicalVectorStore.insert: vector dim ${vec.length} !== expected ${this.cfg.embeddingModelDims}`,
        );
      }
      // Upsert canonical first (vec table FKs to it), then vec row.
      await client.query(
        `INSERT INTO ${this.cfg.schema}.memory_canonical (id, payload, created_at, updated_at)
         VALUES ($1, $2::jsonb, now(), now())
         ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, updated_at = now()`,
        [id, JSON.stringify(payload)],
      );
      await client.query(
        `INSERT INTO ${vecTable} (memory_id, vector, embedded_at)
         VALUES ($1, $2::vector, now())
         ON CONFLICT (memory_id) DO UPDATE SET vector = EXCLUDED.vector, embedded_at = now()`,
        [id, toVectorLiteral(vec)],
      );
    }
  }

  async search(
    query: number[],
    topK = 5,
    filters?: SearchFilters,
  ): Promise<VectorStoreResult[]> {
    const client = await this.getClient();
    const vecTable = `${this.cfg.schema}.${this.cfg.vecTable}`;
    const conds: string[] = [storeKindCond('c.', this.storeKind), `c.state != 'archived'`];
    const params: unknown[] = [toVectorLiteral(query), topK];
    let idx = 3;
    if (filters) {
      for (const [key, value] of Object.entries(filters)) {
        if (value === undefined || value === null) continue;
        if (typeof value !== 'string' && typeof value !== 'number') continue;
        conds.push(`c.payload->>'${safeKey(key)}' = $${idx}`);
        params.push(String(value));
        idx++;
      }
    }
    const where = conds.length > 0 ? `WHERE ${conds.join(' AND ')}` : '';
    const sql = `
      SELECT c.id, c.payload, 1 - (v.vector <=> $1::vector) AS score
      FROM ${vecTable} v
      JOIN ${this.cfg.schema}.memory_canonical c ON c.id = v.memory_id
      ${where}
      ORDER BY v.vector <=> $1::vector
      LIMIT $2
    `;
    const res = await client.query(sql, params);
    return res.rows.map((r: { id: string; payload: Record<string, unknown>; score: string | number }) => ({
      id: r.id,
      payload: r.payload,
      score: Number(r.score),
    }));
  }

  // mem0 calls this for BM25 hybrid scoring. Returning null tells mem0
  // to fall back to pure semantic; we can wire postgres FTS later if
  // the quality gap is real.
  async keywordSearch(
    _query: string,
    _topK?: number,
    _filters?: SearchFilters,
  ): Promise<VectorStoreResult[] | null> {
    return null;
  }

  /**
   * EMBED-FREE lexical search — the degraded-path fallback behind
   * `MemoryBackend.searchLexical` (WI-4214): when the embedder is saturated
   * or down, semantic search is unusable but PG itself is healthy, so a
   * plain token match over the canonical text still serves recall. Pulls
   * candidates matching ANY query token (ILIKE), then scores by
   * token-overlap fraction (matched/total, 0..1) in JS — NOT on the cosine
   * scale; ordering only. Reuses the store-kind + archived guards and the
   * post-filter semantics of search()/list(). Bounded: ≤ LEXICAL_MAX_TOKENS
   * tokens, candidates capped, no vec-table join.
   */
  async lexicalSearch(
    query: string,
    topK = 5,
    filters?: SearchFilters,
  ): Promise<VectorStoreResult[]> {
    const tokens = lexicalTokens(query);
    if (tokens.length === 0) return [];
    const client = await this.getClient();
    const conds: string[] = [storeKindCond('', this.storeKind), `state != 'archived'`];
    const params: unknown[] = [];
    let idx = 1;
    if (filters) {
      for (const [key, value] of Object.entries(filters)) {
        if (value === undefined || value === null) continue;
        if (typeof value !== 'string' && typeof value !== 'number') continue;
        conds.push(`payload->>'${safeKey(key)}' = $${idx}`);
        params.push(String(value));
        idx++;
      }
    }
    // `_` is a LIKE single-char wildcard and survives the tokenizer —
    // escape it so a token like `user_id` matches literally.
    const tokenConds = tokens.map((t) => {
      params.push(`%${t.replace(/[\\%_]/g, (m) => `\\${m}`)}%`);
      return `payload->>'data' ILIKE $${idx++}`;
    });
    conds.push(`(${tokenConds.join(' OR ')})`);
    // Over-fetch candidates newest-first: token-overlap scoring happens in
    // JS, and the created_at order makes score ties resolve to fresh rows.
    const candidateCap = Math.max(topK * 3, 30);
    params.push(candidateCap);
    const res = await client.query(
      `SELECT id, payload
       FROM ${this.cfg.schema}.memory_canonical
       WHERE ${conds.join(' AND ')}
       ORDER BY created_at DESC
       LIMIT $${idx}`,
      params,
    );
    const scored = res.rows.map((r: { id: string; payload: Record<string, unknown> }) => {
      const data = String(r.payload?.data ?? '').toLowerCase();
      const matched = tokens.filter((t) => data.includes(t)).length;
      return { id: r.id, payload: r.payload, score: matched / tokens.length };
    });
    // Stable sort: score desc, ties keep the newest-first candidate order.
    return scored.sort((a: VectorStoreResult, b: VectorStoreResult) => (b.score ?? 0) - (a.score ?? 0)).slice(0, topK);
  }

  async get(id: string): Promise<VectorStoreResult | null> {
    const client = await this.getClient();
    const res = await client.query(
      `SELECT id, payload FROM ${this.cfg.schema}.memory_canonical WHERE id = $1`,
      [id],
    );
    if (res.rowCount === 0) return null;
    return { id: res.rows[0].id, payload: res.rows[0].payload };
  }

  async update(id: string, vector: number[], payload: Record<string, unknown>): Promise<void> {
    await this.insert([vector], [id], [payload]);
  }

  /**
   * Patch a memory row's metadata WITHOUT re-embedding — a shallow merge of
   * `patch` into the existing `payload` jsonb (`payload || patch`, so patch keys
   * override and unspecified keys are preserved). This is the store half of the
   * neutral `MemoryBackend.update({ metadata })` path (mem0's OSS text-only
   * update can't touch metadata). VEC-SAFE: the embedding lives in the separate
   * `memory_vec_*` tables keyed by `memory_id`, untouched here — scope (`user_id`),
   * `workspace_id`, `kind`, anchors etc. are filter/display fields, not embedded,
   * so a metadata fix needs no re-embed. Guarded to MEMORY rows (never an mem0
   * entity-linking row, which carries `entityType`). Returns whether a row matched
   * (false = unknown id → the not-found contract the backend surfaces).
   */
  async updatePayload(id: string, patch: Record<string, unknown>): Promise<boolean> {
    if (!patch || typeof patch !== 'object' || Object.keys(patch).length === 0) return false;
    const client = await this.getClient();
    const res = await client.query(
      `UPDATE ${this.cfg.schema}.memory_canonical
          SET payload = payload || $2::jsonb, updated_at = now()
        WHERE id = $1 AND NOT (payload ? 'entityType')`,
      [id, JSON.stringify(patch)],
    );
    return (res.rowCount ?? 0) > 0;
  }

  async delete(id: string): Promise<void> {
    const client = await this.getClient();
    // CASCADE removes the vec rows (in BOTH model tables) — deleting
    // a memory is a real delete, not a per-model delete.
    await client.query(`DELETE FROM ${this.cfg.schema}.memory_canonical WHERE id = $1`, [id]);
  }

  /**
   * mem0 calls this for collection-wide reset (deleteAll, reset). The
   * canonical table is SHARED across every user + harness (scope lives in
   * `payload.user_id`), so an unscoped `DELETE` here would wipe EVERYONE's
   * memories — a multi-tenant data-loss footgun. We scope the reset to the
   * store's current `userId` instead: cascade clears that user's vec rows in
   * both model tables, and other scopes are untouched. With no active scope we
   * refuse rather than nuke the shared table.
   */
  async deleteCol(): Promise<void> {
    if (!this.userId) {
      console.warn(
        '[memory] CanonicalVectorStore.deleteCol called with no userId scope — ' +
          'refusing to wipe the shared memory_canonical table.',
      );
      return;
    }
    const client = await this.getClient();
    await client.query(
      `DELETE FROM ${this.cfg.schema}.memory_canonical WHERE payload->>'user_id' = $1`,
      [this.userId],
    );
  }

  /**
   * List canonical rows matching the filter. Crucially this does NOT
   * gate on vec-table presence — entries written under another
   * embedder mode are real memories the user expects to see and
   * edit/delete. (Semantic SEARCH naturally requires a vector in the
   * active model; LIST does not.)
   */
  async list(
    filters?: SearchFilters,
    topK = 100,
  ): Promise<[VectorStoreResult[], number]> {
    const client = await this.getClient();
    const conds: string[] = [storeKindCond('', this.storeKind), `state != 'archived'`];
    const params: unknown[] = [];
    let idx = 1;
    if (filters) {
      for (const [key, value] of Object.entries(filters)) {
        if (value === undefined || value === null) continue;
        if (typeof value !== 'string' && typeof value !== 'number') continue;
        conds.push(`payload->>'${safeKey(key)}' = $${idx}`);
        params.push(String(value));
        idx++;
      }
    }
    const where = conds.length > 0 ? `WHERE ${conds.join(' AND ')}` : '';

    const listSql = `
      SELECT id, payload
      FROM ${this.cfg.schema}.memory_canonical
      ${where}
      ORDER BY created_at DESC
      LIMIT $${idx}
    `;
    const countSql = `
      SELECT COUNT(*)::bigint AS n
      FROM ${this.cfg.schema}.memory_canonical
      ${where}
    `;

    const [listRes, countRes] = await Promise.all([
      client.query(listSql, [...params, topK]),
      client.query(countSql, params),
    ]);

    const rows: VectorStoreResult[] = listRes.rows.map(
      (r: { id: string; payload: Record<string, unknown> }) => ({
        id: r.id,
        payload: r.payload,
      }),
    );
    return [rows, Number(countRes.rows[0].n)];
  }

  async getUserId(): Promise<string> {
    return this.userId;
  }

  async setUserId(userId: string): Promise<void> {
    this.userId = userId;
  }
}
