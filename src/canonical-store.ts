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
 */

import { Client as PgClient } from 'pg';

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
  /** Ignored — mem0 always passes this; we don't use collection
   *  segmentation because the canonical row carries scope in payload. */
  collectionName?: string;
  /** Which model's vec table this instance reads/writes. */
  vecTable: 'memory_vec_openai' | 'memory_vec_local';
  /** Sanity check — refuses to insert vectors with the wrong length. */
  embeddingModelDims: number;
}

function safeKey(k: string): string {
  return k.replace(/[^a-zA-Z0-9_]/g, '');
}

function toVectorLiteral(v: number[]): string {
  return `[${v.join(',')}]`;
}

export class CanonicalVectorStore {
  private cfg: CanonicalStoreConfig;
  private userId = '';
  private clientPromise: Promise<PgClient> | null = null;

  constructor(config: CanonicalStoreConfig) {
    this.cfg = config;
  }

  private async getClient(): Promise<PgClient> {
    if (!this.clientPromise) {
      this.clientPromise = (async () => {
        const c = new PgClient({
          host: this.cfg.host,
          port: this.cfg.port,
          user: this.cfg.user,
          password: this.cfg.password,
          database: this.cfg.dbname,
        });
        await c.connect();
        return c;
      })().catch((err) => {
        // Let the next call retry — transient PG hiccups shouldn't
        // poison-cache this store.
        this.clientPromise = null;
        throw err;
      });
    }
    return this.clientPromise;
  }

  async initialize(): Promise<void> {
    // Schema lives in migration 081 (libs/papercusp/libs/db/sql/),
    // applied at embedded-PG boot. Nothing to do per-instance.
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
    const vecTable = `harness_shared.${this.cfg.vecTable}`;
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
        `INSERT INTO harness_shared.memory_canonical (id, payload, created_at, updated_at)
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
    const vecTable = `harness_shared.${this.cfg.vecTable}`;
    const conds: string[] = [];
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
      JOIN harness_shared.memory_canonical c ON c.id = v.memory_id
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

  async get(id: string): Promise<VectorStoreResult | null> {
    const client = await this.getClient();
    const res = await client.query(
      `SELECT id, payload FROM harness_shared.memory_canonical WHERE id = $1`,
      [id],
    );
    if (res.rowCount === 0) return null;
    return { id: res.rows[0].id, payload: res.rows[0].payload };
  }

  async update(id: string, vector: number[], payload: Record<string, unknown>): Promise<void> {
    await this.insert([vector], [id], [payload]);
  }

  async delete(id: string): Promise<void> {
    const client = await this.getClient();
    // CASCADE removes the vec rows (in BOTH model tables) — deleting
    // a memory is a real delete, not a per-model delete.
    await client.query(`DELETE FROM harness_shared.memory_canonical WHERE id = $1`, [id]);
  }

  /**
   * mem0 calls this for collection-wide reset (deleteAll, reset).
   * We interpret it as "drop everything in this store's pool" — i.e.
   * every canonical row + cascade. We do NOT scope this to vec-table
   * presence; reset means reset.
   */
  async deleteCol(): Promise<void> {
    const client = await this.getClient();
    await client.query(`DELETE FROM harness_shared.memory_canonical`);
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
    const conds: string[] = [];
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
      FROM harness_shared.memory_canonical
      ${where}
      ORDER BY created_at DESC
      LIMIT $${idx}
    `;
    const countSql = `
      SELECT COUNT(*)::bigint AS n
      FROM harness_shared.memory_canonical
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
