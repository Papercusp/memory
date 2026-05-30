/**
 * Re-embed memories when the user switches `memoryEmbedderMode`.
 *
 * Background: switching between `openai` and `local` modes uses
 * different embedding models (text-embedding-3-small vs BGE-small).
 * Each lives in a per-mode pgvector collection
 * (`operator_memory_openai` vs `operator_memory_local`) because the
 * vector spaces aren't comparable — same query embedded by different
 * models lands in different parts of the space, so cross-model search
 * returns nothing.
 *
 * Without re-embedding, switching mode silently hides memories until
 * the user switches back. This worker walks the source collection,
 * re-embeds each memory body with the target embedder, and upserts
 * into the target collection.
 *
 * Triggered explicitly via `POST /api/user/memory/reembed`. NOT
 * automatic on mode-change — re-embedding 500 memories at 100ms/each
 * is ~1min; we keep it user-initiated so they see progress.
 *
 * The target-mode embedder is built by the host (`buildEmbedderForMode`
 * seam) — re-embedding must use the *target* model's space regardless of
 * the current preference, so we can't reuse the preference-resolved
 * embedder. Part of P-021.
 */

import { memoryHost } from './config';

const EMBEDDER_DIM = 384;
const MEM0_COLLECTION_PREFIX = 'operator_memory';

type ResolvedMode = 'openai' | 'local';

interface PgFields {
  host: string;
  port: number;
  user: string;
  password: string;
  dbname: string;
}

interface ReembedProgress {
  totalSource: number;
  reembedded: number;
  skipped: number;
  errors: number;
}

export interface ReembedResult extends ReembedProgress {
  fromCollection: string;
  toCollection: string;
  durationMs: number;
}

async function loadPgFields(): Promise<PgFields> {
  const { pgFields } = await import('./mem0-connection');
  return pgFields();
}

/**
 * Walk every row in the source collection, embed under the target's
 * embedder, write to the target collection. Existing rows in the
 * target with the same id are upserted (overwritten).
 *
 * Returns progress counts; throws on fatal errors (PG unavailable,
 * embedder build failure).
 */
export async function reembedMemories(
  fromMode: ResolvedMode,
  toMode: ResolvedMode,
  opts: { progress?: (p: ReembedProgress) => void } = {},
): Promise<ReembedResult> {
  if (fromMode === toMode) {
    throw new Error('reembed_noop_same_mode');
  }
  const started = Date.now();
  const fromCollection = `${MEM0_COLLECTION_PREFIX}_${fromMode}`;
  const toCollection = `${MEM0_COLLECTION_PREFIX}_${toMode}`;

  const pg = await loadPgFields();
  // `require('pg')` throws "require is not defined" in this ESM package
  // (mem0-client.ts hit the same) — use dynamic import + CJS interop.
  const pgMod = (await import('pg')) as typeof import('pg') & { default?: typeof import('pg') };
  const Client = pgMod.Client ?? pgMod.default?.Client;
  if (!Client) throw new Error('pg.Client not resolvable');
  const client = new Client({
    host: pg.host,
    port: pg.port,
    user: pg.user,
    password: pg.password,
    database: pg.dbname,
  });
  await client.connect();
  try {
    // Confirm source collection exists; if not, nothing to do.
    const exists = await client.query<{ to_regclass: string | null }>(
      `SELECT to_regclass($1) AS to_regclass`,
      [fromCollection],
    );
    if (!exists.rows[0]?.to_regclass) {
      return {
        fromCollection,
        toCollection,
        totalSource: 0,
        reembedded: 0,
        skipped: 0,
        errors: 0,
        durationMs: Date.now() - started,
      };
    }

    // Build embedder for the target mode (host seam — the operator owns
    // the openai/local cascade).
    const embed = await memoryHost().buildEmbedderForMode(toMode);

    // Ensure target collection exists with the right shape. mem0's
    // PGVector.createCol DDL is what we mirror here. Doing it manually
    // (rather than spinning up a mem0 Memory instance) keeps the worker
    // independent of the runtime mem0 init.
    await client.query('CREATE EXTENSION IF NOT EXISTS vector');
    await client.query(`
      CREATE TABLE IF NOT EXISTS "${toCollection}" (
        id UUID PRIMARY KEY,
        vector vector(${EMBEDDER_DIM}),
        payload JSONB
      )
    `);

    // Stream source rows.
    const rows = await client.query<{ id: string; payload: Record<string, unknown> }>(
      `SELECT id, payload FROM "${fromCollection}"`,
    );
    const progress: ReembedProgress = {
      totalSource: rows.rowCount ?? 0,
      reembedded: 0,
      skipped: 0,
      errors: 0,
    };

    for (const row of rows.rows) {
      const data = (row.payload?.data ?? row.payload?.memory ?? '') as string;
      if (!data || typeof data !== 'string') {
        progress.skipped += 1;
        opts.progress?.(progress);
        continue;
      }
      try {
        const vec = await embed(data);
        if (vec.length !== EMBEDDER_DIM) {
          progress.errors += 1;
          opts.progress?.(progress);
          continue;
        }
        await client.query(
          `INSERT INTO "${toCollection}" (id, vector, payload)
           VALUES ($1, $2::vector, $3::jsonb)
           ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector, payload = EXCLUDED.payload`,
          [row.id, `[${vec.join(',')}]`, row.payload],
        );
        progress.reembedded += 1;
        opts.progress?.(progress);
      } catch {
        progress.errors += 1;
        opts.progress?.(progress);
      }
    }

    return {
      fromCollection,
      toCollection,
      ...progress,
      durationMs: Date.now() - started,
    };
  } finally {
    try {
      await client.end();
    } catch {
      /* ignore */
    }
  }
}
