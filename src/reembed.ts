/**
 * Re-embed memories when the user switches `memoryEmbedderMode`.
 *
 * Background: switching between `openai` and `local` modes uses different
 * embedding models (text-embedding-3-small @ 384 dims via the `dimensions`
 * param, vs BGE-small @ 384). The vector spaces aren't comparable — the same
 * query embedded by different models lands in different parts of the space —
 * so each mode gets its OWN per-fact vector row.
 *
 * Under the canonical schema the fact text lives once in
 * `<schema>.memory_canonical` and each mode's vector lives in
 * `<schema>.memory_vec_<mode>` (joined by `memory_id`), where `<schema>` is
 * the host-configured schema (default `public`; the operator uses
 * `harness_shared`). The text doesn't
 * move when you switch modes — only which vec table recall reads. So
 * "re-embedding" = embedding each fact under the target model and upserting a
 * row into the target vec table. (The pre-081 worker walked the obsolete
 * per-collection `operator_memory_<mode>` tables, which migration 081 dropped
 * — it read a never-populated source and wrote a target recall never reads.)
 *
 * Without re-embedding, switching mode silently hides memories until the user
 * switches back. This worker walks every fact that has a SOURCE-mode vector,
 * re-embeds its body with the target embedder, and upserts the TARGET-mode
 * vector row.
 *
 * Triggered explicitly via `POST /api/user/memory/reembed`. NOT automatic on
 * mode-change — re-embedding 500 memories at ~100ms/each is ~1min; we keep it
 * user-initiated so they see progress.
 *
 * The target-mode embedder is built by the host (`buildEmbedderForMode` seam)
 * — re-embedding must use the *target* model's space regardless of the current
 * preference, so we can't reuse the preference-resolved embedder. Part of P-021.
 */

import { memoryHost, memorySchema } from './config';

const EMBEDDER_DIM = 384;

type ResolvedMode = 'openai' | 'local' | 'gemma';

/** Map a mode to its (unqualified) vec table — fixed lookup (no interpolation
 *  of caller input into SQL identifiers); schema is prefixed at use.
 *  'gemma' = EmbeddingGemma-300m @ MRL-384 (migration 534). */
const VEC_TABLE: Record<ResolvedMode, string> = {
  openai: 'memory_vec_openai',
  local: 'memory_vec_local',
  gemma: 'memory_vec_gemma',
};

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
 * Walk every canonical fact that has a SOURCE-mode vector, embed its body
 * under the target embedder, and upsert the row into the target vec table.
 * Existing target rows for the same `memory_id` are overwritten.
 *
 * Returns progress counts; throws on fatal errors (PG unavailable, embedder
 * build failure).
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
  const schema = memorySchema();
  const fromTable = `${schema}.${VEC_TABLE[fromMode]}`;
  const toTable = `${schema}.${VEC_TABLE[toMode]}`;

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
    // Build embedder for the target mode (host seam — the operator owns the
    // openai/local cascade).
    const embed = await memoryHost().buildEmbedderForMode(toMode);

    // Source = canonical facts that already carry a source-mode vector. The
    // canonical + vec tables exist from migration 081 (applied at embedded-PG
    // boot) — no DDL here.
    const rows = await client.query<{ id: string; payload: Record<string, unknown> }>(
      `SELECT c.id, c.payload
         FROM ${schema}.memory_canonical c
         JOIN ${fromTable} v ON v.memory_id = c.id`,
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
          `INSERT INTO ${toTable} (memory_id, vector, embedded_at)
           VALUES ($1, $2::vector, now())
           ON CONFLICT (memory_id) DO UPDATE SET vector = EXCLUDED.vector, embedded_at = now()`,
          [row.id, `[${vec.join(',')}]`],
        );
        progress.reembedded += 1;
        opts.progress?.(progress);
      } catch {
        progress.errors += 1;
        opts.progress?.(progress);
      }
    }

    return {
      fromCollection: fromTable,
      toCollection: toTable,
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
