/**
 * Re-embed memories when the user switches `memoryEmbedderMode`.
 *
 * Background: each mode uses a different embedding model, and the vector spaces
 * aren't comparable — the same query embedded by different models lands in
 * different parts of the space — so each mode gets its OWN per-fact vector row.
 *
 * The per-mode WIDTHS are not stated here on purpose: they are one fact, owned by
 * `MODE_DIMS` in vec-write.ts (a restatement here is what went stale in WI-7107).
 * They are not uniform — `local` is bge-small @ native 384 and cannot emit more,
 * while `openai` and `gemma` moved to 768 with migration 727.
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
// ⚠ SINGLE SOURCE — do NOT re-declare these here. This file used to keep private
// copies of ResolvedVecMode / VEC_TABLE / MODE_DIMS, and the 384 -> 768 widening
// (migration 727) updated vec-write.ts's copy while this one silently went stale.
// Because the width guard below SKIPS rather than throws, that made every
// re-embed a no-op that still reported success (WI-7107). One declaration means
// the next width change cannot half-land.
import { MODE_DIMS, VEC_TABLE, type ResolvedVecMode } from './vec-write';

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
 * Walk every canonical fact that has a SOURCE-mode vector but does NOT yet
 * have a TARGET-mode vector, embed its body under the target embedder, and
 * insert the row into the target vec table. This is DELTA-ONLY by design
 * (WI-4092): re-running (or re-firing after a partial/interrupted pass) only
 * ever processes the remaining gap, not the whole corpus — a corpus that
 * grows into the thousands would otherwise re-embed every already-migrated
 * row on every call and blow the route's request timeout well before
 * reaching the rows that actually still need it. The `ON CONFLICT DO UPDATE`
 * on the upsert is a race-safety net only (a row inserted concurrently
 * between the SELECT and this row's INSERT), not the primary mechanism.
 *
 * Returns progress counts; throws on fatal errors (PG unavailable, embedder
 * build failure).
 */
export async function reembedMemories(
  fromMode: ResolvedVecMode,
  toMode: ResolvedVecMode,
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

    // Source = canonical facts that already carry a source-mode vector but
    // NOT yet a target-mode one (WI-4092: delta-only — see the doc comment
    // above). The canonical + vec tables exist from migration 081 (applied
    // at embedded-PG boot) — no DDL here.
    const rows = await client.query<{ id: string; payload: Record<string, unknown> }>(
      `SELECT c.id, c.payload
         FROM ${schema}.memory_canonical c
         JOIN ${fromTable} v ON v.memory_id = c.id
        WHERE NOT EXISTS (
          SELECT 1 FROM ${toTable} t WHERE t.memory_id = c.id
        )`,
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
        if (vec.length !== MODE_DIMS[toMode]) {
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
