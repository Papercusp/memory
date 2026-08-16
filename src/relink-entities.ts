/**
 * Entity re-link backfill (EI-10218) — upgrade existing junk COMPOUND entity
 * rows to clean ones.
 *
 * Background: mem0's entity extractor silently ran its greedy REGEX FALLBACK in
 * prod (EI-10183: the ESM build's `__require("compromise")` threw with no global
 * require), so every COMPOUND entity written before the fix is a lowercase
 * sentence-fragment ("liner in the folder"). Those aren't inert — mem0's
 * search() fuses an ENTITY-BOOST signal off the entity store (index.js:6809), and
 * EI-10206 measured that CLEAN entities beat junk on disambiguation recall. The
 * EI-10183 code fix only helps NEW writes; this backfill re-links the EXISTING
 * corpus so the already-stored memories get clean entities too.
 *
 * Shape mirrors reembed.ts: an explicit, user/owner-triggered maintenance pass
 * (NOT automatic), delta-driven + re-run-safe, progress-reporting, with a dry-run.
 *
 * DELTA (bounds each call like reembed's WI-4092 gap-only walk): the unit of work
 * is "memories still linked from a JUNK COMPOUND entity". For each such memory we
 * delete its junk entity rows (+ their vec rows) and re-link it via mem0's own
 * `_linkEntitiesForMemory` with the fix ACTIVE — clean re-extraction. After a
 * memory is processed its junk entities are gone, so it drops out of the affected
 * set; the next call (or a re-run after an interrupted pass) only ever processes
 * what remains. `maxMemories` caps a single call so a large corpus can't blow a
 * request timeout.
 *
 * PROPER / QUOTED entities are never touched. Re-linking re-adds them via mem0's
 * per-entity "search existing (sim>=0.95) → append link, else insert", so an
 * already-present PROPER/QUOTED just gets its linkedMemoryIds Set-unioned (no
 * dupes). Only junk COMPOUND is deleted; clean COMPOUND is (re)created.
 *
 * Re-linking is SERIAL by design: `_linkEntitiesForMemory` dedups an entity by
 * searching the store for an existing match, so two memories that share an entity
 * must be linked one-after-another or they'd each insert a duplicate row.
 *
 * Kill-switch: PAPERCUSP_MEMORY_ENTITY_RELINK=off.
 */

import { memorySchema } from './config';
import { isLowQualityCompoundEntity } from './canonical-store';
import { getMemoryClient } from './mem0-client';

interface RelinkProgress {
  /** memories that were still linked from a junk COMPOUND entity (this batch). */
  affectedMemories: number;
  /** memories successfully re-linked. */
  memoriesRelinked: number;
  /** junk COMPOUND entity rows deleted. */
  junkDeleted: number;
  errors: number;
}

export interface RelinkResult extends RelinkProgress {
  dryRun: boolean;
  /** junk COMPOUND rows still present after this pass (0 ⇒ corpus fully clean). */
  junkRemaining: number;
  durationMs: number;
}

type Linker = (memoryId: string, text: string, filters: Record<string, string>) => Promise<void>;

/**
 * Re-link the memories that still carry junk COMPOUND entities. Returns progress
 * counts; throws on fatal errors (kill-switch off, PG unavailable, the mem0
 * linker private API missing after a mem0 upgrade).
 */
export async function relinkEntities(
  opts: { dryRun?: boolean; maxMemories?: number; userId?: string; progress?: (p: RelinkProgress) => void } = {},
): Promise<RelinkResult> {
  if (process.env.PAPERCUSP_MEMORY_ENTITY_RELINK === 'off') {
    throw new Error('relink_disabled');
  }
  const started = Date.now();
  const dryRun = opts.dryRun ?? false;
  const maxMemories = opts.maxMemories ?? 1000;
  const schema = memorySchema();

  const { pgFields } = await import('./mem0-connection');
  const pg = await pgFields();
  // `require('pg')` throws in this ESM package (see reembed.ts / mem0-client.ts) —
  // dynamic import + CJS interop.
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
    // 1. All COMPOUND entity rows; the junk ones are decided in JS by the SAME
    //    gate the insert path uses (isLowQualityCompoundEntity), so the backfill
    //    delete matches the write-time filter exactly.
    const compoundRows = await client.query<{ id: string; data: string | null; linked: string[] | null }>(
      `SELECT id,
              payload->>'data' AS data,
              ARRAY(SELECT jsonb_array_elements_text(COALESCE(payload->'linkedMemoryIds','[]'::jsonb))) AS linked
         FROM ${schema}.memory_canonical
        WHERE payload->>'entityType' = 'COMPOUND'
          ${opts.userId ? `AND payload->>'user_id' = $1` : ''}`,
      opts.userId ? [opts.userId] : [],
    );
    const junk = compoundRows.rows.filter((r) => isLowQualityCompoundEntity(r.data ?? ''));
    const junkRemainingTotal = junk.length;

    // 2. Memories still reachable from a junk entity = the affected set (delta).
    const affected = new Set<string>();
    for (const j of junk) for (const mid of j.linked ?? []) if (mid) affected.add(mid);

    const progress: RelinkProgress = {
      affectedMemories: 0,
      memoriesRelinked: 0,
      junkDeleted: 0,
      errors: 0,
    };

    // Bound this pass. Only these memories (and their junk entities) are touched.
    const batchMemIds = Array.from(affected).slice(0, maxMemories);
    const batchSet = new Set(batchMemIds);
    progress.affectedMemories = batchMemIds.length;

    if (dryRun) {
      // Junk rows that would be deleted = those linked ONLY to in-batch memories
      // (a junk entity linked to an out-of-batch memory survives this batch).
      const wouldDelete = junk.filter((j) => (j.linked ?? []).length > 0 && (j.linked ?? []).every((m) => batchSet.has(m))).length;
      return { ...progress, junkDeleted: wouldDelete, dryRun, junkRemaining: junkRemainingTotal, durationMs: Date.now() - started };
    }

    // 3. Delete the junk entity rows whose links are entirely within this batch
    //    (so we never orphan a junk entity that also serves an unprocessed memory).
    const deletableJunkIds = junk
      .filter((j) => (j.linked ?? []).length > 0 && (j.linked ?? []).every((m) => batchSet.has(m)))
      .map((j) => j.id);
    if (deletableJunkIds.length > 0) {
      // Entity vec rows live in the active-mode vec table(s), keyed by the entity
      // row id; delete from both known local tables before the canonical rows.
      await client.query(`DELETE FROM ${schema}.memory_vec_harrier WHERE memory_id = ANY($1)`, [deletableJunkIds]);
      await client.query(`DELETE FROM ${schema}.memory_vec_gemma WHERE memory_id = ANY($1)`, [deletableJunkIds]);
      const del = await client.query(`DELETE FROM ${schema}.memory_canonical WHERE id = ANY($1)`, [deletableJunkIds]);
      progress.junkDeleted = del.rowCount ?? deletableJunkIds.length;
    }

    // 4. Re-link each affected memory via mem0's own linker (fix active ⇒ clean).
    const mem0 = await getMemoryClient();
    if (!mem0) throw new Error('mem0_unavailable');
    const linker = (mem0 as unknown as { _linkEntitiesForMemory?: Linker })._linkEntitiesForMemory;
    if (typeof linker !== 'function') throw new Error('relink_no_linker'); // mem0 upgrade renamed it

    // Fetch (data, user_id) for the batch memories.
    const memRows = await client.query<{ id: string; data: string | null; user_id: string | null }>(
      `SELECT id, payload->>'data' AS data, payload->>'user_id' AS user_id
         FROM ${schema}.memory_canonical
        WHERE id = ANY($1) AND NOT (payload ? 'entityType')`,
      [batchMemIds],
    );
    for (const row of memRows.rows) {
      if (!row.data || !row.user_id) { progress.errors += 1; opts.progress?.(progress); continue; }
      try {
        await linker.call(mem0, row.id, row.data, { user_id: row.user_id });
        progress.memoriesRelinked += 1;
      } catch {
        progress.errors += 1;
      }
      opts.progress?.(progress);
    }

    return {
      ...progress,
      dryRun,
      junkRemaining: junkRemainingTotal - progress.junkDeleted,
      durationMs: Date.now() - started,
    };
  } finally {
    try { await client.end(); } catch { /* ignore */ }
  }
}
