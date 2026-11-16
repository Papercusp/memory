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
 * re-link it via mem0's own `_linkEntitiesForMemory`, then retire only verified
 * successful links from the junk row. Empty junk rows and their vectors are
 * removed together in a transaction. Failed links remain retryable and move
 * behind pending work. `maxMemories` bounds the number of attempted memories.
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

type EntityPayload = Record<string, unknown>;
interface RelinkRow {
  id: string;
  payload: EntityPayload;
  data: string | null;
  user_id: string | null;
  linked: string[] | null;
}

function cleanLink(payload: EntityPayload, memoryId: string, userId: string): boolean {
  return payload.user_id === userId && typeof payload.data === 'string' && payload.data.length > 0 &&
    typeof payload.entityType === 'string' &&
    !(payload.entityType === 'COMPOUND' && isLowQualityCompoundEntity(payload.data)) &&
    Array.isArray(payload.linkedMemoryIds) && payload.linkedMemoryIds.includes(memoryId);
}

/** Observe this invocation's writes without changing the shared mem0 singleton.
 * The SDK intentionally catches embed/store errors and returns void, so neither
 * resolution nor an attempted insert proves that a replacement exists.
 */
async function linkWithReceipt(mem0: object, linker: Linker, row: RelinkRow): Promise<string[]> {
  const original = mem0 as {
    getEntityStore?: () => Promise<object>;
    embedder?: { embed?: (...args: unknown[]) => Promise<unknown> };
  };
  if (typeof original.getEntityStore !== 'function' || typeof original.embedder?.embed !== 'function')
    throw new Error('relink_no_verifiable_linker');
  const context = Object.create(mem0);
  const ids = new Set<string>();
  let failed = false;
  const embedder = original.embedder;
  const embed = embedder.embed!;
  Object.defineProperty(context, 'embedder', { value: Object.assign(Object.create(embedder), {
    async embed(...args: unknown[]) {
      try { return await embed.apply(embedder, args); }
      catch (error) { failed = true; throw error; }
    },
  }) });
  Object.defineProperty(context, 'getEntityStore', { value: async () => {
    const store = await original.getEntityStore!.call(mem0);
    return new Proxy(store, {
      get(target, property) {
        const value = Reflect.get(target, property, target);
        if (typeof value !== 'function') return value;
        if (property !== 'insert' && property !== 'update') return value.bind(target);
        return async (...args: unknown[]) => {
          try {
            const result = await value.apply(target, args);
            const writes: Array<[unknown, unknown]> = property === 'insert'
              ? (Array.isArray(args[1]) ? args[1] : []).map((id, i) =>
                [id, Array.isArray(args[2]) ? args[2][i] : null])
              : [[args[0], args[2]]];
            for (const [id, payload] of writes) {
              if (typeof id === 'string' && payload && typeof payload === 'object' &&
                  cleanLink(payload as EntityPayload, row.id, row.user_id!)) ids.add(id);
            }
            return result;
          } catch (error) { failed = true; throw error; }
        };
      },
    });
  } });
  await linker.call(context, row.id, row.data!, { user_id: row.user_id! });
  if (failed || ids.size === 0) throw new Error('relink_no_verified_replacement');
  return [...ids];
}

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
  if (!Number.isSafeInteger(maxMemories) || maxMemories <= 0)
    throw new Error('relink_invalid_max_memories');
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
    const readJunk = async () => {
      const compoundRows = await client.query<RelinkRow>(
      `SELECT id, payload, payload->>'user_id' AS user_id,
              payload->>'data' AS data,
              ARRAY(SELECT jsonb_array_elements_text(COALESCE(payload->'linkedMemoryIds','[]'::jsonb))) AS linked
         FROM ${schema}.memory_canonical
        WHERE payload->>'entityType' = 'COMPOUND'
          ${opts.userId ? `AND payload->>'user_id' = $1` : ''}
        ORDER BY updated_at, id`,
      opts.userId ? [opts.userId] : [],
      );
      return compoundRows.rows.filter((r) => isLowQualityCompoundEntity(r.data ?? ''));
    };
    const junk = await readJunk();
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

    if (batchMemIds.length === 0)
      return { ...progress, dryRun, junkRemaining: junkRemainingTotal, durationMs: Date.now() - started };

    // Validate the private API before changing any existing entity or vector.
    const mem0 = await getMemoryClient();
    if (!mem0) throw new Error('mem0_unavailable');
    const linker = (mem0 as unknown as { _linkEntitiesForMemory?: Linker })._linkEntitiesForMemory;
    if (typeof linker !== 'function') throw new Error('relink_no_linker'); // mem0 upgrade renamed it

    // Fetch (data, user_id) for the batch memories.
    const memRows = await client.query<RelinkRow>(
      `SELECT id, payload, payload->>'data' AS data, payload->>'user_id' AS user_id
         FROM ${schema}.memory_canonical
        WHERE id = ANY($1) AND NOT (payload ? 'entityType')`,
      [batchMemIds],
    );
    const memories = new Map(memRows.rows.map(row => [row.id, row]));
    for (const memoryId of batchMemIds) {
      const row = memories.get(memoryId);
      const oldJunk = junk.filter(entity => entity.linked?.includes(memoryId));
      let replacements: string[] = [];
      let success = Boolean(row?.data && row.user_id &&
        (!opts.userId || row.user_id === opts.userId) &&
        oldJunk.every(entity => entity.user_id === row.user_id));
      try {
        if (success) replacements = await linkWithReceipt(mem0, linker, row!);
      } catch { success = false; }

      // Do not hold these row locks across the SDK's separate PG connection.
      // Recheck the memory and persisted clean writes, then retire only this
      // successful link from the CURRENT junk payload (including new links).
      let deleted = 0;
      try {
        await client.query('BEGIN');
        if (success) {
          const current = await client.query<RelinkRow>(
            `SELECT id, payload, payload->>'data' AS data, payload->>'user_id' AS user_id
               FROM ${schema}.memory_canonical WHERE id = ANY($1)
                AND NOT (payload ? 'entityType') FOR UPDATE`, [[memoryId]],
          );
          const clean = await client.query<RelinkRow>(
            `SELECT id, payload FROM ${schema}.memory_canonical
              WHERE id = ANY($1) ORDER BY id FOR UPDATE`, [replacements],
          );
          success = current.rows.length === 1 && current.rows[0].data === row!.data &&
            current.rows[0].user_id === row!.user_id && clean.rows.length === replacements.length &&
            clean.rows.every(entity => cleanLink(entity.payload, memoryId, row!.user_id!));
        }
        const currentJunk = await client.query<RelinkRow>(
          `SELECT id, payload FROM ${schema}.memory_canonical
            WHERE id = ANY($1) ORDER BY id FOR UPDATE`, [oldJunk.map(entity => entity.id).sort()],
        );
        for (const entity of currentJunk.rows) {
          const original = oldJunk.find(old => old.id === entity.id)!;
          if (entity.payload.entityType !== 'COMPOUND' || entity.payload.data !== original.data ||
              entity.payload.user_id !== original.user_id) { success = false; continue; }
          const links = Array.isArray(entity.payload.linkedMemoryIds) ? entity.payload.linkedMemoryIds : [];
          if (!links.includes(memoryId)) continue;
          const remaining = links.filter(id => id !== memoryId);
          if (!success) remaining.push(memoryId);
          if (remaining.length) {
            await client.query(`UPDATE ${schema}.memory_canonical
              SET payload = jsonb_set(payload, '{linkedMemoryIds}', $2::jsonb), updated_at = now()
              WHERE id = $1`, [entity.id, JSON.stringify(remaining)]);
          } else {
            await client.query(`DELETE FROM ${schema}.memory_vec_harrier WHERE memory_id = ANY($1)`, [[entity.id]]);
            await client.query(`DELETE FROM ${schema}.memory_vec_gemma WHERE memory_id = ANY($1)`, [[entity.id]]);
            const del = await client.query(`DELETE FROM ${schema}.memory_canonical WHERE id = ANY($1)`, [[entity.id]]);
            deleted += del.rowCount ?? 0;
          }
        }
        await client.query('COMMIT');
        progress.junkDeleted += deleted;
        if (success) progress.memoriesRelinked += 1;
        else progress.errors += 1;
      } catch {
        await client.query('ROLLBACK');
        progress.errors += 1;
      }
      opts.progress?.(progress);
    }

    return {
      ...progress,
      dryRun,
      junkRemaining: (await readJunk()).length,
      durationMs: Date.now() - started,
    };
  } finally {
    try { await client.end(); } catch { /* ignore */ }
  }
}
