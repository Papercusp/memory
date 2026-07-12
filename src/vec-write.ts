/**
 * vec-write.ts — write a memory row's VECTOR into its per-mode vec table
 * WITHOUT touching the stored `payload.data`.
 *
 * The canonical schema stores each fact's text ONCE in
 * `<schema>.memory_canonical` and each embedder mode's vector SEPARATELY in
 * `<schema>.memory_vec_<mode>`, joined by `memory_id` (see canonical-store.ts).
 * Because vector and text live in different tables, a row's embedding can be
 * (re)computed from a DIFFERENT string than the one stored — the seam behind:
 *
 *   - the re-embed pass (mode switch — reembed.ts embeds each body under the
 *     target model and upserts the target vec row), and
 *   - write-time embed AUGMENTATION (EI-10048): store the CLEAN body but embed
 *     an ENRICHED string (clean text + resolved reference titles) so a
 *     ref-only memory ALSO matches queries about the referenced item's TOPIC.
 *     That is the multi-hop recall a flat store can't bridge (bench hop-recall
 *     .038 absolute); query-time graph fusion closes it only at general-lane-
 *     wrecking weights and was REJECTED (D-001). Doing the hop ONCE at write
 *     time costs nothing at query time and never perturbs ranking.
 *
 * The vec table's `memory_id` is unique, so the upsert OVERWRITES the row's
 * baseline vector in place (the one mem0's `add()` just wrote) and leaves the
 * canonical text untouched.
 */

import { memoryHost, memorySchema } from './config';

export type ResolvedVecMode = 'openai' | 'local' | 'gemma' | 'harrier';

/** Mode → its (unqualified) vec table. Fixed lookup — never interpolate
 *  caller input into a SQL identifier; the schema is prefixed at use. */
export const VEC_TABLE: Record<ResolvedVecMode, string> = {
  openai: 'memory_vec_openai',
  local: 'memory_vec_local',
  gemma: 'memory_vec_gemma',
  harrier: 'memory_vec_harrier',
};

/** Per-mode vector width — the wrong-width guard must match the mode's space
 *  (harrier is native-1024; every other shipped mode is 384). */
export const MODE_DIMS: Record<ResolvedVecMode, number> = {
  openai: 384,
  local: 384,
  gemma: 384,
  harrier: 1024,
};

/** The parameterized vec-upsert statement for one mode ($1 = memory_id,
 *  $2 = vector literal). `ON CONFLICT (memory_id)` overwrites in place. */
export function vecUpsertSql(schema: string, mode: ResolvedVecMode): string {
  return `INSERT INTO ${schema}.${VEC_TABLE[mode]} (memory_id, vector, embedded_at)
          VALUES ($1, $2::vector, now())
          ON CONFLICT (memory_id) DO UPDATE SET vector = EXCLUDED.vector, embedded_at = now()`;
}

/** Format a JS number[] as a pgvector literal. */
export function toVectorLiteral(vec: number[]): string {
  return `[${vec.join(',')}]`;
}

/**
 * Embed `text` under the CURRENT-preference embedder and upsert the result as
 * `memoryId`'s vector in that mode's vec table — leaving `payload.data`
 * untouched. This is the default write-time augmentation path (EI-10048):
 * `getMemoryClient()` built its store against the SAME `resolveEmbedder()`
 * mode, so the enriched vector lands in the same table the row's baseline
 * vector was just written to and overwrites it.
 *
 * BEST-EFFORT / NON-FATAL by contract: returns `false` (never throws) when the
 * embedder is disabled, the mode has no vec table, the vector is the wrong
 * width, or PG is unreachable — the caller keeps its clean-text baseline
 * vector. Opens a one-shot pg Client (mirrors reembed.ts); remember() is not a
 * hot loop and this only fires when a caller supplies enriched embed-text, so
 * a per-write connection is acceptable (fold into the canonical pool if it
 * ever gets hot).
 */
export async function embedAndUpsertVector(memoryId: string, text: string): Promise<boolean> {
  try {
    const resolved = await memoryHost().resolveEmbedder();
    if (resolved.mode === 'disabled') return false;
    const mode = resolved.mode as ResolvedVecMode;
    if (!VEC_TABLE[mode]) return false;
    const vec = await resolved.embed(text);
    if (!Array.isArray(vec) || vec.length !== MODE_DIMS[mode]) return false;

    const schema = memorySchema();
    // `require('pg')` throws in this ESM package (see mem0-client/reembed) —
    // dynamic import + CJS interop.
    const pgMod = (await import('pg')) as typeof import('pg') & { default?: typeof import('pg') };
    const Client = pgMod.Client ?? pgMod.default?.Client;
    if (!Client) return false;
    const { pgClientFields } = await import('./mem0-connection');
    const client = new Client(await pgClientFields());
    await client.connect();
    try {
      const r = await client.query(vecUpsertSql(schema, mode), [memoryId, toVectorLiteral(vec)]);
      return (r.rowCount ?? 0) > 0;
    } finally {
      try {
        await client.end();
      } catch {
        /* ignore */
      }
    }
  } catch {
    return false;
  }
}
