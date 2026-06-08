/**
 * Reciprocal-rank fusion for the HybridBackend
 * (memory-backend-improve-and-hybrid P-020 / P-031).
 *
 * The hybrid fuses two retrieval legs over the same canonical store:
 *   - the COSINE leg (semantic / paraphrase — wins the lexical-gap class), and
 *   - the LEXICAL leg (exact tokens / identifiers — wins the exact-identifier class).
 *
 * The caller FP-floors the cosine leg (SearchOptions.minScore) so an out-of-corpus
 * query yields no cosine hits. Fusion then combines the legs in one of two modes:
 *
 *   - `floored-union` (DEFAULT) — the result is the UNION of the floored cosine
 *     hits and the lexical hits whose normalized score clears `minLexScore`. This
 *     is what captures the exact-identifier column: a target the cosine leg ranks
 *     poorly (or misses) but the lexical leg matches on an identifier token is
 *     ADMITTED, lifting exact-id recall toward the lexical leg's ceiling. The dual
 *     gate keeps hard-negatives out: a query with no real answer floors away in
 *     cosine AND fails the lexical identifier bar, so neither leg admits noise.
 *
 *   - `cosine-gated` — the result is exactly the cosine hits; the lexical leg only
 *     RE-RANKS them (a lexical-only hit is never admitted). Strictest hard-negative
 *     discipline, but recall is capped at the cosine leg's (hybrid_recall ⊆ cosine):
 *     it cannot capture an exact-id target the cosine leg dropped. Kept selectable
 *     for the P-031 sweep / when no lexical precision signal is trustworthy.
 *
 * RRF score = Σ 1/(k+rank) over the legs an entry appears in (standard k=60). The
 * fused score is written onto entry.score (ordering-only, per the MemoryEntry
 * contract). The exact mode + the two gate values are swept in P-031 (D-006).
 */
import type { MemoryEntry } from './backend';

export const DEFAULT_RRF_K = 60;
/**
 * Default lexical admission bar for floored-union (normalized scoreEntry 0..1).
 * Locked at 0.30 by the P-031 2D sweep (D-006): genuine exact-identifier matches
 * score 0.33–0.96 on the lexical leg (long queries dilute the normalized score),
 * so 0.30 ADMITS them — lifting exact-id MRR to 0.94–0.96 (≈ the lexical leg's
 * ceiling) vs 0.87 at a 0.40 bar. This is the recall-favoring default for the
 * pull path (where an LLM judges relevance); the no-LLM push path tightens it to
 * 0.40 + a score floor for precision (injection.ts, the read-gate split D-003).
 */
export const DEFAULT_MIN_LEX_SCORE = 0.3;

export type FusionMode = 'floored-union' | 'cosine-gated';

export interface FusionOptions {
  /** RRF damping constant (default 60). */
  k?: number;
  /** Fusion mode (default 'floored-union'). */
  mode?: FusionMode;
  /**
   * Minimum normalized lexical score for a lexical-ONLY hit to be admitted in
   * floored-union mode (default 0.5). Lexical hits that also appear in the cosine
   * set are always kept (they only get a rank boost); this bar gates the
   * lexical-only admissions so generic token overlap can't re-introduce
   * hard-negative false positives. No effect in cosine-gated mode.
   */
  minLexScore?: number;
  /**
   * Weight on the LEXICAL leg's RRF contribution (default 1 = democratic RRF;
   * the cosine leg's weight is fixed at 1). >1 trusts the lexical leg more.
   * MEASURED (P-031b sweep, local-BGE bench so OpenAI 503s couldn't contaminate
   * it; see plan D-007): lexWeight ≥ 2 lifts exact-identifier MRR to the lexical
   * leg's 1.00 CEILING (from 0.94 at w1) but COSTS ~0.08 paraphrase MRR
   * (0.73→0.65), and it SATURATES at 2 (w4/w8 identical). Net overall MRR is
   * slightly LOWER (0.82 vs 0.84) — the paraphrase loss outweighs the exact-id
   * gain — so the default stays 1 (best aggregate quality). Raise it only for an
   * exact-identifier-heavy workload that genuinely values 1.00 exact-id over
   * paraphrase recall.
   */
  lexWeight?: number;
}

/**
 * Fuse the cosine and lexical legs. The cosine leg is assumed already FP-floored
 * by the caller (SearchOptions.minScore). Returns entries ordered by fused RRF
 * score (descending), each carrying that score.
 */
export function fuse(
  cosineHits: readonly MemoryEntry[],
  lexicalHits: readonly MemoryEntry[],
  opts: FusionOptions = {},
): MemoryEntry[] {
  const k = opts.k ?? DEFAULT_RRF_K;
  const mode = opts.mode ?? 'floored-union';
  const minLex = opts.minLexScore ?? DEFAULT_MIN_LEX_SCORE;
  const lexWeight = opts.lexWeight ?? 1;

  // CROSS-LEG IDENTITY: the cosine leg (canonical store) and the lexical leg
  // (its projection) assign DIFFERENT native ids to the SAME memory, so dedup by
  // a shared key, not the native id — else one fact surfaces twice (once per leg).
  // The write-through stamps `metadata.link_id` = the canonical id on the lexical
  // projection; fall back to the native id when there's no link (single-leg hits).
  const keyOf = (e: MemoryEntry): string | undefined =>
    (typeof e.metadata?.link_id === 'string' ? (e.metadata.link_id as string) : undefined) ?? e.id;

  // Lexical rank (1-based) + the lexical entry, keyed by cross-leg key (first wins).
  const lexRank = new Map<string, number>();
  const lexEntry = new Map<string, MemoryEntry>();
  lexicalHits.forEach((e, i) => {
    const key = keyOf(e);
    if (key && !lexRank.has(key)) {
      lexRank.set(key, i + 1);
      lexEntry.set(key, e);
    }
  });

  // Candidate set + each candidate's cosine rank (undefined = lexical-only). The
  // COSINE entry wins the slot when a memory is in both legs (it's canonical).
  const cosRank = new Map<string, number>();
  const candidate = new Map<string, MemoryEntry>();
  cosineHits.forEach((e, i) => {
    const key = keyOf(e);
    if (!key) return;
    if (!cosRank.has(key)) cosRank.set(key, i + 1); // first (best) rank per memory
    if (!candidate.has(key)) candidate.set(key, e);
  });
  if (mode === 'floored-union') {
    // Admit lexical-ONLY hits that clear the identifier-precision bar.
    for (const [key, rank] of lexRank) {
      if (candidate.has(key)) continue;
      const e = lexEntry.get(key)!;
      if ((e.score ?? 0) >= minLex) candidate.set(key, e);
      void rank;
    }
  }

  const fused = [...candidate.entries()].map(([key, e]) => {
    const cr = cosRank.get(key);
    const lr = lexRank.get(key);
    const score = (cr !== undefined ? 1 / (k + cr) : 0) + (lr !== undefined ? lexWeight / (k + lr) : 0);
    return { ...e, score };
  });
  fused.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  return fused;
}

/**
 * Back-compat / strict-discipline wrapper: cosine-gated fusion — the result is the
 * cosine hits, with the lexical leg only re-ranking them (no lexical-only admits).
 */
export function fuseCosineGated(
  cosineHits: readonly MemoryEntry[],
  lexicalHits: readonly MemoryEntry[],
  k: number = DEFAULT_RRF_K,
): MemoryEntry[] {
  return fuse(cosineHits, lexicalHits, { mode: 'cosine-gated', k });
}
