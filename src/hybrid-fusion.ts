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
import type { MemoryEntry, RetrievalProvenance } from './backend';

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
  /**
   * OPT-IN: report the leg-level counts this fusion observed.
   *
   * `lexicalQualifying` is the reason this lives here rather than being
   * recomputed by the caller: the `minLexScore` bar is applied in exactly one
   * place below, and a second copy of that predicate outside this function would
   * drift from it silently — reporting a bar's effect from a stale copy of the
   * bar is worse than not reporting it. Everything else is passed through for
   * free while we are already counting.
   *
   * Invoked at most once, after the candidate set is built. Never throws into
   * the fusion: callers wrap it.
   */
  onFusionStats?: (stats: {
    cosineCandidates: number;
    lexicalCandidates: number;
    /** Lexical hits that cleared `minLexScore` and were given a rank. */
    lexicalQualifying: number;
    /** Distinct entries in the fused candidate set. */
    fused: number;
  }) => void;
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
  // Two linkage forms must BOTH resolve to the same slot:
  //   1. `metadata.link_id` — the canonical id the write-through stamps onto a
  //      projection. mem0 extraction may REWORD the cosine text, so the cosine
  //      side must be reachable by its OWN id (a text key alone misses this —
  //      observed live: one hybrid write surfaced twice on recall).
  //   2. normalized verbatim TEXT — covers a MIGRATED corpus, where the same
  //      fact was stored verbatim in both legs (identical text) WITHOUT a
  //      link_id stamp (so `~/.claude` is kept exactly as-is).
  // So each COSINE entry registers BOTH aliases (own id + normalized text) to
  // one slot; a lexical entry resolves via link_id first, then text.
  const norm = (t: string): string => t.trim().replace(/\s+/g, ' ');

  // Candidate set + each candidate's cosine rank (undefined = lexical-only). The
  // COSINE entry wins the slot when a memory is in both legs (it's canonical).
  const slotOf = new Map<string, string>(); // alias → slot key
  const cosRank = new Map<string, number>();
  const candidate = new Map<string, MemoryEntry>();
  cosineHits.forEach((e, i) => {
    const slot = e.id || norm(e.text);
    if (!slot) return;
    if (!slotOf.has(slot)) slotOf.set(slot, slot);
    const textAlias = e.text ? norm(e.text) : '';
    if (textAlias && !slotOf.has(textAlias)) slotOf.set(textAlias, slot);
    const key = slotOf.get(slot)!;
    if (!cosRank.has(key)) cosRank.set(key, i + 1); // first (best) rank per memory
    if (!candidate.has(key)) candidate.set(key, e);
  });

  const lexKeyOf = (e: MemoryEntry): string => {
    const linkId = typeof e.metadata?.link_id === 'string' ? (e.metadata.link_id as string) : undefined;
    const textAlias = e.text ? norm(e.text) : '';
    // Resolve to a cosine slot when either alias matches; else key on the
    // lexical entry's own identity (link_id keeps two projections of the same
    // canonical row collapsed even when the cosine leg missed the query).
    return (
      (linkId !== undefined ? slotOf.get(linkId) : undefined) ??
      (textAlias ? slotOf.get(textAlias) : undefined) ??
      linkId ??
      textAlias ??
      e.id
    );
  };

  // Lexical rank (1-based) + the lexical entry, keyed by cross-leg slot (first wins).
  //
  // ⚠ `minLexScore` gates conferring a RANK, not merely lexical-ONLY ADMISSION — and
  // that is a fix, not a tightening for its own sake (WI-7154). The bar used to apply
  // only in the `floored-union` block below, which meant that under `cosine-gated` —
  // THE PUSH-PATH DEFAULT, i.e. every per-turn memory injection — it was dead code:
  // every lexical hit, however weak, still handed its slot a full second RRF term.
  //
  // That term decides the ranking. At k=60 over a ~29-candidate cosine list the ENTIRE
  // cosine ordering spans ~0.005 (1/61 → 1/89) while one cross-leg co-occurrence is
  // worth up to ~0.013, so membership in the lexical list outweighed relevance: a
  // candidate at cosine rank 29 (barely above the floor) outranked cosine ranks 2 and 3.
  //
  // Whether a pool GOT that bonus was then decided by its SIZE, because the lexical leg
  // pulls `depth` PER SCOPE. Measured live 2026-08-02 on the initialize query: the
  // 56-memory hive pool had 36/56 = 64% of its corpus in the lexical list, so its cosine
  // hits were nearly always also lexical hits and were always boosted; the 1526-memory
  // harness pool had 36/1526 = 2.4% coverage and its cosine candidates shared ZERO ids
  // with its lexical rows, so they were never boosted. Harness took 5 of the cosine
  // top-12 and 0 of the 12 fused slots, on 336 of 337 consecutive real recalls. The
  // property is scale-INVERTED: the more memories a pool holds, the less likely its best
  // rows are admitted, so writing more memories made recall strictly worse.
  //
  // The decisive measurement is that ALL 108 hits conferring a bonus scored below even
  // the loosest calibrated bar (0.3) — not one was a real lexical match. The score is
  // normalized by query-token count, so an 11-token prose query cannot reach 0.4 (that
  // needs ~5 tokens hitting `name`); this leg exists for EXACT-IDENTIFIER recall, and on
  // a generic prose query it should contribute nothing rather than decide the outcome.
  // Applying the bar restores the honest cosine ranking (harness 5 of 12) and leaves
  // genuine identifier matches — which score 0.33-0.96 — boosting exactly as before.
  //
  // Rank DENSELY over the qualifying set (filter first, then enumerate) rather than
  // keeping each hit's index in the unfiltered list. A disqualified hit must not leave
  // a rank GAP: the RRF term is 1/(k+rank), so an original-index rank would silently
  // shrink a genuine identifier match's bonus in proportion to how much weak noise
  // happened to precede it — making the bar's effect depend on the noise it removed.
  //
  // An UNSCORED entry is never dropped — the same contract `applyScoreFloor` states
  // ("entries without a numeric score are never dropped; we can't judge them"). The
  // live lexical leg always scores (token overlap), so this only spares hand-built
  // callers and test doubles from being silently stripped of their boost.
  const qualifyingLexical = lexicalHits.filter(
    (e) => typeof e.score !== 'number' || e.score >= minLex,
  );
  const lexRank = new Map<string, number>();
  const lexEntry = new Map<string, MemoryEntry>();
  qualifyingLexical.forEach((e, i) => {
    const key = lexKeyOf(e);
    if (key && !lexRank.has(key)) {
      lexRank.set(key, i + 1);
      lexEntry.set(key, e);
    }
  });
  if (mode === 'floored-union') {
    // Admit lexical-ONLY hits. Every entry here already cleared `minLex` above (that
    // filter is now the single application of the bar), so admission needs no re-check.
    for (const [key, rank] of lexRank) {
      if (candidate.has(key)) continue;
      candidate.set(key, lexEntry.get(key)!);
      void rank;
    }
  }

  const fused = [...candidate.entries()].map(([key, e]) => {
    const cr = cosRank.get(key);
    const lr = lexRank.get(key);
    const score = (cr !== undefined ? 1 / (k + cr) : 0) + (lr !== undefined ? lexWeight / (k + lr) : 0);
    // Stamp WHICH LEGS produced this entry, alongside the score that fuses them.
    // The score is a SUM, so it cannot be inverted back to its terms — this is
    // the only point in the system where the answer is still available. See
    // `RetrievalProvenance` for why it is not folded into `metadata`.
    //
    // Both ranks omitted is unreachable: a candidate slot exists only because
    // some leg ranked it. The object is still written unconditionally so a
    // reader never has to distinguish "no provenance recorded" from "no legs".
    //
    // The NATIVE per-leg scores are stamped here for the same reason and at the
    // same last-possible moment (`RetrievalProvenance.cosineScore`): the line
    // below overwrites `e.score` with the RRF sum, and RRF is computed from
    // RANKS, so nothing downstream can recover how SIMILAR anything actually
    // was. Recording the rank without the score leaves the ordering auditable
    // and the RELEVANCE unmeasurable, which is the state that made the recall
    // telemetry unable to answer the one question it exists for.
    //
    // Read each score from the entry that leg actually returned, never from
    // `e`: for a cross-leg hit `e` is the COSINE entry (it wins the slot as
    // canonical), so `e.score` is a cosine score and taking it as the lexical
    // one would silently record the same number twice, in a shape that reads
    // like two independent legs agreeing. A lexical-ONLY admit (`floored-union`)
    // inverts that — there `e` IS the lexical entry and no cosine score exists
    // at all, which the `cr !== undefined` guard is what keeps honest.
    const cosNative = cr !== undefined ? e.score : undefined;
    const lexNative = lr !== undefined ? lexEntry.get(key)?.score : undefined;
    const retrieval: RetrievalProvenance = {
      ...(cr !== undefined ? { cosineRank: cr } : {}),
      ...(lr !== undefined ? { lexicalRank: lr } : {}),
      // Finite-only: an unscored leg entry is ABSENT, not zero. Recording a 0
      // would land inside the real value range and read as "maximally
      // irrelevant, admitted anyway" — a defect-shaped reading of a hand-built
      // or test-double entry that simply never carried a score.
      ...(typeof cosNative === 'number' && Number.isFinite(cosNative) ? { cosineScore: cosNative } : {}),
      ...(typeof lexNative === 'number' && Number.isFinite(lexNative) ? { lexicalScore: lexNative } : {}),
    };
    return { ...e, score, retrieval };
  });
  fused.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  if (opts.onFusionStats) {
    // Telemetry must never be able to fail a search — the same posture the
    // operator's recall-stats writer takes. A throwing reporter loses its own
    // numbers and nothing else.
    try {
      opts.onFusionStats({
        cosineCandidates: cosineHits.length,
        lexicalCandidates: lexicalHits.length,
        lexicalQualifying: qualifyingLexical.length,
        fused: fused.length,
      });
    } catch {
      /* swallow — a reporter must not break retrieval */
    }
  }
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
