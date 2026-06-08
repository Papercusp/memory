/**
 * Cosine-gated reciprocal-rank fusion for the HybridBackend
 * (memory-backend-improve-and-hybrid P-020).
 *
 * The hybrid fuses two retrieval legs over the same canonical store:
 *   - the COSINE leg (semantic / paraphrase — wins the lexical-gap class), and
 *   - the LEXICAL leg (exact tokens / identifiers — wins the exact-identifier class).
 *
 * Fusion is COSINE-GATED: the result set is exactly the cosine leg's hits (which
 * the caller has already FP-floored via SearchOptions.minScore, so an out-of-corpus
 * query yields none — the hard-negative discipline). The lexical leg only RE-RANKS
 * that set: an entry that also ranks high lexically gets a reciprocal-rank boost,
 * lifting exact matches to the top WITHOUT admitting lexical noise the cosine gate
 * already rejected. So:
 *   - paraphrase recall — cosine finds it, no lexical boost, still returned;
 *   - exact-identifier precision — cosine returns it, lexical boosts it toward #1;
 *   - hard-negative — cosine gate empty → nothing returned.
 *
 * RRF score = Σ 1/(k+rank) over the legs an entry appears in. k (standard 60)
 * dampens the weight of low ranks. The fused score is written onto entry.score
 * (ordering-only, per the MemoryEntry contract).
 */
import type { MemoryEntry } from './backend';

export const DEFAULT_RRF_K = 60;

export function fuseCosineGated(
  cosineHits: readonly MemoryEntry[],
  lexicalHits: readonly MemoryEntry[],
  k: number = DEFAULT_RRF_K,
): MemoryEntry[] {
  const lexRank = new Map<string, number>();
  lexicalHits.forEach((e, i) => {
    if (e.id && !lexRank.has(e.id)) lexRank.set(e.id, i + 1);
  });
  const fused = cosineHits.map((e, i) => {
    const cosRank = i + 1;
    const lr = e.id ? lexRank.get(e.id) : undefined;
    const score = 1 / (k + cosRank) + (lr !== undefined ? 1 / (k + lr) : 0);
    return { ...e, score };
  });
  fused.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  return fused;
}
