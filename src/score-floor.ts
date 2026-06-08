/**
 * Relevance floor for search results (memory-backend-improve-and-hybrid P-001).
 *
 * A vector store returns the nearest neighbours for ANY query — so an
 * out-of-corpus query still surfaces noise (the bench's hard-negative
 * FP@5 = 100%). The fix is to DROP hits whose score is too low to be a real
 * match, applied on the auto-inject (push) path where no LLM is in the loop to
 * filter (memory-backend-improve-and-hybrid D-003).
 *
 * Two complementary gates, both opt-in via `SearchOptions`:
 *   - `minScore` (ABSOLUTE) — drop any hit below this score. Catches the
 *     hard-negative "nothing is close" case (the top score itself is low).
 *     Calibrated to the backend's score scale; the canonical/mem0 store scores
 *     by cosine similarity (0..1), and the bench separates hard-negatives
 *     (top ~0.385) from real hits (~0.51–0.58), so ~0.45 sits in the gap.
 *   - `minScoreRatio` (RELATIVE) — drop hits weaker than `ratio × topScore`.
 *     Catches the "one strong hit + a long mediocre tail" case without a magic
 *     absolute number. The stricter of the two floors wins.
 *
 * Entries WITHOUT a numeric score are never dropped (we can't judge them).
 * Pure — returns a new array, never mutates.
 */
import type { MemoryEntry } from './backend';

export interface ScoreFloorOptions {
  /** Absolute floor — drop hits with `score < minScore`. */
  minScore?: number;
  /** Relative floor — drop hits with `score < minScoreRatio × topScore` (0..1). */
  minScoreRatio?: number;
}

export function applyScoreFloor(
  entries: readonly MemoryEntry[],
  opts: ScoreFloorOptions,
): MemoryEntry[] {
  const { minScore, minScoreRatio } = opts;
  if (minScore === undefined && minScoreRatio === undefined) return [...entries];
  const scores = entries
    .map((e) => e.score)
    .filter((s): s is number => typeof s === 'number');
  if (scores.length === 0) return [...entries]; // unscored set — nothing to floor against
  const top = Math.max(...scores);
  const absFloor = minScore ?? -Infinity;
  const relFloor = minScoreRatio !== undefined ? minScoreRatio * top : -Infinity;
  const floor = Math.max(absFloor, relFloor);
  return entries.filter((e) => typeof e.score !== 'number' || e.score >= floor);
}
