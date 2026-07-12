/**
 * Read-time diversity re-rank (MMR) over search results (EI-10230,
 * su-ideate "stop near-duplicate memories crowding the top-K").
 *
 * Every ranking pass in this store (mem0's own semantic+BM25+entity fusion,
 * `applyScoreFloor`, the recency-decay re-order in mem0-backend.ts) optimizes
 * for RELEVANCE only. Nothing penalizes REDUNDANCY, so when a scope holds many
 * near-duplicate rows (repetitive self-checkpoints, "grounds fact X"
 * observations, loop notes), pure-relevance ranking spends the whole top-K
 * budget on near-copies of the single best hit and starves the 2nd/3rd
 * genuinely distinct fact.
 *
 * `diversityRerank` is a pure, OPTIONAL final pass: greedy maximal-marginal-
 * relevance selection —
 *
 *   next = argmax_i  λ·score_i − (1−λ)·max_{j∈selected} sim(i, j)
 *
 * λ=1 (the default) is an exact no-op / identity ordering (mirrors
 * `score-floor.ts`'s "off means off" contract) — this module NEVER drops or
 * adds entries, only reorders the input array. Callers apply it AFTER any
 * relevance floor / decay re-order, so the admitted SET is unaffected by
 * whether diversity re-ranking is on; only intra-top-K ORDER changes.
 *
 * `similarity` is caller-supplied so this module stays free of any embedding
 * dependency (pure + unit-testable in isolation, same seam shape as
 * `score-floor.ts`). `lexicalSimilarity` / `textSimilarity` below are the
 * trigram-Jaccard fallback proxy named in the proposal for when no embedder /
 * stored vector is available; a caller with real vectors handy (the
 * preferred EI-9694 "reuse stored vectors, don't re-embed" path) should pass
 * a cosine-similarity function over those instead.
 */
import type { MemoryEntry } from './backend';

export interface DiversityRerankOptions {
  /**
   * Relevance/diversity trade-off, 0..1. 1 (default) = pure relevance,
   * identity ordering (no-op). 0 = pure diversity (ignores score entirely
   * after the first pick). Values are clamped to [0,1].
   */
  lambda?: number;
  /** Pairwise similarity between two entries, expected roughly on a 0..1 scale. */
  similarity: (a: MemoryEntry, b: MemoryEntry) => number;
}

/**
 * Greedy MMR re-rank. Pure — returns a new array in NEW order; never mutates
 * `entries` and never adds/drops elements (same length, same members, in and
 * out — only order can change).
 */
export function diversityRerank(
  entries: readonly MemoryEntry[],
  opts: DiversityRerankOptions,
): MemoryEntry[] {
  const lambda = clamp01(opts.lambda ?? 1);
  if (entries.length <= 1 || lambda >= 1) return [...entries];
  const { similarity } = opts;

  // Normalize scores into 0..1 so they sit on the same scale as `similarity`
  // (unscored entries default to 0 — they contribute nothing to the
  // relevance term but can still be picked once diversity dominates).
  const rawScores = entries.map((e) => (typeof e.score === 'number' ? e.score : 0));
  const maxScore = Math.max(0, ...rawScores) || 1;

  const remaining = entries.map((e, i) => ({ e, norm: rawScores[i] / maxScore }));
  const selected: MemoryEntry[] = [];

  while (remaining.length > 0) {
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const cand = remaining[i];
      let maxSim = 0;
      for (const s of selected) {
        const sim = similarity(cand.e, s);
        if (sim > maxSim) maxSim = sim;
      }
      const val = lambda * cand.norm - (1 - lambda) * maxSim;
      if (val > bestVal) {
        bestVal = val;
        bestIdx = i;
      }
    }
    selected.push(remaining[bestIdx].e);
    remaining.splice(bestIdx, 1);
  }
  return selected;
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 1;
  return Math.min(1, Math.max(0, n));
}

/** Lowercased, whitespace-collapsed character trigram set of `s`. */
function trigrams(s: string): Set<string> {
  const norm = s.toLowerCase().replace(/\s+/g, ' ').trim();
  if (norm.length === 0) return new Set();
  if (norm.length < 3) return new Set([norm]);
  const out = new Set<string>();
  for (let i = 0; i <= norm.length - 3; i++) out.add(norm.slice(i, i + 3));
  return out;
}

/**
 * Trigram-Jaccard lexical similarity, 0..1 — the embedder-free redundancy
 * proxy: two near-verbatim paraphrases score high, two unrelated facts score
 * near 0. Not as good a redundancy signal as cosine over real embeddings,
 * but requires nothing beyond the entry text (safe default / test fallback).
 */
export function lexicalSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const ta = trigrams(a);
  const tb = trigrams(b);
  if (ta.size === 0 || tb.size === 0) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  const union = ta.size + tb.size - inter;
  return union === 0 ? 0 : inter / union;
}

/** `DiversityRerankOptions.similarity` adapter over `MemoryEntry.text`. */
export function textSimilarity(a: MemoryEntry, b: MemoryEntry): number {
  return lexicalSimilarity(a.text, b.text);
}
