/**
 * Rank metrics + latency percentiles for the memory-backend benchmark.
 * Pure math — no I/O, no backend coupling.
 */

import type { GoldQueryClass, LatencyStats, QueryOutcome, RankMetrics } from './types';

/** Precision@k: relevant∩top-k / k. 0 when nothing was expected. */
export function precisionAtK(expected: readonly string[], ranked: readonly string[], k: number): number {
  if (expected.length === 0 || k <= 0) return 0;
  const exp = new Set(expected);
  const top = ranked.slice(0, k);
  const hits = top.filter((key) => exp.has(key)).length;
  return hits / k;
}

/** Recall@k: relevant∩top-k / |relevant|. 0 when nothing was expected. */
export function recallAtK(expected: readonly string[], ranked: readonly string[], k: number): number {
  if (expected.length === 0 || k <= 0) return 0;
  const exp = new Set(expected);
  const top = ranked.slice(0, k);
  const hits = top.filter((key) => exp.has(key)).length;
  return hits / expected.length;
}

/** Reciprocal rank of the FIRST relevant hit (1-based); 0 on a miss. */
export function reciprocalRank(expected: readonly string[], ranked: readonly string[]): number {
  const exp = new Set(expected);
  const idx = ranked.findIndex((key) => exp.has(key));
  return idx === -1 ? 0 : 1 / (idx + 1);
}

function mean(xs: readonly number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}

function median(xs: readonly number[]): number | undefined {
  if (xs.length === 0) return undefined;
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * Aggregate one query class (or a whole positives set) into RankMetrics.
 * Hard negatives get `fpAt5` (fraction with ≥1 RAW hit in the top 5)
 * instead of the positive-rank metrics — a vector store returns nearest
 * neighbors for ANY query, so what matters there is whether hits come
 * back at all and how their scores separate from real ones.
 */
export function aggregateOutcomes(outcomes: readonly QueryOutcome[]): RankMetrics {
  const negatives = outcomes.filter((o) => o.expected.length === 0);
  const positives = outcomes.filter((o) => o.expected.length > 0);
  const metrics: RankMetrics = {
    n: outcomes.length,
    p5: mean(positives.map((o) => precisionAtK(o.expected, o.rankedKeys, 5))),
    r10: mean(positives.map((o) => recallAtK(o.expected, o.rankedKeys, 10))),
    mrr: mean(positives.map((o) => reciprocalRank(o.expected, o.rankedKeys))),
  };
  const scored = outcomes.filter((o) => typeof o.topScore === 'number');
  const med = median(scored.map((o) => o.topScore as number));
  if (med !== undefined) metrics.medianTopScore = med;
  if (negatives.length > 0) {
    metrics.fpAt5 = negatives.filter((o) => o.rawHits > 0).length / negatives.length;
  }
  return metrics;
}

/** Group outcomes by class and aggregate each (plus a positives-only overall). */
export function aggregateByClass(outcomes: readonly QueryOutcome[]): {
  byClass: Partial<Record<GoldQueryClass, RankMetrics>>;
  overall: RankMetrics;
} {
  const byClass: Partial<Record<GoldQueryClass, RankMetrics>> = {};
  const classes = [...new Set(outcomes.map((o) => o.class))];
  for (const cls of classes) {
    byClass[cls] = aggregateOutcomes(outcomes.filter((o) => o.class === cls));
  }
  return { byClass, overall: aggregateOutcomes(outcomes.filter((o) => o.expected.length > 0)) };
}

/** Latency percentiles over raw wall-clock samples. */
export function latencyStats(samplesMs: readonly number[]): LatencyStats {
  if (samplesMs.length === 0) return { n: 0, p50: 0, p95: 0, mean: 0, max: 0 };
  const sorted = [...samplesMs].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))];
  return {
    n: sorted.length,
    p50: at(0.5),
    p95: at(0.95),
    mean: mean(sorted),
    max: sorted[sorted.length - 1],
  };
}
