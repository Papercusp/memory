/**
 * FP-floor sweep (memory-backend-improve-and-hybrid P-031 / D-006).
 *
 * The auto-inject (push) path floors recall by an absolute cosine score so an
 * out-of-corpus query injects nothing (the hard-negative discipline). Too low a
 * floor lets false positives through; too high a floor floors out real answers.
 * This sweep replays the frozen gold set at several candidate floors over ONE
 * seeded store and reports, per floor, the precision/recall tradeoff so the
 * F1-maximizing value can be locked as the default.
 *
 * Query-level scoring (the floor is a binary admit/reject gate):
 *   - a POSITIVE query (has an expected answer) is a TP if its answer is in the
 *     top-k after flooring, else a FN (the floor buried a real hit);
 *   - a HARD-NEGATIVE query (no answer) is a FP if it still returns any hit, else
 *     a TN (correctly floored to nothing).
 * precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R).
 */
import type { MemoryBackend } from '../backend';
import { runGoldSet } from './retrieval';
import type { GoldQuery, QueryOutcome } from './types';

export interface QueryPRF {
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
}

/** Query-level precision/recall/F1 over a replay's outcomes (positives judged @k). */
export function queryLevelPRF(outcomes: readonly QueryOutcome[], k = 5): QueryPRF {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (const o of outcomes) {
    if (o.expected.length === 0) {
      if (o.rawHits > 0) fp += 1; // hard-negative returned something → false positive
    } else {
      const exp = new Set(o.expected);
      const hit = o.rankedKeys.slice(0, k).some((key) => exp.has(key));
      if (hit) tp += 1;
      else fn += 1; // positive's answer not in top-k (floored or just missed)
    }
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { tp, fp, fn, precision, recall, f1 };
}

export interface FloorPoint {
  floor: number;
  /** Hard-negative false-positive rate (fraction returning any hit) — →0 is the goal. */
  fpAt5: number;
  /** Overall recall@10 over positives — must not regress vs the unfloored run. */
  r10: number;
  /** Overall MRR over positives. */
  mrr: number;
  /** Exact-identifier class MRR (the column the hybrid must keep ~0.99). */
  exactIdMrr?: number;
  /** Positives that returned NOTHING at this floor (the floor buried them). */
  positivesEmptied: number;
  prf: QueryPRF;
}

export interface FloorSweepResult {
  points: FloorPoint[];
  /** The F1-maximizing floor (ties broken toward the LOWER floor = higher recall). */
  bestFloor: number;
  bestF1: number;
}

export interface FloorSweepOptions {
  scope: string;
  /** Candidate floors to try (0 = unfloored baseline). */
  floors: readonly number[];
  limit?: number;
  concurrency?: number;
  judgeK?: number;
  onProgress?: (floor: number, done: number, total: number) => void;
}

/**
 * Replay the gold set against ONE already-seeded backend at each candidate floor.
 * The floor is a search-time parameter, so no re-seeding is needed between floors.
 */
export async function runFloorSweep(
  backend: MemoryBackend,
  gold: readonly GoldQuery[],
  opts: FloorSweepOptions,
): Promise<FloorSweepResult> {
  const judgeK = opts.judgeK ?? 5;
  const points: FloorPoint[] = [];
  for (const floor of opts.floors) {
    const res = await runGoldSet(backend, gold, {
      scope: opts.scope,
      limit: opts.limit ?? 10,
      concurrency: opts.concurrency ?? 4,
      ...(floor > 0 ? { minScore: floor } : {}),
      onProgress: opts.onProgress ? (d, t) => opts.onProgress!(floor, d, t) : undefined,
    });
    const positivesEmptied = res.perQuery.filter((o) => o.expected.length > 0 && o.rawHits === 0).length;
    points.push({
      floor,
      fpAt5: res.byClass['hard-negative']?.fpAt5 ?? 0,
      r10: res.overall.r10,
      mrr: res.overall.mrr,
      ...(res.byClass['exact-identifier']?.mrr !== undefined ? { exactIdMrr: res.byClass['exact-identifier']!.mrr } : {}),
      positivesEmptied,
      prf: queryLevelPRF(res.perQuery, judgeK),
    });
  }
  // F1-max; ties → lower floor (favor recall + less anchoring harm).
  let best = points[0];
  for (const p of points) if (p.prf.f1 > (best?.prf.f1 ?? -1)) best = p;
  return { points, bestFloor: best?.floor ?? 0, bestF1: best?.prf.f1 ?? 0 };
}

/** One markdown table over the sweep points. */
export function renderFloorSweepMarkdown(result: FloorSweepResult): string {
  const lines: string[] = [];
  lines.push('| floor | FP@5 (hard-neg) | R@10 | MRR | exact-id MRR | pos. emptied | precision | recall | F1 |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const p of result.points) {
    const mark = p.floor === result.bestFloor ? ' ⭐' : '';
    lines.push(
      `| ${p.floor.toFixed(2)}${mark} | ${(p.fpAt5 * 100).toFixed(0)}% | ${(p.r10 * 100).toFixed(0)}% | ${p.mrr.toFixed(2)} | ` +
        `${p.exactIdMrr !== undefined ? p.exactIdMrr.toFixed(2) : '—'} | ${p.positivesEmptied} | ` +
        `${p.prf.precision.toFixed(2)} | ${p.prf.recall.toFixed(2)} | ${p.prf.f1.toFixed(3)} |`,
    );
  }
  lines.push('');
  lines.push(`**F1-max floor: ${result.bestFloor.toFixed(2)}** (F1 ${result.bestF1.toFixed(3)}).`);
  return lines.join('\n');
}
