/**
 * Gold-set replay — the retrieval tier of the memory-backend benchmark
 * (D-003/D-004). Replays every frozen query against one backend through
 * `backend.search()` and resolves ranked hits back to corpus keys via
 * the `metadata.corpus_key` stamp the seeder wrote.
 */

import type { MemoryBackend, MemoryEntry, SearchFloorPolicy } from '../backend';
import { aggregateByClass, latencyStats } from './metrics';
import type { GoldQuery, QueryOutcome, RetrievalRunResult } from './types';

export interface RetrievalOptions {
  /** The seeded pool to search. */
  scope: string;
  /** Hits requested per query (default 10 — recall@10 needs them). */
  limit?: number;
  /** Parallel search() calls (default 4). */
  concurrency?: number;
  /**
   * Absolute FP score floor passed through to backend.search (the push-path
   * relevance gate — D-003). Omit to replay UNFLOORED (raw recall); set to a
   * value to measure the floored push path (the floor sweep, P-031).
   */
  minScore?: number;
  /** Relative score-ratio trim passed through to backend.search. */
  minScoreRatio?: number;
  /** Hybrid-only: lexical admission bar passed through to backend.search (P-031 sweep). */
  minLexScore?: number;
  /** Hybrid-only: fusion mode passed through to backend.search (P-031 sweep). */
  fusionMode?: 'floored-union' | 'cosine-gated';
  /** Progress callback (done, total). */
  onProgress?: (done: number, total: number) => void;
}

/** Resolve one ranked hit list to deduped corpus keys. */
export function rankedCorpusKeys(hits: readonly MemoryEntry[]): string[] {
  const keys: string[] = [];
  const seen = new Set<string>();
  for (const hit of hits) {
    const key = hit.metadata?.corpus_key;
    if (typeof key !== 'string' || key.length === 0 || seen.has(key)) continue;
    seen.add(key);
    keys.push(key);
  }
  return keys;
}

/** Replay the gold set against one backend. */
export async function runGoldSet(
  backend: MemoryBackend,
  gold: readonly GoldQuery[],
  opts: RetrievalOptions,
): Promise<RetrievalRunResult> {
  const limit = opts.limit ?? 10;
  const concurrency = Math.max(1, opts.concurrency ?? 4);
  const outcomes: QueryOutcome[] = new Array(gold.length);

  let next = 0;
  let done = 0;
  // The floor/fusion pairing the sweep will run under, resolved ONCE (it is
  // loop-invariant). A sweep that sets a floor must name the fusion shape it is
  // measuring — see `SearchFloorPolicy`; 'floored-union' is what this harness
  // got by omission before that pairing was enforced, so it stays the fallback
  // and no historical bench number shifts.
  const floorPolicy: SearchFloorPolicy =
    opts.minScore !== undefined || opts.minScoreRatio !== undefined
      ? {
          ...(opts.minScore !== undefined ? { minScore: opts.minScore } : {}),
          ...(opts.minScoreRatio !== undefined ? { minScoreRatio: opts.minScoreRatio } : {}),
          fusionMode: opts.fusionMode ?? 'floored-union',
        }
      : { ...(opts.fusionMode !== undefined ? { fusionMode: opts.fusionMode } : {}) };

  async function worker(): Promise<void> {
    for (;;) {
      const i = next++;
      if (i >= gold.length) return;
      const q = gold[i];
      const t0 = performance.now();
      let hits: MemoryEntry[] = [];
      try {
        hits = await backend.search(q.query, {
          scope: opts.scope,
          limit,
          ...(opts.minLexScore !== undefined ? { minLexScore: opts.minLexScore } : {}),
          ...floorPolicy,
        });
      } catch {
        hits = []; // an unavailable backend scores zero, it doesn't crash the run
      }
      const ms = performance.now() - t0;
      outcomes[i] = {
        queryId: q.id,
        class: q.class,
        expected: q.expected,
        rankedKeys: rankedCorpusKeys(hits),
        rawHits: hits.length,
        ...(typeof hits[0]?.score === 'number' ? { topScore: hits[0].score } : {}),
        ...(typeof hits[0]?.text === 'string' ? { topText: hits[0].text } : {}),
        ms,
      };
      done += 1;
      opts.onProgress?.(done, gold.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, gold.length || 1) }, worker));

  const { byClass, overall } = aggregateByClass(outcomes);
  return {
    backend: backend.name,
    perQuery: outcomes,
    byClass,
    overall,
    latency: latencyStats(outcomes.map((o) => o.ms)),
  };
}
