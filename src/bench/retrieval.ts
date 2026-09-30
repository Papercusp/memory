/**
 * Gold-set replay — the retrieval tier of the memory-backend benchmark
 * (D-003/D-004). Replays every frozen query against one backend through
 * `backend.search()` and resolves ranked hits back to corpus keys via
 * the `metadata.corpus_key` stamp the seeder wrote.
 */

import type { MemoryBackend, MemoryEntry, SearchFloorPolicy } from '../backend';
import { aggregateByClass, latencyStats } from './metrics';
import type { CandidateHit, GoldQuery, QueryOutcome, RetrievalRunResult } from './types';

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
  /**
   * Keep every returned hit on its outcome (`QueryOutcome.candidates`), for an
   * admission-filter arm that must judge exactly what this floor admitted.
   * Off by default: the text payload is only worth holding when it is used.
   */
  captureCandidates?: boolean;
  /**
   * Keep going when `backend.search` throws: the query scores as an empty hit
   * list with `QueryOutcome.error` set. OFF by default — a run with ANY errored
   * query then REJECTS with `searchFailureReason`, because an errored query is
   * indistinguishable from a genuine miss in every metric, and a bench over a
   * failing backend otherwise prints confident zeros. Opt in only where the
   * caller surfaces `searchFailureReason` itself (a multi-backend comparison).
   */
  tolerateSearchErrors?: boolean;
  /** Progress callback (done, total). */
  onProgress?: (done: number, total: number) => void;
}

/** One hit as a capturable candidate (see `RetrievalOptions.captureCandidates`). */
export function toCandidateHit(hit: MemoryEntry): CandidateHit {
  const key = hit.metadata?.corpus_key;
  return {
    id: hit.id,
    key: typeof key === 'string' && key.length > 0 ? key : null,
    text: hit.text,
    ...(typeof hit.score === 'number' ? { score: hit.score } : {}),
  };
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

/**
 * Why a replay's metrics cannot be trusted, or null when every search ran. The
 * search-side twin of `seedFailureReason`: names the errored count and the first
 * error, so a refusal says what broke instead of only that something did.
 */
export function searchFailureReason(run: { readonly perQuery: readonly QueryOutcome[] }): string | null {
  const errored = run.perQuery.filter((o) => o !== undefined && o.error !== undefined);
  if (errored.length === 0) return null;
  const first = errored[0];
  return `gold replay incomplete: ${errored.length}/${run.perQuery.length} searches threw — first error (${first.queryId}): ${first.error}`;
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
      let error: string | undefined;
      try {
        hits = await backend.search(q.query, {
          scope: opts.scope,
          limit,
          ...(opts.minLexScore !== undefined ? { minLexScore: opts.minLexScore } : {}),
          // A per-query leg split (GoldQuery.lexicalQuery). Passed through
          // ONLY when the arm set one, so an ordinary gold set replays with
          // both legs on `query` exactly as before.
          ...(q.lexicalQuery ? { lexicalQuery: q.lexicalQuery } : {}),
          ...floorPolicy,
        });
      } catch (e) {
        // Scored as an empty list so the run can finish, but RECORDED: the query
        // measured nothing, and the run is refused below unless tolerated.
        hits = [];
        error = e instanceof Error ? e.message : String(e);
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
        ...(opts.captureCandidates ? { candidates: hits.map(toCandidateHit) } : {}),
        ...(error !== undefined ? { error } : {}),
        ms,
      };
      done += 1;
      opts.onProgress?.(done, gold.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, gold.length || 1) }, worker));

  const failure = searchFailureReason({ perQuery: outcomes });
  if (failure !== null && !opts.tolerateSearchErrors) {
    throw new Error(`${failure} — refusing to report metrics over searches that never ran (backend ${backend.name})`);
  }

  const { byClass, overall } = aggregateByClass(outcomes);
  return {
    backend: backend.name,
    perQuery: outcomes,
    byClass,
    overall,
    latency: latencyStats(outcomes.map((o) => o.ms)),
  };
}
