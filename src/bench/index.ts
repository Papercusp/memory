/**
 * @papercusp/memory/bench — the backend-parameterized benchmark engine
 * (memory-backend-benchmark-2026-06-05 D-001/D-010).
 *
 * Measures ANY `MemoryBackend` through the neutral seam alone: seed the
 * same corpus, replay the same frozen gold set, run the same write
 * round-trips, collect latency + op counts, render one scorecard. The
 * corpus/gold-set CONTENT and live wiring (credentials, PG schema,
 * pricing) belong to the host.
 */

export {
  GOLD_QUERY_CLASSES,
  type BackendScorecard,
  type CorpusEntry,
  type GoldQuery,
  type GoldQueryClass,
  type LatencyStats,
  type OpCounts,
  type QueryOutcome,
  type RankMetrics,
  type RetrievalRunResult,
  type RoundtripOutcome,
  type RoundtripSpec,
  type SeedManifest,
} from './types';

export {
  aggregateByClass,
  aggregateOutcomes,
  latencyStats,
  precisionAtK,
  recallAtK,
  reciprocalRank,
} from './metrics';

export { seedCorpus, unseedCorpus, type SeedOptions } from './seed';
export { rankedCorpusKeys, runGoldSet, type RetrievalOptions } from './retrieval';
export { distinctiveToken, runRoundtrips, type RoundtripOptions } from './roundtrip';
export { generateSyntheticCorpus, mulberry32 } from './synthetic';
export { rememberP50, renderScorecardMarkdown } from './scorecard';
