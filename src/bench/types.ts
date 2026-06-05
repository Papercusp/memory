/**
 * Memory-backend benchmark — neutral types.
 *
 * The bench engine measures ANY `MemoryBackend` through the seam alone
 * (memory-backend-benchmark-2026-06-05 D-001): the same corpus is seeded
 * into every backend under test, the same frozen gold set is replayed,
 * and the same write-round-trip checks run — no backend-specific paths.
 * A `NoopBackend` run is the control: any check a no-op "passes"
 * measures nothing.
 *
 * Everything here is host-agnostic. The corpus/gold-set CONTENT and the
 * live wiring (credentials, PG schema, cost pricing) belong to the host
 * (the operator keeps them in packages/operator-core/lib/memory/bench).
 */

/** One corpus fact, keyed stably so ranked hits can be resolved back. */
export interface CorpusEntry {
  /**
   * Stable corpus key (e.g. the source topic-file slug). Seeding stamps
   * it onto each created entry as `metadata.corpus_key`; retrieval
   * resolves ranked hits back to keys through that stamp.
   */
  key: string;
  /** The fact body — seeded byte-identical into every backend (D-002). */
  text: string;
  /** Neutral kind tag (identity/preference/project/…). */
  kind?: string;
  /** One-line description (file backends use it for the index hook). */
  description?: string;
  /** Extra metadata persisted with the entry. */
  metadata?: Record<string, unknown>;
}

/** The four adversarial gold-set classes (D-003). */
export const GOLD_QUERY_CLASSES = [
  'lexical-gap',
  'exact-identifier',
  'hard-negative',
  'session-start-intent',
] as const;
export type GoldQueryClass = (typeof GOLD_QUERY_CLASSES)[number];

/** One frozen query→expected pair. */
export interface GoldQuery {
  id: string;
  class: GoldQueryClass;
  query: string;
  /**
   * Corpus keys that count as relevant. EMPTY for hard negatives —
   * those queries must surface nothing.
   */
  expected: string[];
  /** Optional authoring note (why this query / what it probes). */
  note?: string;
}

/** Result of seeding one corpus into one backend through the seam. */
export interface SeedManifest {
  backend: string;
  scope: string;
  /** corpus key → backend ids created for it (0..N per the contract). */
  ids: Record<string, string[]>;
  /** Corpus keys whose remember() persisted nothing (honest failures). */
  failed: string[];
  /** Wall-clock ms per remember() call, in corpus order. */
  rememberMs: number[];
  /** Total characters written (the embed-cost driver). */
  totalChars: number;
}

/** One replayed gold query with ranked, key-resolved results. */
export interface QueryOutcome {
  queryId: string;
  class: GoldQueryClass;
  expected: string[];
  /** Ranked corpus keys (deduped, unknown-key hits dropped). */
  rankedKeys: string[];
  /** Ranked raw hit count BEFORE key resolution (noise visibility). */
  rawHits: number;
  /** Backend-native score of the top hit, if reported. */
  topScore?: number;
  /** Search wall-clock ms. */
  ms: number;
}

/** Aggregated rank metrics for one query class (or overall). */
export interface RankMetrics {
  n: number;
  /** Mean precision@5 across queries (positives only). */
  p5: number;
  /** Mean recall@10 across queries (positives only). */
  r10: number;
  /** Mean reciprocal rank (positives only). */
  mrr: number;
  /**
   * Hard negatives: fraction of queries that returned ≥1 raw hit in the
   * top-5 (false-positive discipline). Undefined for positive classes.
   */
  fpAt5?: number;
  /** Median top-hit score (threshold-separability signal). */
  medianTopScore?: number;
}

export interface RetrievalRunResult {
  backend: string;
  perQuery: QueryOutcome[];
  byClass: Partial<Record<GoldQueryClass, RankMetrics>>;
  /** Positives-only overall (hard negatives reported via their class row). */
  overall: RankMetrics;
  /** Latency stats over every search call in the run. */
  latency: LatencyStats;
}

export interface LatencyStats {
  n: number;
  p50: number;
  p95: number;
  mean: number;
  max: number;
}

/** One write-round-trip probe spec (content supplied by the host). */
export interface RoundtripSpec {
  id: string;
  /** A novel fact NOT in the corpus. */
  fact: string;
  /** A paraphrase of the fact with low lexical overlap. */
  paraphrase: string;
  /** Replacement text for the update check. */
  updatedText: string;
  /** A near-duplicate restatement (dedup behavior probe). */
  nearDup: string;
  /**
   * Distinctive marker token that must survive the write (a cipher word
   * present in `fact`; extraction-transforming backends keep distinctive
   * content words). Default: the longest token of `fact`.
   */
  marker?: string;
  /** Marker for the updated text (default: longest token of updatedText). */
  updatedMarker?: string;
}

/** Structured outcome of one round-trip check (observed, not judged). */
export interface RoundtripOutcome {
  specId: string;
  /** remember() persisted ≥1 entry. */
  stored: boolean;
  /** The paraphrase search found the fact in the top 3. */
  paraphraseFound: boolean;
  /** Rank of the fact in the paraphrase search (1-based; 0 = miss). */
  paraphraseRank: number;
  /** update() round-tripped (get/search reflects the new text). */
  updateHonored: boolean;
  /** forget() round-tripped (get null after delete). */
  forgetHonored: boolean;
  /** Entries created by the near-dup write (0 = merged/dedup, 1+ = new). */
  nearDupNewEntries: number;
  /** remember() wall-clock ms (the fact write). */
  rememberMs: number;
  /** Error string when a verb threw (unavailable backends etc.). */
  error?: string;
}

/** Operation counters the host converts to $ with its own pricing. */
export interface OpCounts {
  remembers: number;
  searches: number;
  totalCharsWritten: number;
  totalCharsQueried: number;
}

/** One backend's full scorecard row-set (D-004). */
export interface BackendScorecard {
  backend: string;
  seeded: number;
  seedFailed: number;
  retrieval: RetrievalRunResult;
  roundtrips: RoundtripOutcome[];
  rememberP50Ms: number;
  /** $/1k ops, computed by the host's cost model; undefined = free/unknown. */
  costPer1kRemembers?: number;
  costPer1kSearches?: number;
  /** Scale curve points (corpus size → quality + latency). */
  scale?: Array<{ size: number; p5: number; mrr: number; searchP50Ms: number }>;
  /** Reach checklist — qualitative capability rows (D-004). */
  reach?: Record<string, string>;
  /** Op counters for the host's cost conversion. */
  ops: OpCounts;
}
