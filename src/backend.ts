/**
 * The neutral `MemoryBackend` seam — the swap point for the persistent
 * memory store (generalize-memory-backend-swappable-2026-06-05, D-001).
 *
 * Deliberately a SMALL, store-agnostic surface: no mem0 vocabulary
 * (`user_id`, pgvector, vec tables, collections), no operator coupling.
 * A backend maps the neutral concepts onto its own store:
 *
 *   - `scope` — an opaque string naming a memory pool. The operator uses
 *     `<user-id>`, `harness:<slug>`, and the legacy `workspace:<id>`;
 *     the backend treats it as a partition key (mem0 maps it to its
 *     `user_id` filter; a file backend might map it to a directory).
 *   - `kind` — an optional caller-defined tag (`identity` / `preference`
 *     / `project` / `correction` in the operator). Backends without a
 *     first-class column store it in `metadata.kind`.
 *   - `text` — the fact body. Backends may transform on write (mem0's
 *     LLM fact-extraction can split one input into several entries, or
 *     decide nothing is memorable), which is why `remember` returns
 *     0..N ids.
 *
 * Capability methods (`rememberConversation`, `invalidate`) are OPTIONAL
 * — mem0-grade features a plain store shouldn't be forced to fake.
 * Callers feature-test (`backend.rememberConversation?.(…)`).
 *
 * Availability: `available()` is the non-throwing probe. Every other
 * method throws `MemoryUnavailableError` when the store is unreachable
 * or deliberately disabled — "unavailable" is NOT the same as "empty",
 * and silently dropping a write would lie to the caller.
 */

/**
 * What SCALE a `MemoryEntry.score` is expressed on.
 *
 * `score` has always been documented as "backend-native; ordering only", which
 * is correct and is precisely the problem: ordering-only values from DIFFERENT
 * backends land in the SAME telemetry column, and nothing downstream could tell
 * them apart. Making the scale machine-readable is what lets a reader refuse to
 * compare across scales instead of silently averaging them.
 *
 * The scales are not merely differently-calibrated — they answer different
 * questions, which is why no linear rescaling can reconcile them:
 *
 * - `cosine`  — a SIMILARITY. Absolute and comparable across calls: 0.9 means
 *               "very close to the query" no matter which call produced it. A
 *               bad query yields low numbers, so the score can reveal it.
 * - `rrf`     — a RANK, reciprocal-rank-fused: `Σ lexWeight/(k + rank)` over the
 *               legs an entry appears in. Bounded above by `(1+lexWeight)/(k+1)`
 *               (0.0328 at the k=60/lexWeight=1 defaults) and comparable ONLY
 *               within one call. Critically it is NOT a relevance measure at
 *               all: SOMETHING is always rank 1, so a meaningless query scores
 *               the same as a perfect one. An rrf score can never reveal a bad
 *               query — see context-injection-audit-2026-07-28 D-011.
 * - `lexical` — a backend-native token/keyword overlap from the embed-free
 *               fallback leg (WI-4214). Ordering only; typically small rational
 *               fractions.
 * - `unknown` — the backend declined to say. Treated as its own bucket and
 *               NEVER pooled with a labelled one.
 */
export type ScoreScale = 'cosine' | 'rrf' | 'lexical' | 'unknown';

/**
 * WHICH RETRIEVAL LEG(S) produced an entry, and at what rank within each.
 *
 * A fused `score` is a SUM over legs (see `ScoreScale.rrf`), and a sum is not
 * invertible: 1/(k+3) from cosine alone and 1/(k+7)+1/(k+9) from both legs are
 * different numbers with no way back to the legs that made them. So the moment
 * `fuse()` writes the fused score, the answer to "which leg found this?" is
 * gone — for every downstream reader, permanently.
 *
 * That matters because the two legs FAIL DIFFERENTLY and are fixed differently:
 * a cosine miss is a query/embedding problem, a lexical miss is a tokenization
 * or identifier problem. Telemetry that records only the fused score can report
 * a healthy-looking recall while one leg has been contributing nothing at all —
 * which is exactly what a live measurement found (see the `minLexScore` note in
 * `hybrid-fusion.ts`: a 1526-memory pool whose cosine candidates shared ZERO ids
 * with its lexical rows, for 336 of 337 consecutive recalls).
 *
 * Ranks are 1-based and LEG-LOCAL — a rank is only comparable within its own
 * leg and its own call. `undefined` means the entry was absent from that leg,
 * which is a real observation and not a zero.
 *
 * ⚠ This is a per-CALL retrieval fact, deliberately NOT `metadata`. `metadata`
 * is backend-passthrough state belonging to the STORED fact; a rank belongs to
 * the query that just ran. Putting it there invites a writer to persist it, and
 * a persisted rank is wrong the instant any other query runs.
 */
export interface RetrievalProvenance {
  /** 1-based rank in the cosine leg; undefined = the cosine leg did not return it. */
  cosineRank?: number;
  /** 1-based rank in the qualifying lexical leg; undefined = absent from it. */
  lexicalRank?: number;
  /**
   * The cosine leg's NATIVE pre-fusion score for this entry — the actual
   * similarity, on 0..1, that the relevance floor (`SearchOptions.minScore`)
   * was applied to. Undefined when the cosine leg did not return this entry
   * (`cosineRank` undefined) or returned it unscored.
   *
   * ⚠ THIS IS THE ONLY SURVIVING COPY. `fuse()` REPLACES `MemoryEntry.score`
   * with the fused RRF sum, so after fusion the native score is unrecoverable:
   * RRF is computed from RANKS, so the fused value carries no information about
   * how similar anything actually was. Two entries at cosine 0.95 and 0.59 fuse
   * to identical scores if they hold the same ranks.
   *
   * That is not a theoretical loss. It made "how relevant was what we
   * injected?" unanswerable from telemetry BY CONSTRUCTION — the recall table
   * recorded only the post-fusion value, whose ceiling is 0.0328, and an agent
   * comparing that column against the 0.58 cosine floor reported "99.9% of
   * injections are below the floor" (2026-07-28). The floor was working fine;
   * the two numbers were simply never on the same scale.
   */
  cosineScore?: number;
  /**
   * The lexical leg's NATIVE pre-fusion score, recorded for the same reason and
   * with the same lifetime as `cosineScore`. Undefined when the lexical leg did
   * not rank this entry (`lexicalRank` undefined) or returned it unscored.
   *
   * Note this is the score of the LEXICAL entry, which for a cross-leg hit is a
   * different row object than the one carried in the result (the cosine entry
   * wins the slot — it is canonical). Both legs' scores are therefore present
   * on a hit that neither leg alone could have produced.
   */
  lexicalScore?: number;
}

/** What ONE retrieval leg did on a single call. */
export interface LegRunStats {
  /**
   * Did this leg EXECUTE? `false` is a distinct observation from "ran and
   * returned nothing", and the two need opposite fixes — `cosine-gated` mode
   * deliberately never starts the lexical leg when the cosine leg is empty, and
   * a leg that threw degrades to `[]` as well. A reader that cannot tell those
   * apart reads a short-circuit as a retrieval failure.
   */
  ran: boolean;
  /** Rows the leg returned, pre-fusion. Undefined when it did not run. */
  candidates?: number;
  /**
   * Rows that cleared this leg's OWN admission bar and were therefore given a
   * rank in the fusion (the lexical leg's `minLexScore`). Equal to `candidates`
   * for a leg with no such bar. `candidates - qualifying` is the count the bar
   * removed — the number that says whether the bar is doing anything.
   */
  qualifying?: number;
  /**
   * The per-scope row budget IN EFFECT for this leg on this call — recorded,
   * never inferred later from a constant. Saturation (`candidates` at the
   * ceiling) means the pool was UNDER-SAMPLED, not that it held nothing more;
   * computing that against a retuned constant misjudges every older row. Same
   * contract as `RecallPoolStats.limit` on the operator's telemetry.
   */
  depth?: number;
}

/** What every leg did on one fused call — reported via `SearchOptions.onLegStats`. */
export interface SearchLegStats {
  cosine: LegRunStats;
  lexical: LegRunStats;
  /** The fusion mode in effect (`floored-union` | `cosine-gated`). */
  mode: string;
  /** Distinct entries in the FUSED candidate set, before any caller-side limit. */
  fused: number;
}

/** One stored fact, in the neutral shape every surface renders. */
export interface MemoryEntry {
  id: string;
  /** The fact body. */
  text: string;
  /** Caller-defined tag (e.g. identity/preference/project/correction). */
  kind?: string;
  /** The pool this entry lives in (opaque scope string). */
  scope: string;
  /** Relevance score for `search` results (backend-native; ordering only). */
  score?: number;
  /**
   * Which leg(s) produced this entry, for a multi-leg backend that fuses.
   * Set by `fuse()`; absent from single-leg backends and from stored entries
   * (it describes a retrieval, not a fact). Optional on purpose — no existing
   * constructor of this type is stranded by it.
   */
  retrieval?: RetrievalProvenance;
  /** Backend-passthrough metadata (anchors, provenance, timestamps, …). */
  metadata?: Record<string, unknown>;
}

export interface RememberOptions {
  /** The pool to write into (opaque scope string). Required. */
  scope: string;
  /** Optional kind tag; backends without a column store it in metadata.kind. */
  kind?: string;
  /** Arbitrary metadata persisted with the entry. */
  metadata?: Record<string, unknown>;
  /**
   * Store the text AS-IS — skip any extract/transform step the backend
   * would otherwise run on write (mem0 maps this to `infer: false`, so
   * the LLM fact-extraction is bypassed and exactly one entry is
   * created). Backends that never transform (file stores, noop) ignore
   * it. Bulk imports of already-curated facts set this so the corpus
   * lands byte-identical (memory-backend-benchmark D-008).
   */
  verbatim?: boolean;
  /**
   * OPT-IN federation egress (F0-2): true = this memory may federate to
   * other hives. Default false — memories are hive-private. Backends that
   * support federation should store this as a first-class column or in
   * metadata for capture-trigger filtering.
   */
  shareable?: boolean;
  /**
   * WRITE-TIME EMBED AUGMENTATION (EI-10048). When set (and different from
   * `text`), the backend (re)computes the entry's VECTOR from `embedText`
   * while the STORED text stays exactly `text` — the embedding is enriched
   * but the canonical body is untouched. Callers pass `text` plus resolved
   * reference titles (e.g. "WI-4028: <title>") so a ref-only memory ALSO
   * matches queries about the referenced item's TOPIC — the multi-hop recall
   * a flat store can't bridge (query-time graph fusion was rejected, D-001).
   * Doing the hop once at write time costs nothing at query time and never
   * perturbs ranking. Best-effort: a backend that can't separate stored-text
   * from embed-text (file stores, noop) ignores this and stores `text` as-is.
   */
  embedText?: string;
}

export interface SearchOptionsCommon {
  /**
   * One or more pools to search. `limit` applies PER SCOPE; the merged
   * result is sorted by score (desc) but NOT globally truncated —
   * callers slice if they need a global cap. Per-pool limits are what
   * the operator's fan-out semantics (user pool at one limit, harness
   * pools at another) need.
   */
  scope: string | readonly string[];
  /** Max hits per scope pool. Backend default applies when omitted. */
  limit?: number;
  /**
   * HYBRID-ONLY (memory-backend-improve-and-hybrid P-031). The lexical
   * admission bar for a lexical-only hit in floored-union mode. Falls back to
   * the backend's constructor default; non-hybrid backends ignore it. Plumbed
   * as a search-time option so the P-031 sweep can tune it per-call without
   * rebuilding the backend. (Its sibling `fusionMode` lives on
   * `SearchFloorPolicy` below, because it is load-bearing rather than a knob.)
   */
  minLexScore?: number;
  /**
   * HYBRID-ONLY. An ALTERNATE query text for the LEXICAL leg — the cosine leg
   * still searches `query`. Omit and both legs get `query` (byte-identical to
   * the single-query behaviour, which is what every caller but the operator's
   * injection path wants).
   *
   * WHY A SECOND QUERY RATHER THAN ONE CONCATENATED STRING
   * (context-injection-audit-2026-07-28 P-044 / F-L). The two legs want
   * DIFFERENT text, and until now one string had to serve both:
   *
   *  - the COSINE leg embeds the query as ONE vector, so every non-semantic
   *    token drags that vector off-topic. Measured live (D-041): the operator's
   *    mid-turn query carried `apps/operator-vite/src` twice plus a full
   *    component path against four content words, and the record that answered
   *    the question was retrieved ONLY by a query naming its literal id. The
   *    identifiers were not merely wasted there — they were DILUTING.
   *  - the LEXICAL leg is exactly the opposite: token-matching, embed-free, and
   *    the one leg that wins on identifiers.
   *
   * So the operator sends the prose to `query` and the prose+identifiers to
   * `lexicalQuery`. That is a QUERY-composition change only — per D-014,
   * NOTHING here re-ranks or reorders a leg's OUTPUT, because reordering a
   * fusion leg rewrites the fusion INPUT.
   *
   * ⚠ KEEP IT SHORT. `canonical-store.lexicalSearch` normalizes its score by
   * `tokens × 3`, so every token added to THIS string lowers the normalized
   * score of every hit it returns — and `minLexScore` is an absolute bar on
   * exactly that number. Genuine exact-identifier matches already span 0.33–0.96
   * (see DEFAULT_MIN_LEX_SCORE), so padding this query is not free: it can push
   * a real identifier match under the admission bar and REMOVE the hit the
   * identifier was supposed to rescue. Add a bounded handful of identifiers, not
   * a dump.
   */
  lexicalQuery?: string;
  /**
   * TEMPORAL-LITE (memory-temporal-lite-validity-windows-2026-07-11).
   * `asOf` = point-in-time read: only entries whose validity window covers
   * this ISO timestamp (valid_at ⇒ created_at when unset; < invalid_at).
   * `includeSuperseded` = opt-in to entries whose window has CLOSED
   * (superseded / soft-forgotten) — excluded from recall by default.
   * Entries touched by either control carry `metadata.validity:
   * { valid_at, invalid_at, superseded_by, status: 'current'|'superseded' }`.
   * Backends without validity semantics ignore both.
   */
  asOf?: string;
  includeSuperseded?: boolean;
  /**
   * OPTIONAL read-time diversity re-rank (EI-10230, MMR). Off by default —
   * omit for today's pure-relevance ordering. When set, the backend applies
   * a maximal-marginal-relevance pass AFTER any relevance floor + recency
   * decay, so the ADMITTED set is identical either way; only intra-top-K
   * ORDER changes (near-duplicate hits stop crowding out distinct facts).
   * `lambda` (0..1, default 1 = no-op) trades relevance for diversity; see
   * `diversityRerank` in `./diversity-rerank`. Backends that don't implement
   * it ignore this.
   */
  diversify?: { lambda?: number };
  /**
   * OPTIONAL: a PRECOMPUTED query embedding (EI-12992). When supplied, a backend
   * that embeds its query (Mem0Backend's cosine leg) uses this vector INSTEAD OF
   * re-embedding `query` — the caller already paid for one embed and is sharing
   * it across several `search()` calls (e.g. the operator's per-turn memory
   * injection fans out 3 independent pulls — user/harness/hive pools — over the
   * SAME query text; embedding it 3 times was ~3 serialized embed round-trips,
   * ~0.5-1s each, riding right up against the injection timeout). Backends that
   * don't embed (a lexical/file store) ignore this. Get one via
   * `backend.embedQuery?.(text)`.
   */
  vector?: number[];
  /**
   * HYBRID-ONLY, OPT-IN: report what each leg did on this call.
   *
   * A callback rather than a field on the result, because `search` returns
   * `MemoryEntry[]` through the shared `MemoryBackend` interface — widening that
   * return type would strand every backend and every caller to carry a value
   * almost none of them want. A callback is opt-in, costs nothing when omitted,
   * and cannot be forgotten into module state (which on a fan-out path would be
   * read by whichever concurrent call finished last).
   *
   * Invoked at most once per `search`, AFTER fusion, BEFORE the caller-side
   * limit — so `fused` reports the real candidate pool, not the slice. It is
   * called inside the backend's own try-scope: a throw from here must never
   * fail a search, so implementations wrap it.
   *
   * Per-ENTRY leg attribution is not here — it rides on `MemoryEntry.retrieval`,
   * so an entry that reaches a render surface still knows which leg found it.
   */
  onLegStats?: (stats: SearchLegStats) => void;
}

/**
 * THE FLOOR/FUSION PAIRING — a relevance floor may not be requested without
 * saying WHICH FUSION SHAPE it applies to.
 *
 * WHY THIS IS A TYPE AND NOT A DEFAULT (context-injection-audit-2026-07-28
 * P-001, re-scoped by D-020). The original P-001 read the bug as "the floor is
 * lost inside the backend". It is not: the floor works, and F-B/D-010 fixed the
 * push path by selecting `cosine-gated` per call. Fusion semantics are genuinely
 * CALLER-SPECIFIC — dedup and classification callers want unfloored
 * nearest-neighbour behaviour on purpose, and the PULL path legitimately admits
 * a strong identifier match with weak cosine (measured: a healthy 29.2% zero-hit
 * rate there, versus the push path's 0.0% before the fix). So per-call mode
 * selection is the correct design, not a bypass to close.
 *
 * What was genuinely left is that **the permissive mode is what you get by
 * omission**: `floored-union` admits lexical-only hits on `minLexScore` ALONE,
 * independently of the cosine floor, so a push-like caller added tomorrow could
 * set `minScore` believing it had bought precision and silently get none. That
 * is the same class as the founding bug — a 0.45 floor running at 0.03 for
 * months because nothing asserted it — and it is a DEFAULTS problem, not a
 * fusion-algorithm problem.
 *
 * The fix makes the unsafe combination INEXPRESSIBLE rather than merely
 * discouraged: pass a floor and you must state the fusion mode. Note the shape
 * deliberately permits `minScore: number | undefined` alongside an explicit
 * `fusionMode` — an env-tunable floor that resolves to undefined is still a
 * caller who has thought about fusion, which is the whole property being
 * enforced.
 *
 * Deliberately NOT done here: flipping the default to `cosine-gated` and making
 * the pull path opt in. That was D-020's other candidate, and it silently
 * changes retrieval for the eight existing callers that take the default — a
 * relevance change this plan's method rule (D-041) says must be measured per
 * caller against the live corpus, not assumed. This change alters no behaviour
 * at all; it only refuses a query that cannot state what it means.
 */
export type SearchFloorPolicy =
  | {
      /** No relevance floor requested — fusion mode is a free choice. */
      minScore?: undefined;
      minScoreRatio?: undefined;
      fusionMode?: 'floored-union' | 'cosine-gated';
    }
  | {
      /**
       * Relevance floor (memory-backend-improve-and-hybrid P-001). Opt-in,
       * applied on the auto-inject (push) path where no LLM filters the result
       * (D-003): a hit below the floor is dropped, so an out-of-corpus query
       * returns nothing instead of nearest-neighbour noise (the bench's
       * hard-negative FP@5 fix). `minScore` is an ABSOLUTE floor on the
       * backend's score scale (cosine similarity for the canonical/mem0 store);
       * `minScoreRatio` is a RELATIVE floor (× the top hit's score). The
       * stricter of the two wins. Backends whose score scale differs (or that
       * don't score) ignore these.
       */
      minScore?: number;
      minScoreRatio?: number;
      /**
       * REQUIRED once a floor is in play. `cosine-gated` seeds the candidate set
       * from cosine hits only, so membership is governed by the cosine floor and
       * the result is a real 0..K ceiling. `floored-union` admits lexical-only
       * hits independently of that floor — correct for recall-oriented callers,
       * wrong for a push path, and never what you should get by accident.
       */
      fusionMode: 'floored-union' | 'cosine-gated';
    };

export type SearchOptions = SearchOptionsCommon & SearchFloorPolicy;

export interface ListOptions {
  /** One or more pools to list. */
  scope: string | readonly string[];
  /** Filter to entries whose kind matches. */
  kind?: string;
  /** TEMPORAL-LITE point-in-time read — see `SearchOptions.asOf`. */
  asOf?: string;
  /** Include entries whose validity window has closed — see `SearchOptions`. */
  includeSuperseded?: boolean;
}

export interface UpdatePatch {
  /** Replacement fact body. */
  text?: string;
  /**
   * Metadata merge-patch. OPTIONAL for backends. mem0's OSS `update()` is
   * text-only, so the Mem0Backend rides the canonical-store merge path for
   * metadata (vec-safe `payload || patch`, no re-embed); other backends may
   * not implement it — check your backend before relying on it.
   */
  metadata?: Record<string, unknown>;
}

export type MemoryAvailability = { ok: true } | { ok: false; reason: string };

/**
 * Thrown by backend methods when the store is unreachable or disabled.
 * `reason` is a stable machine-readable token (e.g. `mem0_unavailable`,
 * `memory_backend_disabled`) callers can surface verbatim.
 */
export class MemoryUnavailableError extends Error {
  constructor(public readonly reason: string) {
    super(`memory backend unavailable: ${reason}`);
    this.name = 'MemoryUnavailableError';
  }
}

/**
 * The swappable store contract. Implementations: `Mem0Backend` (the
 * pgvector-backed mem0 store), `NoopBackend` (deliberate "no store"),
 * and out-of-lib backends registered via `registerMemoryBackend()`
 * (e.g. a Claude-topic-file bridge).
 */
export interface MemoryBackend {
  /** Stable backend identifier (registry key, diagnostics). */
  readonly name: string;

  /**
   * Non-throwing availability probe. `{ ok: false, reason }` means the
   * other methods will throw `MemoryUnavailableError(reason)`. A store
   * that is merely EMPTY is `{ ok: true }`.
   */
  available(): Promise<MemoryAvailability>;

  /**
   * Store one fact. Returns the ids of entries NEWLY created — a
   * backend may merge into existing entries or decide nothing is
   * memorable, so 0..N ids. Backends that can tell SHOULD also report
   * `storedEvents`: the count of store-affecting operations (new
   * inserts + merges into existing entries). `ids: [], storedEvents: 0`
   * means NOTHING was persisted (e.g. mem0's extractor failed or
   * declined) — callers use it to report honest capture failures
   * instead of assuming a resolved promise stored something (EI-25).
   * Backends that can't distinguish merges may omit it; callers fall
   * back to `ids.length`.
   */
  remember(text: string, opts: RememberOptions): Promise<{ ids: string[]; storedEvents?: number }>;

  /** Semantic/text search. See `SearchOptions` for limit semantics. */
  search(query: string, opts: SearchOptions): Promise<MemoryEntry[]>;

  /**
   * OPTIONAL capability: EMBED-FREE lexical search over the same pools —
   * the degraded-path fallback for when the semantic leg is unusable (an
   * overloaded/saturated embedder times `search` out while the store
   * itself is fine — WI-4214). Must never invoke an embedder; a plain
   * token/keyword match is the contract, and scores are backend-native
   * (ordering only, NOT on the cosine scale). Backends whose `search` is
   * already lexical (file stores) alias it; callers feature-test
   * (`backend.searchLexical?.(…)`).
   */
  searchLexical?(query: string, opts: SearchOptions): Promise<MemoryEntry[]>;

  /**
   * The scale of `score` on entries returned by THIS backend's `search()`.
   *
   * OPTIONAL on purpose: `MemoryBackend` is implemented by test doubles and
   * out-of-lib backends, and a REQUIRED field on a shared interface invalidates
   * every construction site at once. An omitted declaration reads as `unknown`,
   * which downstream keeps in its own bucket rather than pooling it — the
   * honest degradation.
   *
   * Declared PER-METHOD rather than per-backend because one backend can emit
   * two scales: `HybridBackend.search` returns fused `rrf`, while its
   * `searchLexical` returns the raw `lexical` leg. A single backend-level
   * property would therefore be wrong for exactly the degraded path that most
   * needs to be distinguishable.
   */
  readonly scoreScale?: ScoreScale;

  /**
   * The scale of `score` on entries returned by `searchLexical()`. Only
   * meaningful when `searchLexical` is implemented.
   */
  readonly lexicalScoreScale?: ScoreScale;

  /** Enumerate entries in the given pools (insertion order unspecified). */
  list(opts: ListOptions): Promise<MemoryEntry[]>;

  /** Fetch one entry by id, or null when it doesn't exist. */
  get(id: string): Promise<MemoryEntry | null>;

  /** Delete one entry by id. Resolves even if the id is already gone. */
  forget(id: string): Promise<void>;

  /** Patch one entry. See `UpdatePatch` for what backends must accept. */
  update(id: string, patch: UpdatePatch): Promise<void>;

  /**
   * OPTIONAL capability: extract memorable facts from a conversation
   * window (mem0's LLM fact-extraction). Backends without an extractor
   * omit it; callers feature-test before invoking.
   */
  rememberConversation?(
    messages: ReadonlyArray<{ role: string; content: string }>,
    opts: RememberOptions,
  ): Promise<{ ids: string[]; storedEvents?: number }>;

  /** OPTIONAL capability: drop cached clients/state so the next call rebuilds. */
  invalidate?(): void;

  /**
   * OPTIONAL capability: close one ENTRY's validity window (temporal-lite
   * soft-forget / supersession) WITHOUT deleting it — the entry drops out
   * of default recall but stays retrievable via `includeSuperseded` /
   * `asOf`. `supersededBy` records the replacing entry's id. First-wins:
   * returns false when no OPEN entry matched (unknown id, or its window
   * was already closed — the earlier closure stands). NOT the cache-drop
   * `invalidate()` above. Backends without validity semantics omit it;
   * callers feature-test (`backend.invalidateEntry?.(…)`).
   */
  invalidateEntry?(id: string, opts?: { supersededBy?: string }): Promise<boolean>;

  /**
   * OPTIONAL capability (EI-12992): embed `text` ONCE with this backend's own
   * embedder, for a caller that will issue several `search()` calls against the
   * SAME query text and wants to pay the embed cost once — pass the result as
   * `SearchOptions.vector` on each call. Returns null when the embedder is
   * unavailable/degraded (never throws) — callers fall back to per-call
   * embedding (today's behavior) on a null. Backends that don't embed (a
   * lexical/file store) omit it; callers feature-test (`backend.embedQuery?.(…)`).
   */
  embedQuery?(text: string): Promise<number[] | null>;
}

/** Normalize a `scope: string | readonly string[]` arg to a de-duped array. */
export function scopesOf(scope: string | readonly string[]): string[] {
  const arr = typeof scope === 'string' ? [scope] : [...scope];
  return [...new Set(arr.filter((s) => typeof s === 'string' && s.length > 0))];
}
