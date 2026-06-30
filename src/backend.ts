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
}

export interface SearchOptions {
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
   * Relevance floor (memory-backend-improve-and-hybrid P-001). Opt-in, applied
   * on the auto-inject (push) path where no LLM filters the result (D-003): a
   * hit below the floor is dropped, so an out-of-corpus query returns nothing
   * instead of nearest-neighbour noise (the bench's hard-negative FP@5 fix).
   * `minScore` is an ABSOLUTE floor on the backend's score scale (cosine
   * similarity for the canonical/mem0 store); `minScoreRatio` is a RELATIVE
   * floor (× the top hit's score). The stricter of the two wins. Backends
   * whose score scale differs (or that don't score) ignore these.
   */
  minScore?: number;
  minScoreRatio?: number;
  /**
   * HYBRID-ONLY overrides (memory-backend-improve-and-hybrid P-031). The hybrid
   * backend fuses a cosine + a lexical leg; `fusionMode` picks the fusion shape
   * and `minLexScore` is the lexical admission bar for a lexical-only hit in
   * floored-union mode. Both fall back to the backend's constructor defaults;
   * non-hybrid backends ignore them. Plumbed as search-time options so the
   * P-031 sweep can tune them per-call without rebuilding the backend.
   */
  fusionMode?: 'floored-union' | 'cosine-gated';
  minLexScore?: number;
}

export interface ListOptions {
  /** One or more pools to list. */
  scope: string | readonly string[];
  /** Filter to entries whose kind matches. */
  kind?: string;
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
}

/** Normalize a `scope: string | readonly string[]` arg to a de-duped array. */
export function scopesOf(scope: string | readonly string[]): string[] {
  const arr = typeof scope === 'string' ? [scope] : [...scope];
  return [...new Set(arr.filter((s) => typeof s === 'string' && s.length > 0))];
}
