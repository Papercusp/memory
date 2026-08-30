/**
 * `Mem0Backend` — the mem0/pgvector store as ONE implementation of the
 * neutral `MemoryBackend` seam (generalize-memory-backend-swappable
 * D-002).
 *
 * Everything mem0-shaped is confined here:
 *   - the neutral `scope` string maps onto mem0's `user_id` filter
 *     (`filters: { user_id: scope }` on search/getAll, `userId: scope`
 *     on add);
 *   - the neutral `kind` is stored/read via `metadata.kind` (mem0 has
 *     no first-class kind column);
 *   - mem0's PGVector adapter has no OR/IN filter support, so a
 *     multi-scope search/list fans out one query per pool and merges;
 *   - mem0's `add()` runs LLM fact-extraction and reports per-row
 *     events (`ADD`/`UPDATE`/`NONE`/`DELETE`); `remember` returns the
 *     ids of `ADD` rows only — newly created entries (the contract's
 *     0..N semantics);
 *   - mem0's OSS `update()` is text-only, so a metadata patch rides the
 *     canonical-store merge path (vec-safe `payload || patch`) instead.
 *
 * The mem0-internal machinery (client TTL/poison-cache, embedder
 * cascade, canonical store, re-embed) stays in ./mem0-client and
 * friends; this adapter only translates shapes.
 */

import { applyScoreFloor } from './score-floor';
import { createTextSimilarity, diversityDisabledByEnv, diversityRerank } from './diversity-rerank';
import { embedAndUpsertVector } from './vec-write';
import {
  MemoryUnavailableError,
  scopesOf,
  type ListOptions,
  type MemoryAvailability,
  type MemoryBackend,
  type MemoryEntry,
  type RememberOptions,
  type SearchOptions,
  type UpdatePatch,
} from './backend';
import {
  embedForCurrentClient,
  getMemoryClient,
  invalidateEntryCanonical,
  invalidateMemoryClient,
  lexicalSearchCanonical,
  updateMemoryPayload,
  vectorSearchCanonical,
  type Mem0Row,
  type MemoryClient,
} from './mem0-client';

/**
 * Temporal-lite read controls as mem0 FILTER keys. mem0 forwards filters to
 * the CanonicalVectorStore verbatim, whose splitTemporalControls picks these
 * two out before the payload post-filter loop (P-002). String values so the
 * `Record<string, string>` lexical seam accepts them unchanged.
 */
function temporalFilters(opts: { asOf?: string; includeSuperseded?: boolean }): Record<string, string> {
  return {
    ...(opts.asOf !== undefined ? { as_of: opts.asOf } : {}),
    ...(opts.includeSuperseded ? { include_superseded: 'true' } : {}),
  };
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const DEFAULT_SEARCH_LIMIT = 8;
/** mem0's getAll caps at topK; high so list isn't silently truncated. */
const LIST_TOP_K = 5000;

/**
 * Maximum number of lexical scope pulls a single backend call may have in
 * flight.  `memory:search` normally supplies the user's pool plus every
 * harness pool in the workspace (102 pools were present during the
 * 2026-08-30 OOM).  Starting one promise per pool made every returned JSONB
 * payload stay reachable until the whole fan-out completed.  Workers below
 * merge each scope immediately, so this is also a hard cap on the number of
 * result arrays retained by one call.
 *
 * Keep this at or below the canonical store's four-query admission ceiling;
 * one spare connection in its five-slot PG pool remains available to the
 * semantic/read and health paths.
 */
export const LEXICAL_SCOPE_CONCURRENCY = 4;

/** Bounded-concurrency worker map. Results are deliberately not collected: a
 * caller that needs to retain every per-item array defeats the memory bound.
 */
async function forEachWithConcurrency<T>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<void>,
): Promise<void> {
  if (items.length === 0) return;
  let next = 0;
  const workerCount = Math.max(1, Math.min(Math.floor(limit), items.length));
  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      const index = next++;
      if (index >= items.length) return;
      await fn(items[index], index);
    }
  });
  await Promise.all(workers);
}

// The worker cap above is per invocation. Several request workers (or the two
// legs of multiple HybridBackend calls) can still invoke this method at once,
// so keep a second process-wide lease around the payload-bearing seam. The
// canonical store has the same defense for direct callers; this layer also
// protects injected/alternate lexical implementations from the old global
// Promise.all amplification.
type LexicalScopeWaiter = () => void;
let lexicalScopePullsInFlight = 0;
const lexicalScopeWaiters: LexicalScopeWaiter[] = [];

async function withLexicalScopeAdmission<T>(run: () => Promise<T>): Promise<T> {
  await new Promise<void>((resolve) => {
    if (lexicalScopePullsInFlight < LEXICAL_SCOPE_CONCURRENCY) {
      lexicalScopePullsInFlight += 1;
      resolve();
      return;
    }
    lexicalScopeWaiters.push(resolve);
  });
  try {
    return await run();
  } finally {
    const next = lexicalScopeWaiters.shift();
    if (next) next();
    else lexicalScopePullsInFlight -= 1;
  }
}

/**
 * Search-time memory decay (recency ranking bias) — the OSS-side
 * equivalent of mem0 platform's "Memory Decay", which does NOT exist in
 * mem0ai OSS ≤3.0.3 (the OSS dist has no decay/recency code; it is a
 * hosted-platform feature). Applied as a pure RE-ORDERING bias after the
 * relevance floor: an entry's sort key is `score × recencyFactor(age)`,
 * but its REPORTED score stays the raw mem0 combined score, so every
 * score-threshold consumer (P-016 dedup-on-write, the P-001/D-003
 * relevance floor, orient's recall floor) keeps its calibration — decay
 * changes what ranks first, never what qualifies.
 *
 * The factor is bounded below by DECAY_FLOOR, so an old-but-strong match
 * loses at most (1 − floor) of its sort key: decay nudges near-ties
 * toward fresh memories; it never buries a clearly better old one.
 */
function decayEnabled(): boolean {
  return process.env.PAPERCUSP_MEMORY_DECAY !== '0';
}
function decayHalfLifeDays(): number {
  const v = Number(process.env.PAPERCUSP_MEMORY_DECAY_HALF_LIFE_DAYS);
  return Number.isFinite(v) && v > 0 ? v : 30;
}
function decayFloor(): number {
  const v = Number(process.env.PAPERCUSP_MEMORY_DECAY_FLOOR);
  return Number.isFinite(v) && v >= 0 && v <= 1 ? v : 0.9;
}
/** Multiplicative sort-key factor in [floor, 1]; 1 when age is unknown. */
export function recencyFactor(tsMs: number | null, nowMs: number): number {
  if (tsMs === null) return 1;
  const ageDays = Math.max(0, (nowMs - tsMs) / 86_400_000);
  const floor = decayFloor();
  return floor + (1 - floor) * Math.pow(0.5, ageDays / decayHalfLifeDays());
}

// Diversity re-rank kill switch (EI-10230) — now `diversityDisabledByEnv()`,
// shared from ./diversity-rerank so the SAME switch covers every seam that
// applies the pass (this backend and HybridBackend's fused list). It was
// private here while there was one caller; a second seam would otherwise have
// gained a kill switch that only half-worked.
/** Freshness timestamp of a mem0 search row (updatedAt, else createdAt). */
function rowTimestampMs(row: Mem0Row): number | null {
  for (const key of ['updatedAt', 'createdAt'] as const) {
    const raw = row[key];
    if (typeof raw === 'string' || typeof raw === 'number') {
      const t = new Date(raw).getTime();
      if (Number.isFinite(t)) return t;
    }
  }
  return null;
}

/**
 * Freshness timestamp of a CANONICAL payload row (the batched multi-scope
 * path, EI-12962). mem0 stamps `updated_at`/`created_at` into the payload it
 * stores (snake_case on this wire, unlike the search-row camelCase above);
 * absent/unparseable ⇒ null, which `recencyFactor` treats as neutral.
 */
function canonicalPayloadTimestampMs(payload: Record<string, unknown>): number | null {
  for (const key of ['updated_at', 'updatedAt', 'created_at', 'createdAt'] as const) {
    const raw = payload?.[key];
    if (typeof raw === 'string' || typeof raw === 'number') {
      const t = new Date(raw).getTime();
      if (Number.isFinite(t)) return t;
    }
  }
  return null;
}

/**
 * Read one result row's event. mem0's add() has TWO wire shapes: the
 * infer (LLM-extraction) path reports a top-level `event`, while the
 * `infer: false` path nests it as `metadata.event` (verified against
 * mem0ai 3.0.3 dist — addToVectorStore's no-infer branch). Accept both.
 */
function rowEvent(row: { event?: unknown; metadata?: unknown } | null | undefined): string {
  const direct = row?.event;
  if (typeof direct === 'string') return direct.toUpperCase();
  const nested = (row?.metadata as { event?: unknown } | undefined)?.event;
  return typeof nested === 'string' ? nested.toUpperCase() : '';
}

/**
 * mem0's `client.add(...)` returns `{ results: Array<{ id, event, … }> }`.
 * Extract the ids of newly-inserted rows ('ADD' events). 'UPDATE' and
 * 'NONE' rows are existing memories the extractor merged into — not new
 * inserts. (Moved here from the operator's persist-anchors.ts; this is
 * mem0 result-shape parsing and belongs behind the backend.)
 */
export function extractAddedIds(result: unknown): string[] {
  const rows = Array.isArray((result as { results?: unknown } | null)?.results)
    ? ((result as { results: Array<{ id?: unknown; event?: unknown; metadata?: unknown }> }).results)
    : [];
  const out: string[] = [];
  for (const row of rows) {
    if (rowEvent(row) !== 'ADD') continue;
    const id = typeof row.id === 'string' ? row.id : null;
    if (id && UUID_RE.test(id)) out.push(id);
  }
  return out;
}

/**
 * Count the store-affecting events ('ADD' + 'UPDATE') in a mem0 `add()`
 * result. An UPDATE is a legitimate store (the extractor merged the fact
 * into an existing memory), so `extractAddedIds` alone under-reports
 * whether anything was persisted. A swallowed extraction failure (mem0
 * catches its LLM error internally and resolves with `{ results: [] }`)
 * yields 0 — the signal callers use to report an HONEST capture failure
 * instead of a false success (EI-25).
 */
export function extractStoredEventCount(result: unknown): number {
  const rows = Array.isArray((result as { results?: unknown } | null)?.results)
    ? ((result as { results: Array<{ event?: unknown; metadata?: unknown }> }).results)
    : [];
  let n = 0;
  for (const row of rows) {
    const event = rowEvent(row);
    if (event === 'ADD' || event === 'UPDATE') n += 1;
  }
  return n;
}

/** Map one raw mem0 row to the neutral entry shape. */
function toEntry(row: Mem0Row, scope: string): MemoryEntry {
  const metadata = row.metadata ?? undefined;
  const kind = typeof metadata?.kind === 'string' ? metadata.kind : undefined;
  return {
    id: row.id,
    text: row.memory ?? '',
    ...(kind !== undefined ? { kind } : {}),
    scope,
    ...(typeof row.score === 'number' ? { score: row.score } : {}),
    ...(metadata !== undefined ? { metadata } : {}),
  };
}

/**
 * Map a raw canonical-store row (id + jsonb payload) onto the neutral entry
 * shape — the lexical-fallback sibling of `toEntry` (which maps mem0's
 * search-row shape). The memory text lives at `payload.data`; `user_id` is
 * the scope key (redundant with the fan-out scope, dropped); everything else
 * in the payload is caller-visible metadata (kind, anchors, provenance, …).
 */
function canonicalRowToEntry(
  row: { id: string; payload: Record<string, unknown>; score?: number },
  scope: string,
): MemoryEntry {
  const { data, user_id: _userId, ...rest } = (row.payload ?? {}) as {
    data?: unknown;
    user_id?: unknown;
  } & Record<string, unknown>;
  const kind = typeof rest.kind === 'string' ? rest.kind : undefined;
  return {
    id: row.id,
    text: typeof data === 'string' ? data : String(data ?? ''),
    ...(kind !== undefined ? { kind } : {}),
    scope,
    ...(typeof row.score === 'number' ? { score: row.score } : {}),
    ...(Object.keys(rest).length > 0 ? { metadata: rest } : {}),
  };
}

/** Merge fan-out results, de-duping by id (keep the higher-scored hit). */
function mergeById(entries: MemoryEntry[]): MemoryEntry[] {
  const byId = new Map<string, MemoryEntry>();
  for (const e of entries) {
    const prior = byId.get(e.id);
    if (!prior || (e.score ?? 0) > (prior.score ?? 0)) byId.set(e.id, e);
  }
  return [...byId.values()];
}

export interface Mem0BackendDeps {
  /** Test seam — defaults to the real cached mem0 client accessor. */
  getClient?: () => Promise<MemoryClient | null>;
  /** Test seam — the metadata merge-patch (defaults to the real canonical-store path). */
  updatePayload?: (id: string, patch: Record<string, unknown>) => Promise<boolean>;
  /** Test seam — the embed-free canonical lexical query (defaults to the real store path). */
  lexicalSearch?: (
    query: string,
    topK: number,
    filters: Record<string, string>,
  ) => Promise<Array<{ id: string; payload: Record<string, unknown>; score?: number }>>;
  /** Test seam — the validity-window close (defaults to the real canonical-store path). */
  invalidateEntry?: (id: string, opts: { supersededBy?: string }) => Promise<boolean>;
  /**
   * Test seam — write-time embed augmentation (EI-10048): overwrite a new
   * row's vector with `embed(embedText)` under the current mode, stored text
   * untouched. Defaults to the real resolve-embedder + memory_vec_<mode>
   * upsert. Best-effort (returns false, never throws).
   */
  reembedVector?: (id: string, scope: string, embedText: string) => Promise<boolean>;
  /**
   * Test seam — embed a query ONCE with the current client's embed fn for the
   * batched multi-scope search (EI-12962). null ⇒ the caller falls back to the
   * legacy per-scope `client.search` path. Defaults to `embedForCurrentClient`.
   */
  embedQuery?: (text: string) => Promise<number[] | null>;
  /**
   * Test seam — the precomputed-vector canonical search the batched path fans
   * out per scope (EI-12962). Defaults to `vectorSearchCanonical`.
   */
  vectorSearch?: (
    vector: number[],
    topK: number,
    filters: Record<string, string>,
  ) => Promise<Array<{ id: string; payload: Record<string, unknown>; score?: number }>>;
}

/**
 * Thrown when the query embed blows the caller's opt-in
 * `SearchOptions.embedTimeoutMs` (EI-21348316803580175).
 *
 * The message carries the literal token `embed_budget_exceeded` because the
 * operator's `embedFailureReason()` classifies embed failures BY MESSAGE. That
 * classification is what routes this into the existing embed-free lexical
 * fallback; an unclassified throw would surface as an opaque handler error —
 * strictly worse than the slow search this budget replaces.
 */
export class EmbedBudgetExceededError extends Error {
  constructor(readonly budgetMs: number) {
    super(`embed_budget_exceeded: query embed exceeded ${budgetMs}ms budget`);
    this.name = 'EmbedBudgetExceededError';
  }
}

/**
 * Await `embed()` but give up after `budgetMs`. `undefined`/`<= 0` is a bare
 * pass-through — byte-identical to awaiting `embed()` directly, which is what
 * keeps every caller that has not opted in unchanged.
 *
 * `MemoryBackend.embedQuery` takes no `AbortSignal`, so an over-budget embed
 * cannot be cancelled — it is abandoned and its late settle swallowed. That is
 * not pure waste here: the in-flight embed keeps warming the module's
 * short-TTL coalescing cache, so the NEXT call for the same text can still hit
 * it. (`libs/generic/search`'s sibling helper does pass a derived signal; its
 * `Embedder` contract accepts one and returns a non-nullable vector, which is
 * why that helper cannot simply be imported here.)
 */
async function embedWithinBudget(
  embed: () => Promise<number[] | null>,
  budgetMs: number | undefined,
): Promise<number[] | null> {
  if (budgetMs === undefined || budgetMs <= 0) return embed();
  const pending = embed();
  return await new Promise<number[] | null>((resolve, reject) => {
    let settled = false;
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };
    const timer = setTimeout(
      () => finish(() => reject(new EmbedBudgetExceededError(budgetMs))),
      budgetMs,
    );
    pending.then(
      (v) => finish(() => resolve(v)),
      (e) => finish(() => reject(e)),
    );
  });
}

export class Mem0Backend implements MemoryBackend {
  readonly name = 'mem0';

  /** pgvector similarity — an absolute, cross-call-comparable measure. */
  readonly scoreScale = 'cosine' as const;

  /** The embed-free fallback scores token-overlap fractions, NOT cosine
   *  (see `searchLexical` below) — a different scale from the same backend. */
  readonly lexicalScoreScale = 'lexical' as const;
  private readonly getClient: () => Promise<MemoryClient | null>;
  private readonly updatePayload: (id: string, patch: Record<string, unknown>) => Promise<boolean>;
  private readonly lexicalSearch: (
    query: string,
    topK: number,
    filters: Record<string, string>,
  ) => Promise<Array<{ id: string; payload: Record<string, unknown>; score?: number }>>;
  private readonly invalidateEntryStore: (id: string, opts: { supersededBy?: string }) => Promise<boolean>;
  private readonly reembedVector: (id: string, scope: string, embedText: string) => Promise<boolean>;
  /** PUBLIC (EI-12992): satisfies MemoryBackend.embedQuery? — a caller issuing
   *  several search() calls against the same query text embeds once here and
   *  passes the result as SearchOptions.vector on each call. */
  readonly embedQuery: (text: string) => Promise<number[] | null>;
  private readonly vectorSearch: (
    vector: number[],
    topK: number,
    filters: Record<string, string>,
  ) => Promise<Array<{ id: string; payload: Record<string, unknown>; score?: number }>>;

  constructor(deps: Mem0BackendDeps = {}) {
    this.getClient = deps.getClient ?? getMemoryClient;
    this.updatePayload = deps.updatePayload ?? updateMemoryPayload;
    this.lexicalSearch = deps.lexicalSearch ?? lexicalSearchCanonical;
    this.invalidateEntryStore = deps.invalidateEntry ?? invalidateEntryCanonical;
    // scope is unused by the default (memory_id is globally unique + the vec
    // table is per-mode, not per-scope) but kept in the seam for symmetry.
    this.reembedVector =
      deps.reembedVector ?? ((id, _scope, embedText) => embedAndUpsertVector(id, embedText));
    // Batched-path seams (EI-12962). A custom `getClient` WITHOUT explicit
    // batched seams DISABLES batching (embedQuery → null ⇒ legacy fallback):
    // `embedForCurrentClient`/`vectorSearchCanonical` are only coherent with
    // the REAL module client — a swapped-in client's vec space could silently
    // mismatch them, and test stubs expect the per-scope `client.search` contract.
    this.embedQuery =
      deps.embedQuery ?? (deps.getClient ? async () => null : embedForCurrentClient);
    this.vectorSearch = deps.vectorSearch ?? vectorSearchCanonical;
  }

  async available(): Promise<MemoryAvailability> {
    const client = await this.getClient().catch(() => null);
    return client ? { ok: true } : { ok: false, reason: 'mem0_unavailable' };
  }

  /** Resolve the client or throw the contract's unavailable error. */
  private async client(): Promise<MemoryClient> {
    const client = await this.getClient().catch(() => null);
    if (!client) throw new MemoryUnavailableError('mem0_unavailable');
    return client;
  }

  async remember(text: string, opts: RememberOptions): Promise<{ ids: string[]; storedEvents: number }> {
    const client = await this.client();
    const metadata: Record<string, unknown> = { ...(opts.metadata ?? {}) };
    if (opts.kind !== undefined) metadata.kind = opts.kind;
    if (opts.shareable !== undefined) metadata.shareable = opts.shareable;
    // `verbatim` → mem0's `infer: false`: skip the LLM fact-extraction and
    // embed + insert the raw text as exactly one ADD (D-008 — bulk seeding).
    const result = await client.add(text, {
      userId: opts.scope,
      metadata,
      ...(opts.verbatim ? { infer: false } : {}),
    });
    const ids = extractAddedIds(result);
    // Entity-linking parity for verbatim writes (owner directive
    // 2026-07-10: enable mem0 v3 entity linking). mem0's `infer: false`
    // branch early-returns BEFORE its Phase-7 entity linking, so verbatim
    // rows — memory:remember's default — would never join the entity
    // graph that search()'s entity boost ranks against, while
    // extraction-path and update() rows do. Call the same per-memory
    // linker mem0's update() uses. It is a private API, hence
    // feature-detected and best-effort: a mem0 upgrade that renames it
    // makes this a silent no-op, and a linking failure never fails the
    // write (mirrors mem0's own non-fatal entity-store posture).
    if (opts.verbatim && ids.length > 0) {
      const linker = (client as {
        _linkEntitiesForMemory?: (memoryId: string, text: string, filters: Record<string, string>) => Promise<void>;
      })._linkEntitiesForMemory;
      if (typeof linker === 'function') {
        for (const id of ids) {
          await linker.call(client, id, text, { user_id: opts.scope }).catch(() => {});
        }
      }
    }
    // Write-time embed augmentation (EI-10048): when the caller supplies an
    // enriched embed-text (clean body + resolved reference titles), overwrite
    // each new row's VECTOR with embed(embedText) while the stored
    // payload.data stays = text. A ref-only memory then also matches queries
    // about the referenced item's TOPIC (multi-hop recall the flat store
    // can't bridge; query-time graph fusion rejected, D-001). Best-effort +
    // non-fatal — mirrors the entity-linker posture above: a failure leaves
    // the baseline clean-text vector.
    if (opts.embedText && opts.embedText !== text && ids.length > 0) {
      for (const id of ids) {
        await this.reembedVector(id, opts.scope, opts.embedText).catch(() => {});
      }
    }
    return { ids, storedEvents: extractStoredEventCount(result) };
  }

  async search(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    const client = await this.client();
    const limit = opts.limit ?? DEFAULT_SEARCH_LIMIT;
    // Scope invariant (P-003): every search fans out per scope with a `user_id`
    // filter, so a query only ever races its OWN pool (a user / harness / project),
    // never the whole table. This is what keeps recall up at scale — the bench's
    // 10k single-pool MRR collapse (0.86→0.05) is a NO-scoping worst case that
    // scoping structurally avoids. A re-rank pass would only matter if one scope
    // ever neared ~1k entries (not the case today). An empty scope produces zero
    // pulls (scopesOf drops blanks), never an unscoped full-table scan.
    const scopes = scopesOf(opts.scope);
    // id → freshness ts for the decay re-order below (collected from the raw
    // rows because `toEntry` deliberately drops mem0's createdAt/updatedAt).
    const tsById = new Map<string, number | null>();

    // EI-12962 — MULTI-scope batched path: embed the query ONCE, then fan out
    // per-scope pgvector queries with the precomputed vector. `client.search`
    // re-embeds the query on EVERY call, so an N-scope pull cost N embeds
    // (~200-450ms each, largely serialized on the embedder) — the operator
    // chat's 20-harness injection reliably blew its 5s deadline and delivered
    // ZERO memories. One embed + N ~10ms SQL queries keeps the same per-scope
    // ranking + floors. Tradeoff, deliberate: this path skips mem0's
    // entity-graph boost (single-scope pulls keep it); a fast plain-cosine
    // result beats a timed-out boosted one. A null embed (embedder down /
    // client cold) falls back to the legacy per-scope path below.
    //
    // EI-12992 — CROSS-CALL shared embed: `opts.vector` (a caller-precomputed
    // embedding, from a prior `embedQuery()` call — see MemoryBackend.embedQuery)
    // takes this SAME batched path even for a SINGLE scope, skipping the embed
    // entirely. This is how a caller issuing several search() calls for the SAME
    // query text (e.g. the operator's per-turn memory injection, which pulls
    // user+harness+hive pools) collapses N embeds into 1. The entity-graph-boost
    // tradeoff above applies identically here — it's the CALLER's call whether to
    // pass `vector` for a pool that would otherwise keep the boost (see
    // injection.ts for which pools opt in).
    let merged: MemoryEntry[] | null = null;
    if (opts.vector || scopes.length > 1) {
      // EI-21348316803580175: bound the query embed when the caller opted in.
      // Unbounded, this await inherited the sidecar's own 15s timeout and waited
      // out a saturated embed sidecar — the whole regression.
      //
      // ⚠ A BUDGET MISS THROWS; it must never become a null vector. `merged`
      // staying null falls through to the LEGACY per-scope path below, which
      // re-embeds ONCE PER SCOPE — so under the exact saturation this budget
      // exists to survive, degrading to null would turn one blocked embed into
      // N more. Throwing is what lets the caller degrade to the embed-free
      // lexical leg instead.
      const vector =
        opts.vector ?? (await embedWithinBudget(() => this.embedQuery(query), opts.embedTimeoutMs));
      if (vector) {
        const pulls = scopes.map(async (scope) => {
          const rows = await this.vectorSearch(vector, limit, {
            user_id: scope,
            ...temporalFilters(opts),
          });
          return rows.map((row) => {
            tsById.set(row.id, canonicalPayloadTimestampMs(row.payload));
            return canonicalRowToEntry(row, scope);
          });
        });
        merged = mergeById((await Promise.all(pulls)).flat());
      }
    }

    if (merged === null) {
      const pulls = scopes.map(async (scope) => {
        // mem0's Memory.search reads `topK` (default 20) and IGNORES a
        // `limit` key — passing only `limit` silently over-fetched and
        // broke the seam's per-scope limit contract (caught by the
        // memory-backend-benchmark P-007 run). Keep `limit` for any
        // non-mem0 MemoryClient test doubles.
        const r = await client.search(query, {
          filters: { user_id: scope, ...temporalFilters(opts) },
          topK: limit,
          limit,
        });
        return (r.results ?? []).map((row) => {
          tsById.set(row.id, rowTimestampMs(row));
          return toEntry(row, scope);
        });
      });
      merged = mergeById((await Promise.all(pulls)).flat());
    }
    const ranked = merged.sort(
      (a, b) => (b.score ?? 0) - (a.score ?? 0),
    );
    // Relevance floor (P-001 / D-003): drop hits too weak to be real matches, so
    // an out-of-corpus query returns nothing instead of nearest-neighbour noise.
    const floored = applyScoreFloor(ranked, { minScore: opts.minScore, minScoreRatio: opts.minScoreRatio });
    // Memory decay (recency ranking bias) — re-order AFTER the floor so the
    // admitted set is identical with decay on or off; see recencyFactor.
    const now = Date.now();
    const decayed = !decayEnabled()
      ? floored
      : [...floored].sort(
          (a, b) =>
            (b.score ?? 0) * recencyFactor(tsById.get(b.id) ?? null, now) -
            (a.score ?? 0) * recencyFactor(tsById.get(a.id) ?? null, now),
        );
    // Diversity re-rank (EI-10230, MMR) — opt-in per call via `diversify`,
    // applied LAST so it only reorders the already-admitted, already-decayed
    // set (never changes which entries qualify). Lexical (trigram-Jaccard)
    // similarity fallback — see diversity-rerank.ts for the rationale and the
    // preferred real-vector path this can be upgraded to later.
    if (!opts.diversify || diversityDisabledByEnv()) return decayed;
    return diversityRerank(decayed, {
      lambda: opts.diversify.lambda,
      similarity: createTextSimilarity(),
    });
  }

  /**
   * EMBED-FREE lexical fallback (WI-4214, the `MemoryBackend.searchLexical`
   * capability): same per-scope fan-out + merge semantics as search(), but
   * served by a plain token-ILIKE query over `memory_canonical` — no
   * embedder anywhere on the path, so it works while the semantic leg is
   * saturated or down. Scores are token-overlap fractions (0..1, ordering
   * only — NOT cosine), so the relevance floor and decay deliberately do
   * not apply.
   */
  async searchLexical(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    const limit = opts.limit ?? DEFAULT_SEARCH_LIMIT;
    const scopes = scopesOf(opts.scope);
    // Do not collect one result array per scope and flatten it after a
    // Promise.all.  The default memory:search scope is user + every harness;
    // retaining all of those arrays was the strongest remaining retaining
    // producer in the 2026-08-30 OOM investigation.  Each worker maps and
    // merges its rows immediately, allowing the raw JSONB payload array to be
    // released before the next scope is admitted.
    const merged = new Map<string, { entry: MemoryEntry; scopeIndex: number; rowIndex: number }>();
    await forEachWithConcurrency(scopes, LEXICAL_SCOPE_CONCURRENCY, async (scope, scopeIndex) => {
      await withLexicalScopeAdmission(async () => {
        const rows = await this.lexicalSearch(query, limit, { user_id: scope, ...temporalFilters(opts) });
        for (const [rowIndex, row] of rows.entries()) {
          const entry = canonicalRowToEntry(row, scope);
          const prior = merged.get(entry.id);
          if (!prior || (entry.score ?? 0) > (prior.entry.score ?? 0)) {
            // `mergeById` historically retained the first scope's insertion
            // order for score ties. Keep that deterministic order even though
            // workers now complete out of order.
            merged.set(entry.id, {
              entry,
              scopeIndex: prior?.scopeIndex ?? scopeIndex,
              rowIndex: prior?.rowIndex ?? rowIndex,
            });
          }
        }
      });
    });
    return [...merged.values()]
      .sort(
        (a, b) =>
          (b.entry.score ?? 0) - (a.entry.score ?? 0) ||
          a.scopeIndex - b.scopeIndex ||
          a.rowIndex - b.rowIndex,
      )
      .map(({ entry }) => entry);
  }

  async list(opts: ListOptions): Promise<MemoryEntry[]> {
    const client = await this.client();
    const pulls = scopesOf(opts.scope).map(async (scope) => {
      const r = await client.getAll({
        filters: { user_id: scope, ...temporalFilters(opts) },
        topK: LIST_TOP_K,
      });
      return (r.results ?? []).map((row) => toEntry(row, scope));
    });
    let merged = mergeById((await Promise.all(pulls)).flat());
    if (opts.kind !== undefined) merged = merged.filter((e) => e.kind === opts.kind);
    return merged;
  }

  async get(id: string): Promise<MemoryEntry | null> {
    const client = await this.client();
    const row = await client.get(id);
    if (!row) return null;
    // mem0's get() doesn't reliably echo the pool; surface user_id when
    // present, else leave the scope opaque-unknown.
    const scope = typeof row.user_id === 'string' ? row.user_id : '';
    return toEntry(row, scope);
  }

  async forget(id: string): Promise<void> {
    const client = await this.client();
    await client.delete(id);
  }

  async update(id: string, patch: UpdatePatch): Promise<void> {
    // Metadata merge-patch (scope/user_id, workspace_id, kind, anchors, …) — the
    // canonical store does a vec-safe `payload || patch` merge (no re-embed). mem0's
    // OSS update is text-only, so this rides our own store path, not client.update.
    if (patch.metadata !== undefined) {
      const matched = await this.updatePayload(id, patch.metadata);
      // Mirror the text path's not-found contract (mem0 throws "Memory with ID …
      // not found"); the tool layer maps it to a graceful {ok:false}.
      if (!matched) throw new Error(`Memory with ID ${id} not found`);
    }
    if (patch.text !== undefined) {
      const client = await this.client();
      await client.update(id, patch.text);
    }
  }

  /** mem0's LLM fact-extraction over a conversation window. */
  async rememberConversation(
    messages: ReadonlyArray<{ role: string; content: string }>,
    opts: RememberOptions,
  ): Promise<{ ids: string[]; storedEvents: number }> {
    const client = await this.client();
    const metadata: Record<string, unknown> = { ...(opts.metadata ?? {}) };
    if (opts.kind !== undefined) metadata.kind = opts.kind;
    const result = await client.add([...messages], { userId: opts.scope, metadata });
    return { ids: extractAddedIds(result), storedEvents: extractStoredEventCount(result) };
  }

  invalidate(): void {
    invalidateMemoryClient();
  }

  /**
   * Temporal-lite validity close (the `MemoryBackend.invalidateEntry`
   * capability): soft-forget / supersession as a vec-safe column update on
   * the canonical store — never a delete, never a re-embed. Returns false
   * when no OPEN memory row matched (unknown id or already closed).
   */
  async invalidateEntry(id: string, opts: { supersededBy?: string } = {}): Promise<boolean> {
    return this.invalidateEntryStore(id, opts);
  }
}
