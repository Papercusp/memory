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
 *   - mem0's OSS `update()` is text-only, so a metadata patch throws.
 *
 * The mem0-internal machinery (client TTL/poison-cache, embedder
 * cascade, canonical store, re-embed) stays in ./mem0-client and
 * friends; this adapter only translates shapes.
 */

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
  getMemoryClient,
  invalidateMemoryClient,
  type Mem0Row,
  type MemoryClient,
} from './mem0-client';

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const DEFAULT_SEARCH_LIMIT = 8;
/** mem0's getAll caps at topK; high so list isn't silently truncated. */
const LIST_TOP_K = 5000;

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
}

export class Mem0Backend implements MemoryBackend {
  readonly name = 'mem0';
  private readonly getClient: () => Promise<MemoryClient | null>;

  constructor(deps: Mem0BackendDeps = {}) {
    this.getClient = deps.getClient ?? getMemoryClient;
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
    // `verbatim` → mem0's `infer: false`: skip the LLM fact-extraction and
    // embed + insert the raw text as exactly one ADD (D-008 — bulk seeding).
    const result = await client.add(text, {
      userId: opts.scope,
      metadata,
      ...(opts.verbatim ? { infer: false } : {}),
    });
    return { ids: extractAddedIds(result), storedEvents: extractStoredEventCount(result) };
  }

  async search(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    const client = await this.client();
    const limit = opts.limit ?? DEFAULT_SEARCH_LIMIT;
    const pulls = scopesOf(opts.scope).map(async (scope) => {
      // mem0's Memory.search reads `topK` (default 20) and IGNORES a
      // `limit` key — passing only `limit` silently over-fetched and
      // broke the seam's per-scope limit contract (caught by the
      // memory-backend-benchmark P-007 run). Keep `limit` for any
      // non-mem0 MemoryClient test doubles.
      const r = await client.search(query, { filters: { user_id: scope }, topK: limit, limit });
      return (r.results ?? []).map((row) => toEntry(row, scope));
    });
    const merged = mergeById((await Promise.all(pulls)).flat());
    return merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  async list(opts: ListOptions): Promise<MemoryEntry[]> {
    const client = await this.client();
    const pulls = scopesOf(opts.scope).map(async (scope) => {
      const r = await client.getAll({ filters: { user_id: scope }, topK: LIST_TOP_K });
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
    if (patch.metadata !== undefined) {
      throw new Error('mem0 backend does not support metadata patches (text-only update)');
    }
    if (patch.text === undefined) return;
    const client = await this.client();
    await client.update(id, patch.text);
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
}
