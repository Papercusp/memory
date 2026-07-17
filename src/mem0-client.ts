/**
 * mem0 client singleton for the operator's persistent memory.
 *
 * Plan 3 (storage layer rewritten 2026-05-24 per memory-harness-scope plan).
 * Extracted to `@papercusp/memory` 2026-05-30 (P-021): the embedder
 * cascade, credentials, embedded-pg discovery, and adaptive-instruction
 * feed are now injected via the `configureMemory()` host seam — see
 * ./config. The store core itself carries no operator coupling.
 *
 * Stack:
 *   - storage: CanonicalVectorStore (our own pgvector layer) —
 *              one canonical `memory_canonical` row per fact, with
 *              per-embedder-model vec tables joined by memory_id.
 *              Switching embedder mode no longer "loses" your old
 *              memories; only the recall side picks a different vec
 *              table. Re-embedding into the other model is an INSERT
 *              into the other vec table without touching the canonical
 *              row. See ./canonical-store + migration 081-memory-canonical.sql.
 *   - LLM:     Anthropic Haiku 4.5 (cheap fact extraction; only generative
 *              vendor we use — Anthropic does not offer embeddings)
 *   - embedder: resolved by the host (`resolveEmbedder` seam) — the
 *              operator runs the openai/local/disabled cascade and hands
 *              back a pre-built embed fn + mode + dims. mem0 is always
 *              configured with a `custom` embedder over that fn, so the
 *              store never needs vendor keys itself.
 *
 * Per-user scoping: userId = session.user.id.
 * Harness-scoped: userId = `harness:<slug>`.
 * Legacy workspace-shared (deprecated): userId = `workspace:<id>`,
 *   metadata.shared. Read-through only; the write path no longer
 *   emits these (see memory:remember).
 *
 * Vector dimensions: the resolved embedder reports its own `dims` (384
 * across both shipped embedders — OpenAI's native 1536 truncated via its
 * `dimensions` parameter; BGE-small is natively 384). Same wire shape
 * lets both vec tables share migration 081's schema.
 *
 * Vector SPACES still differ across embedders, so each model writes
 * to its own vec table — `memory_vec_openai` vs `memory_vec_local`.
 * The canonical text is shared.
 *
 * Failure mode: if a fatal load error occurs (missing npm package),
 * memory tools return graceful no-ops. Logs once per process per
 * distinct reason. Recoverable failures (transient PG, missing key the
 * user can add via /settings/api-keys) DON'T poison-cache, so the next
 * call retries.
 */

import { CanonicalVectorStore } from './canonical-store';
import { memoryHost, memorySchema } from './config';
import { coalesceEmbedFn } from './embed-coalesce';
import { FallbackExtractionLlm, type ExtractionLlm } from './extraction-llm';

const LLM_MODEL = 'claude-haiku-4-5';
// The collectionName is passed to mem0 for its internal bookkeeping
// but our CanonicalVectorStore ignores it — scope lives in payload.
const MEM0_COLLECTION_PREFIX = 'operator_memory';

let _factoryPatched = false;
let _embedderFactoryPatched = false;
let _llmFactoryPatched = false;
// The host's pre-built embed fn for the CURRENT build. mem0's
// ConfigManager.mergeConfig strips function-valued fields out of
// embedder.config during Zod validation, so the patched 'custom'
// embedder (below) reads this live module var instead of config.embed.
// Set in tryLoad() before each (re)build.
let _currentEmbedFn: ((text: string) => Promise<number[]>) | null = null;
/**
 * Register CanonicalVectorStore as a `'canonical'` provider on mem0's
 * VectorStoreFactory. mem0's OSS factory uses a hard-coded switch with
 * no plugin hook, so we patch the static `create` method in place at
 * first construction. Idempotent — subsequent calls no-op.
 */
function patchVectorStoreFactory(mem0Module: {
  VectorStoreFactory: { create: (provider: string, config: Record<string, unknown>) => unknown };
}): void {
  if (_factoryPatched) return;
  const Factory = mem0Module.VectorStoreFactory;
  if (!Factory || typeof Factory.create !== 'function') {
    warnOnce('mem0 VectorStoreFactory not patchable (interface changed?); canonical store unavailable');
    return;
  }
  const orig = Factory.create.bind(Factory);
  Factory.create = (provider: string, config: Record<string, unknown>) => {
    if (provider === 'canonical') {
      // mem0 passes the inner config object directly (not wrapped in
      // { provider, config }) — see Memory constructor invoking
      // VectorStoreFactory.create(this.config.vectorStore.provider,
      // this.config.vectorStore.config).
      const store = new CanonicalVectorStore(
        config as unknown as ConstructorParameters<typeof CanonicalVectorStore>[0],
      );
      // Track it so its pg.Client is closed when the mem0 client is rebuilt
      // (TTL / invalidate) — otherwise each rebuild orphans a connection.
      _liveCanonicalStores.add(store);
      return store;
    }
    return orig(provider, config);
  };
  _factoryPatched = true;
}

/**
 * CanonicalVectorStore instances created via the patched factory. Each holds a
 * cached `pg.Client`; they must be disposed when the mem0 client is dropped so
 * the connection doesn't leak across the hourly TTL rebuild / invalidate.
 */
const _liveCanonicalStores = new Set<CanonicalVectorStore>();

/** Close + forget every tracked canonical store's PG client. Fire-and-forget
 *  safe — `dispose()` is idempotent and tolerant of an in-flight client. */
async function disposeLiveCanonicalStores(): Promise<void> {
  const stores = [..._liveCanonicalStores];
  _liveCanonicalStores.clear();
  await Promise.allSettled(stores.map((s) => s.dispose()));
}

/**
 * Merge-patch a memory row's METADATA in the canonical store (the store half of
 * `Mem0Backend.update({ metadata })` — mem0's OSS update is text-only). Ensures the
 * mem0 client is built (so a `CanonicalVectorStore` exists), then runs the vec-safe
 * `payload || patch` merge via any live store — they all point at the one shared
 * `memory_canonical` table, and `updatePayload` self-guards to memory (non-entity)
 * rows. Returns whether a row matched (false ⇒ unknown id, surfaced as not-found).
 * Throws if the store is unavailable (the backend's `available()` gates this first).
 */
export async function updateMemoryPayload(id: string, patch: Record<string, unknown>): Promise<boolean> {
  await getMemoryClient(); // build (or reuse) the client so the canonical store is registered
  const store = [..._liveCanonicalStores][0];
  if (!store) throw new Error('mem0_unavailable');
  return store.updatePayload(id, patch);
}

/**
 * EMBED-FREE lexical search over the canonical store (the store half of
 * `Mem0Backend.searchLexical` — WI-4214's degraded-path fallback). Same
 * live-store access pattern as `updateMemoryPayload`: any live
 * `CanonicalVectorStore` works (they all point at the one shared
 * `memory_canonical` table; which vec table it owns is irrelevant — this
 * path never joins one). NOTE the availability asymmetry this fallback
 * relies on: in the saturated-embedder case the mem0 client is already
 * built and cached, so `getMemoryClient()` returns instantly and NO embed
 * happens anywhere on this path; only a cold client build can fail here,
 * and the caller surfaces that as fallback-unavailable.
 */
export async function lexicalSearchCanonical(
  query: string,
  topK: number,
  filters: Record<string, string>,
): Promise<Array<{ id: string; payload: Record<string, unknown>; score?: number }>> {
  await getMemoryClient(); // reuse the cached client so a canonical store exists
  const store = [..._liveCanonicalStores][0];
  if (!store) throw new Error('mem0_unavailable');
  return store.lexicalSearch(query, topK, filters);
}

/**
 * Vector search over the canonical store with a PRECOMPUTED query vector —
 * the store half of `Mem0Backend`'s batched multi-scope search (EI-12962).
 * Same live-store access pattern as `lexicalSearchCanonical`. The caller
 * embeds ONCE (`embedForCurrentClient`) and fans this out per scope, so an
 * N-scope recall costs one embed + N ~10ms pgvector queries instead of N
 * full embed+search round-trips — the 20-scope operator-chat injection was
 * reliably blowing its 5s deadline on per-scope embeds (a flat 5s/turn tax
 * that injected NOTHING).
 */
export async function vectorSearchCanonical(
  vector: number[],
  topK: number,
  filters: Record<string, string>,
): Promise<Array<{ id: string; payload: Record<string, unknown>; score?: number }>> {
  await getMemoryClient(); // reuse the cached client so a canonical store exists
  const store = [..._liveCanonicalStores][0];
  if (!store) throw new Error('mem0_unavailable');
  return store.search(vector, topK, filters);
}

/**
 * Embed `text` with the EXACT embed fn the current mem0 client was built
 * against (`_currentEmbedFn`), so a precomputed query vector lives in the
 * same vec space as the store `vectorSearchCanonical` reads. Returns null
 * when no client/embedder is up OR the embed fails — callers must fall back
 * to the legacy per-scope `client.search` path (correctness over speed).
 */
export async function embedForCurrentClient(text: string): Promise<number[] | null> {
  const client = await getMemoryClient().catch(() => null);
  if (!client || !_currentEmbedFn) return null;
  try {
    return await _currentEmbedFn(text);
  } catch {
    return null;
  }
}

/**
 * Close a memory row's validity window in the canonical store (the store half
 * of `Mem0Backend.invalidateEntry` — temporal-lite soft-forget/supersession).
 * Same live-store access pattern as `updateMemoryPayload`: any live
 * `CanonicalVectorStore` works (one shared `memory_canonical` table), and
 * `invalidate` self-guards to open memory (non-entity) rows. Returns whether
 * an OPEN row was closed (false ⇒ unknown id or already closed — first-wins).
 */
export async function invalidateEntryCanonical(
  id: string,
  opts: { supersededBy?: string } = {},
): Promise<boolean> {
  await getMemoryClient(); // reuse the cached client so a canonical store exists
  const store = [..._liveCanonicalStores][0];
  if (!store) throw new Error('mem0_unavailable');
  return store.invalidate(id, opts);
}

/**
 * mem0ai 3.x has NO `custom` embedder provider — its EmbedderFactory
 * switch only knows openai/ollama/lmstudio/google/azure_openai/langchain
 * and throws "Unsupported embedder provider: custom" otherwise. It
 * exposes no plugin hook, but it DOES export the EmbedderFactory class,
 * so (exactly like VectorStoreFactory) we patch its static `create` in
 * place. Without this the Memory constructor throws on every build and
 * the whole memory subsystem is dead — every memory:* tool returns
 * mem0_unavailable.
 *
 * The 'custom' embedder reads `_currentEmbedFn`, NOT `config.embed`:
 * mem0's mergeConfig drops function-valued config fields during Zod
 * validation, so by the time create() runs the fn is already gone.
 */
// Exported for the regression test (mem0-client.test.ts) — it can't drive
// getMemoryClient() under vitest because the `new Function('return import')`
// dynamicImport trick has no import callback in vitest's module runner, so it
// patches + asserts EmbedderFactory directly. Not re-exported from index.ts.
export function patchEmbedderFactory(mem0Module: {
  EmbedderFactory?: { create: (provider: string, config: Record<string, unknown>) => unknown };
}): void {
  if (_embedderFactoryPatched) return;
  const Factory = mem0Module.EmbedderFactory;
  if (!Factory || typeof Factory.create !== 'function') {
    warnOnce('mem0 EmbedderFactory not patchable (interface changed?); custom embedder unavailable');
    return;
  }
  const orig = Factory.create.bind(Factory);
  Factory.create = (provider: string, config: Record<string, unknown>) => {
    if (provider === 'custom') {
      return {
        embed: (text: string) => _currentEmbedFn!(text),
        embedBatch: (texts: string[]) => Promise.all(texts.map((t) => _currentEmbedFn!(t))),
      };
    }
    return orig(provider, config);
  };
  _embedderFactoryPatched = true;
}

/**
 * Test seam (mem0-client.test.ts) — set the embed fn the patched 'custom'
 * embedder routes to. In production `tryLoad()` sets this from the host's
 * `resolveEmbedder()` before each (re)build. Not re-exported from index.ts.
 */
export function _setCurrentEmbedFnForTest(
  fn: ((text: string) => Promise<number[]>) | null,
): void {
  _currentEmbedFn = fn;
}

/**
 * The live extraction LLM the patched 'custom' LLM provider routes to —
 * same module-var pattern as `_currentEmbedFn` (mem0's mergeConfig Zod
 * validation strips non-scalar fields out of llm.config, so the instance
 * can't ride the config object). Set in tryLoad() before each (re)build
 * when the host supplies a session-backed extractor.
 */
let _currentExtractionLlm: ExtractionLlm | null = null;

/**
 * mem0ai 3.x has NO `custom` LLM provider — `LLMFactory.create` is a
 * hard-coded switch (openai/anthropic/groq/…) that throws "Unsupported
 * LLM provider: custom" otherwise. Exactly like VectorStoreFactory and
 * EmbedderFactory, we patch its static `create` in place: provider
 * `'custom'` yields a delegator reading `_currentExtractionLlm` live.
 * Exported for the conformance test (extraction-llm.test.ts) — pinned
 * there against the REAL mem0ai module so an upstream interface change
 * fails a test, not production. Not re-exported from index.ts.
 */
export function patchLlmFactory(mem0Module: {
  LLMFactory?: { create: (provider: string, config: Record<string, unknown>) => unknown };
}): void {
  if (_llmFactoryPatched) return;
  const Factory = mem0Module.LLMFactory;
  if (!Factory || typeof Factory.create !== 'function') {
    warnOnce('mem0 LLMFactory not patchable (interface changed?); session extraction unavailable');
    return;
  }
  const orig = Factory.create.bind(Factory);
  Factory.create = (provider: string, config: Record<string, unknown>) => {
    if (provider === 'custom') {
      const llm: ExtractionLlm = {
        generateResponse: (messages, responseFormat, tools) =>
          _currentExtractionLlm!.generateResponse(messages, responseFormat, tools),
        generateChat: (messages) => _currentExtractionLlm!.generateChat(messages),
      };
      return llm;
    }
    return orig(provider, config);
  };
  _llmFactoryPatched = true;
}

/**
 * Test seam (extraction-llm.test.ts) — set the LLM the patched 'custom'
 * provider routes to. Not re-exported from index.ts.
 */
export function _setCurrentExtractionLlmForTest(llm: ExtractionLlm | null): void {
  _currentExtractionLlm = llm;
}

// Exported for Mem0Backend (./mem0-backend.ts), which adapts this raw
// mem0 surface onto the neutral MemoryBackend interface (./backend.ts).
export type Mem0Row = {
  id: string;
  memory?: string;
  metadata?: Record<string, unknown>;
  score?: number;
  [key: string]: unknown;
};

export type Mem0ListResult = {
  results?: Mem0Row[];
  [key: string]: unknown;
};

export type MemoryClient = {
  add(content: string | unknown[], opts: Record<string, unknown>): Promise<unknown>;
  delete(id: string): Promise<unknown>;
  get(id: string): Promise<Mem0Row | null>;
  getAll(opts: Record<string, unknown>): Promise<Mem0ListResult>;
  search(query: string, opts: Record<string, unknown>): Promise<Mem0ListResult>;
  update(id: string, content: string): Promise<unknown>;
};

type Mem0Module = {
  Memory: new (config: Record<string, unknown>) => MemoryClient;
};

// 'disabled' never reaches `_clientMode` at runtime (a disabled embedder
// returns null before the client is built), but the public
// `getResolvedMode()` type keeps it so callers can branch on it — see the
// operator's memoryPreflight().
type ResolvedMode = 'openai' | 'local' | 'gemma' | 'harrier' | 'disabled';

const dynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as <T>(specifier: string) => Promise<T>;

const MEM0_PACKAGE = 'mem0ai/oss';

let _client: MemoryClient | null = null;
let _clientMode: ResolvedMode | null = null;
let _clientBuiltAt = 0;
/**
 * mem0's customInstructions is set at construction time and the
 * Memory class doesn't expose a mutator. We refresh by tearing the
 * client down after this TTL so the next call rebuilds with current
 * learning-loop instructions. Cheap — no network, just JS object
 * construction.
 */
const _clientTtlMs = 60 * 60 * 1000; // 1 hour
/**
 * "Permanent" failure cache — only set when the failure is from a
 * truly fixed cause (mem0ai npm package missing). For recoverable
 * failures (missing API key, transient PG hiccup), we do NOT cache
 * so the next call retries.
 */
let _clientPermanentFailure = false;
const _seenWarnReasons = new Set<string>();

function warnOnce(reason: string): void {
  if (_seenWarnReasons.has(reason)) return;
  _seenWarnReasons.add(reason);
  console.warn(`[mem0] ${reason}.`);
}

/**
 * Per-key result cache for the Anthropic-key probe. Keyed by the key
 * STRING, so a rotation (new string) re-probes; an invalid key is never
 * re-probed in this process. Bounded by the handful of keys a box sees.
 */
const _anthropicKeyProbeCache = new Map<string, boolean>();

/**
 * Cheap auth probe for an Anthropic key (the models endpoint — no
 * tokens billed). Returns false ONLY on an explicit auth rejection
 * (401/403); transient/network failures resolve true so an offline box
 * never loses its extractor over a blip.
 */
async function anthropicKeyUsable(
  key: string,
  fetchImpl: typeof fetch = fetch,
): Promise<boolean> {
  const cached = _anthropicKeyProbeCache.get(key);
  if (cached !== undefined) return cached;
  let usable = true;
  try {
    const r = await fetchImpl('https://api.anthropic.com/v1/models?limit=1', {
      headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01' },
    });
    usable = !(r.status === 401 || r.status === 403);
  } catch {
    usable = true; // network blip — assume usable, don't downgrade
  }
  _anthropicKeyProbeCache.set(key, usable);
  return usable;
}

/** Test hook — clear the per-key probe cache. */
export function _resetAnthropicKeyProbeCacheForTest(): void {
  _anthropicKeyProbeCache.clear();
}

/**
 * Resolve mem0's fact-extraction LLM config from the available keys.
 * A PRESENT-but-rejected Anthropic key no longer kills extraction
 * (memory-backend-benchmark D-007: a stale credential made every add()
 * 401 inside mem0, silently swallowed into `{results: []}` — the whole
 * store looked dead). The cascade now VALIDATES the Anthropic key at
 * client build (cached per key string) and falls back:
 *
 *   claude-session adapter (host-supplied via getExtractionLlm —
 *     probe-validated host-side; mem0-extraction-via-claude-session D-002)
 *   → anthropic (key present + not auth-rejected)
 *   → openai gpt-4o-mini (key present)
 *   → anthropic anyway (dead key, nothing else — extraction will fail,
 *     but search + verbatim writes still work; warn loud)
 *   → null (no keys at all)
 *
 * The session rung returns `{provider:'custom'}` — tryLoad() then routes
 * the patched LLMFactory's 'custom' provider to the session adapter
 * wrapped in a FallbackExtractionLlm over the key rungs (D-004: an auth
 * failure demotes loudly mid-process, never silently no-ops).
 *
 * Exported for tests (probe injectable); not re-exported from index.ts.
 */
export async function resolveExtractionLlmConfig(
  keys: { anthropicKey: string; openaiKey: string },
  probe: (key: string) => Promise<boolean> = anthropicKeyUsable,
  sessionLlm?: ExtractionLlm | null,
): Promise<Record<string, unknown> | null> {
  if (sessionLlm) {
    return { provider: 'custom', config: {} };
  }
  const { anthropicKey, openaiKey } = keys;
  const anthropicOk = anthropicKey ? await probe(anthropicKey) : false;
  if (anthropicKey && anthropicOk) {
    return { provider: 'anthropic', config: { apiKey: anthropicKey, model: LLM_MODEL } };
  }
  if (openaiKey) {
    if (anthropicKey) {
      // Slightly more expensive than Haiku 4.5 ($1.50 vs $0.80/M input),
      // but a working extractor beats a silently-dead one.
      warnOnce(
        'anthropic key rejected (401/403) — falling back to OpenAI gpt-4o-mini for fact extraction; rotate the key at /settings/api-keys',
      );
    }
    return { provider: 'openai', config: { apiKey: openaiKey, model: 'gpt-4o-mini' } };
  }
  if (anthropicKey) {
    warnOnce(
      'anthropic key rejected (401/403) and no OpenAI key — mem0 fact extraction WILL fail until the key is rotated at /settings/api-keys (search + verbatim writes still work)',
    );
    return { provider: 'anthropic', config: { apiKey: anthropicKey, model: LLM_MODEL } };
  }
  return null;
}

/**
 * Try to load the mem0 client. Returns null if dependencies aren't
 * installed, mode is 'disabled', or config can't be assembled.
 */
async function tryLoad(): Promise<MemoryClient | null> {
  if (_client && Date.now() - _clientBuiltAt < _clientTtlMs) return _client;
  if (_client) {
    // TTL expired — drop and rebuild so learning-loop instructions
    // refresh. mem0 doesn't ship a mutator for customInstructions.
    _client = null;
    // Close the previous store's PG client (the new one is built below and
    // re-tracked). Fire-and-forget: the set is snapshot+cleared synchronously.
    void disposeLiveCanonicalStores();
  }
  if (_clientPermanentFailure) return null;

  // Resolve the embedder via the host (openai/local/disabled cascade).
  // 'disabled' is a hard stop — don't cache, the user can add a key or
  // flip the preference any time.
  const resolved = await memoryHost().resolveEmbedder();
  if (resolved.mode === 'disabled') {
    if (resolved.reason && resolved.reason !== 'user_disabled') {
      warnOnce(`embedder unavailable: ${resolved.reason} (set memoryEmbedderMode in /settings/user)`);
    }
    return null;
  }
  // WI-5094: coalesce + short-TTL-memoize same-text embeds. The pre-turn
  // memory injection issues three scope pulls with the SAME query text per
  // turn; on an in-process embedder that was 3 serialized embeds (~2.6-4.5s
  // of searchMs — the dominant prompt-build phase). Wrapped HERE, once per
  // (re)build, so a rebuilt/flipped embedder always starts a fresh cache.
  // (`resolved` is narrowed to the non-disabled variant by the guard above,
  // so `.embed` is well-typed here — TS2339 fixed 2026-07-17, WI-5217.)
  const coalescedEmbed = coalesceEmbedFn(resolved.embed);

  // Resolve PG connection as discrete fields — mem0's PGVector provider
  // expects user/password/host/port/dbname, NOT a connectionString.
  // Transient failures retry next call.
  let pgFields: { host: string; port: number; user: string; password: string; dbname: string };
  try {
    const { pgFields: getFields } = await import('./mem0-connection');
    pgFields = await getFields();
  } catch (e) {
    warnOnce(`couldn't resolve PG connection: ${(e as Error).message}`);
    return null;
  }

  // LLM for fact extraction. Cascade: the host's session-backed
  // extractor first (Claude-session anthropic-direct, $0 marginal, no
  // credential to rot — mem0-extraction-via-claude-session D-001/D-002),
  // then Anthropic Haiku on a raw API key, then OpenAI gpt-4o-mini.
  // Credentials come from the host. Recoverable failure — don't
  // poison-cache.
  const creds = await memoryHost().getCredentials();
  const anthropicKey = creds.anthropic_api_key ?? process.env.ANTHROPIC_API_KEY ?? '';
  const openaiKey = creds.openai_api_key ?? process.env.OPENAI_API_KEY ?? '';

  let sessionLlm: ExtractionLlm | null = null;
  try {
    sessionLlm = (await memoryHost().getExtractionLlm?.()) ?? null;
  } catch (e) {
    warnOnce(`host getExtractionLlm threw (${(e as Error).message}); using key rungs`);
  }

  const llmConfig = await resolveExtractionLlmConfig({ anthropicKey, openaiKey }, undefined, sessionLlm);
  if (!llmConfig) {
    warnOnce('no Claude session and no Anthropic or OpenAI API key in operator-credentials or env (add at /settings/api-keys)');
    return null;
  }

  // Per-mode collection name kept only for mem0's internal logging /
  // bookkeeping (its DEFAULT_MEMORY_CONFIG reads it). Our
  // CanonicalVectorStore ignores it — scope lives in payload.user_id.
  const collectionName = `${MEM0_COLLECTION_PREFIX}_${resolved.mode}`;
  // Per-embedder-model vec table = the embedding SPACE (embedding-space-vs-dimension
  // / EI-8913). Each mode writes to its own table so vectors from different models
  // never mix. 'gemma' = EmbeddingGemma-300m @ MRL-384 (migration 534);
  // 'harrier' = harrier-oss-0.6b @ native-1024 (migration 547).
  const vecTable =
    resolved.mode === 'openai'
      ? 'memory_vec_openai'
      : resolved.mode === 'gemma'
        ? 'memory_vec_gemma'
        : resolved.mode === 'harrier'
          ? 'memory_vec_harrier'
          : 'memory_vec_local';

  // Vector-store selection. Prefer the canonical store (migration 081)
  // when pgvector is present; fall back to mem0's in-process `memory`
  // provider when it isn't. The in-process store is volatile (lost on
  // restart) but lets memory:* work end-to-end during local-dev /
  // pre-extension setups.
  let vectorStoreConfig: Record<string, unknown>;
  try {
    // Probe: does PG have the vector extension available?
    // Dynamic import, NOT require(): @papercusp/memory runs as ESM (it's
    // not in Next's transpilePackages), where `require` is undefined.
    // require('pg') here threw "require is not defined", silently failing
    // the probe and downgrading the store to mem0's volatile in-process
    // provider.
    const pgMod = (await import('pg')) as typeof import('pg') & { default?: typeof import('pg') };
    const Client = pgMod.Client ?? pgMod.default?.Client;
    if (!Client) throw new Error('pg.Client not resolvable');
    const probe = new Client({ host: pgFields.host, port: pgFields.port, user: pgFields.user, password: pgFields.password, database: pgFields.dbname });
    await probe.connect();
    const r = await probe.query("SELECT 1 FROM pg_available_extensions WHERE name='vector'");
    await probe.end();
    if (r.rowCount && r.rowCount > 0) {
      vectorStoreConfig = {
        provider: 'canonical',
        config: {
          host: pgFields.host,
          port: pgFields.port,
          user: pgFields.user,
          password: pgFields.password,
          dbname: pgFields.dbname,
          schema: memorySchema(),
          collectionName,
          vecTable,
          embeddingModelDims: resolved.dims,
          // mem0 `_autoInitialize` reads `vectorStore.config.dimension` to skip its
          // live-embed detection probe (see embedderConfig note) — EI-4027.
          dimension: resolved.dims,
        },
      };
    } else {
      warnOnce('pgvector extension not available in embedded PG — falling back to in-process memory store (volatile)');
      vectorStoreConfig = {
        provider: 'memory',
        config: { collectionName, embeddingModelDims: resolved.dims },
      };
    }
  } catch (e) {
    warnOnce(`pgvector probe failed: ${(e as Error).message}; falling back to in-process memory store`);
    vectorStoreConfig = {
      provider: 'memory',
      config: { collectionName, embeddingModelDims: resolved.dims },
    };
  }

  // mem0 is always driven through a `custom` embedder over the host's
  // pre-built embed fn — both openai and local resolve to a
  // `(text) => number[]` on the operator side, so the store stays
  // vendor-agnostic and never holds an API key itself.
  // Pass the embedding dimension EXPLICITLY so mem0's `_autoInitialize` skips its
  // dimension-detection probe — a LIVE embed() call. Without it, a failing embed
  // (e.g. the OpenAI embedder hitting 429 `insufficient_quota`) makes
  // `_autoInitialize` THROW on every client build; in the bg-host that uncaught
  // throw recurs each tick and wedges the event loop (EI-4027: git-sync +
  // green-checkpoint + substrate routines all stall). `resolved.dims` is 384 for
  // both shipped embedders. mem0 reads `embedder.config.embeddingDims` (and
  // `vectorStore.config.dimension` below) to bypass the probe.
  const embedderConfig: Record<string, unknown> = {
    provider: 'custom',
    config: { embed: coalescedEmbed, embeddingDims: resolved.dims },
  };

  // mem0 tracks add/update/delete event history in SQLite. Default is
  // `:memory:` (lost on restart). When the host provides a `localStoreDir`
  // (default the OS tmpdir; the operator passes ~/.papercusp), persist
  // there so the event log survives restarts. `localStoreDir: null`
  // forces the in-memory history.
  let historyDbPath = ':memory:';
  const localStoreDir = memoryHost().localStoreDir;
  if (localStoreDir !== null) {
    try {
      const os = await import('node:os');
      const path = await import('node:path');
      const fs = await import('node:fs/promises');
      const dir = localStoreDir ?? os.tmpdir();
      await fs.mkdir(dir, { recursive: true });
      historyDbPath = path.join(dir, 'mem0-history.db');
    } catch {
      /* fall back to in-memory if we can't write */
    }
  }

  // Feed user-feedback patterns into mem0's extraction prompt so it
  // adapts over time. Best-effort: supplied by the host's learning loop
  // (optional seam). If unavailable, we just don't pass customInstructions.
  let customInstructions: string | undefined;
  try {
    customInstructions = (await memoryHost().getLearningInstructions?.()) ?? undefined;
  } catch { /* best-effort */ }

  try {
    // EI-10183: mem0's local entity extractor captures `nlp = __require("compromise")`
    // at MODULE-EVAL time. We import the ESM build (`mem0ai/oss` → index.mjs), where
    // esbuild's `__require` shim throws when no global `require` exists — so `nlp` is
    // undefined and mem0 falls back to a greedy regex noun-chunker that emitted
    // 100%-junk COMPOUND entities in prod (lowercase sentence fragments like
    // "so the re", "just before end of", "liner in the folder"; 5906/7071 live
    // entity rows). Provide a real `require` for the FIRST mem0 import so
    // `__require("compromise")` resolves and the clean noun-phrase path
    // (`extractCompoundsWithNlp` / doc.nouns()) runs instead. mem0 binds its
    // `__require` const to this reference at eval, so we restore the global right
    // after the import — blast radius stays inside mem0's own CJS interop.
    // Kill-switch: PAPERCUSP_MEMORY_ENTITY_NLP=off. (compromise ships as a mem0 dep.)
    type Mem0Loaded = Mem0Module & {
      VectorStoreFactory: { create: (provider: string, config: Record<string, unknown>) => unknown };
      EmbedderFactory?: { create: (provider: string, config: Record<string, unknown>) => unknown };
      LLMFactory?: { create: (provider: string, config: Record<string, unknown>) => unknown };
    };
    const nlpFix = process.env.PAPERCUSP_MEMORY_ENTITY_NLP !== 'off';
    const gt = globalThis as unknown as { require?: unknown };
    const hadGlobalRequire = 'require' in globalThis;
    let installedRequire = false;
    if (nlpFix && !hadGlobalRequire) {
      try {
        const { createRequire } = await import('node:module');
        gt.require = createRequire(import.meta.url);
        installedRequire = true;
      } catch {
        /* best-effort — mem0 stays on its regex fallback if this fails */
      }
    }
    let mem0: Mem0Loaded;
    try {
      mem0 = await dynamicImport<Mem0Loaded>(MEM0_PACKAGE);
    } finally {
      if (installedRequire) {
        try {
          delete gt.require;
        } catch {
          gt.require = undefined;
        }
      }
    }
    patchVectorStoreFactory(mem0);
    _currentEmbedFn = coalescedEmbed; // feed the patched 'custom' embedder (mem0 strips config.embed)
    patchEmbedderFactory(mem0);
    if (sessionLlm) {
      // Session rung live: route the 'custom' LLM provider to the host's
      // adapter wrapped in the key-rung fallback cascade. The fallback is
      // built LAZILY (first demotion) via mem0's own factory over the key
      // config the rungs below the session would have resolved — so a
      // mid-process auth death lands writes on the next rung within the
      // same call (D-004), never a silent no-op.
      patchLlmFactory(mem0);
      _currentExtractionLlm = new FallbackExtractionLlm(
        sessionLlm,
        async () => {
          const cfg = await resolveExtractionLlmConfig({ anthropicKey, openaiKey });
          if (!cfg || !mem0.LLMFactory) return null;
          return mem0.LLMFactory.create(
            cfg.provider as string,
            cfg.config as Record<string, unknown>,
          ) as ExtractionLlm;
        },
        { warn: warnOnce, primaryLabel: 'claude-session extraction' },
      );
    }
    const { Memory } = mem0;
    _client = new Memory({
      embedder: embedderConfig,
      vectorStore: vectorStoreConfig,
      llm: llmConfig,
      historyDbPath,
      ...(customInstructions ? { customInstructions } : {}),
    });
    _clientMode = resolved.mode;
    _clientBuiltAt = Date.now();
    // Eagerly WARM the lazy mem0 init + embedder OFF the hot path. mem0's
    // `_ensureInitialized` fires on the FIRST op (search/add); under box load that
    // cold-init can exceed the memory-tool deadline, so the first real search/remember
    // after each (hourly TTL) rebuild times out — and a burst of concurrent ops during
    // that window ALL time out (the "mem0 flaky all session" reports). A fire-and-forget
    // warm-up pays that init here, in the background, so user-facing ops hit a warm
    // client. Best-effort: a warm-up failure never affects the returned client (the next
    // real op re-attempts + surfaces any error). The dummy scope returns nothing + writes
    // nothing — it only triggers init + warms the embedder worker.
    const clientToWarm = _client;
    void (async () => {
      try {
        await clientToWarm.search('warmup', { filters: { user_id: '__mem0_warmup__' }, topK: 1, limit: 1 });
      } catch {
        /* best-effort warm-up — the next real op re-attempts and surfaces errors */
      }
    })();
    return _client;
  } catch (e) {
    const msg = (e as Error).message ?? String(e);
    if (/cannot find module|MODULE_NOT_FOUND|Cannot resolve/i.test(msg)) {
      _clientPermanentFailure = true;
      warnOnce(`${msg}. Run 'npm install --legacy-peer-deps' in apps/operator to enable.`);
    } else {
      // Unknown failure — retry next call.
      warnOnce(msg);
    }
    return null;
  }
}

/**
 * Returns the mem0 client OR null if dependencies aren't installed,
 * the user disabled memory, or PG isn't reachable. Callers handle the
 * null case as a no-op.
 */
export async function getMemoryClient(): Promise<MemoryClient | null> {
  return tryLoad();
}

/**
 * Inspection helper for the settings UI / status displays.
 */
export function getResolvedMode(): ResolvedMode | null {
  return _clientMode;
}

/**
 * Force the cached client to be rebuilt on next access. Called by the
 * feedback path after high-impact mutations (forget_all) so the
 * learning-loop instructions apply immediately rather than waiting
 * for the 1-hour TTL.
 */
export function invalidateMemoryClient(): void {
  _client = null;
  _clientBuiltAt = 0;
  // Release the discarded store's PG client so invalidation doesn't leak a
  // connection (e.g. on credential change / mode switch).
  void disposeLiveCanonicalStores();
}

/**
 * Awaitable invalidate: like `invalidateMemoryClient`, but resolves only
 * after every tracked canonical store's PG client has actually closed.
 * Callers that immediately do something the open connections would block
 * (the bench's `DROP SCHEMA … CASCADE` — fire-and-forget disposal races
 * the drop into a 55P03 lock timeout) await this instead.
 */
export async function disposeMemoryClient(): Promise<void> {
  _client = null;
  _clientBuiltAt = 0;
  await disposeLiveCanonicalStores();
}
