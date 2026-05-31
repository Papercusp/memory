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
import { memoryHost } from './config';

const LLM_MODEL = 'claude-haiku-4-5';
// The collectionName is passed to mem0 for its internal bookkeeping
// but our CanonicalVectorStore ignores it — scope lives in payload.
const MEM0_COLLECTION_PREFIX = 'operator_memory';

let _factoryPatched = false;
let _embedderFactoryPatched = false;
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

type MemoryEntry = {
  id: string;
  memory?: string;
  metadata?: Record<string, unknown>;
  score?: number;
  [key: string]: unknown;
};

type MemoryListResult = {
  results?: MemoryEntry[];
  [key: string]: unknown;
};

type MemoryClient = {
  add(content: string | unknown[], opts: Record<string, unknown>): Promise<unknown>;
  delete(id: string): Promise<unknown>;
  get(id: string): Promise<MemoryEntry | null>;
  getAll(opts: Record<string, unknown>): Promise<MemoryListResult>;
  search(query: string, opts: Record<string, unknown>): Promise<MemoryListResult>;
  update(id: string, content: string): Promise<unknown>;
};

type Mem0Module = {
  Memory: new (config: Record<string, unknown>) => MemoryClient;
};

// 'disabled' never reaches `_clientMode` at runtime (a disabled embedder
// returns null before the client is built), but the public
// `getResolvedMode()` type keeps it so callers can branch on it — see the
// operator's memoryPreflight().
type ResolvedMode = 'openai' | 'local' | 'disabled';

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

  // LLM for fact extraction. Cascade: prefer Anthropic Haiku (cheaper)
  // if a raw API key is available; fall back to OpenAI gpt-4o-mini
  // when only OpenAI is configured. Credentials come from the host.
  // Recoverable failure — don't poison-cache.
  const creds = await memoryHost().getCredentials();
  const anthropicKey = creds.anthropic_api_key ?? process.env.ANTHROPIC_API_KEY ?? '';
  const openaiKey = creds.openai_api_key ?? process.env.OPENAI_API_KEY ?? '';

  let llmConfig: Record<string, unknown>;
  if (anthropicKey) {
    llmConfig = {
      provider: 'anthropic',
      config: { apiKey: anthropicKey, model: LLM_MODEL },
    };
  } else if (openaiKey) {
    // Slightly more expensive than Haiku 4.5 ($1.50 vs $0.80/M input)
    // but lets the user run mem0 with just one vendor configured.
    llmConfig = {
      provider: 'openai',
      config: { apiKey: openaiKey, model: 'gpt-4o-mini' },
    };
  } else {
    warnOnce('no Anthropic or OpenAI API key in operator-credentials or env (add at /settings/api-keys)');
    return null;
  }

  // Per-mode collection name kept only for mem0's internal logging /
  // bookkeeping (its DEFAULT_MEMORY_CONFIG reads it). Our
  // CanonicalVectorStore ignores it — scope lives in payload.user_id.
  const collectionName = `${MEM0_COLLECTION_PREFIX}_${resolved.mode}`;
  const vecTable = resolved.mode === 'openai' ? 'memory_vec_openai' : 'memory_vec_local';

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
          collectionName,
          vecTable,
          embeddingModelDims: resolved.dims,
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
  const embedderConfig: Record<string, unknown> = {
    provider: 'custom',
    config: { embed: resolved.embed },
  };

  // mem0 tracks add/update/delete event history in SQLite. Default was
  // `:memory:` (lost on restart). Persist under ~/.papercusp/ so the
  // event log survives so /settings/user/memory can show it.
  let historyDbPath = ':memory:';
  try {
    const os = await import('node:os');
    const path = await import('node:path');
    const fs = await import('node:fs/promises');
    const dir = path.join(os.homedir(), '.papercusp');
    await fs.mkdir(dir, { recursive: true });
    historyDbPath = path.join(dir, 'mem0-history.db');
  } catch {
    /* fall back to in-memory if we can't write */
  }

  // Feed user-feedback patterns into mem0's extraction prompt so it
  // adapts over time. Best-effort: supplied by the host's learning loop
  // (optional seam). If unavailable, we just don't pass customInstructions.
  let customInstructions: string | undefined;
  try {
    customInstructions = (await memoryHost().getLearningInstructions?.()) ?? undefined;
  } catch { /* best-effort */ }

  try {
    const mem0 = await dynamicImport<Mem0Module & {
      VectorStoreFactory: { create: (provider: string, config: Record<string, unknown>) => unknown };
      EmbedderFactory?: { create: (provider: string, config: Record<string, unknown>) => unknown };
    }>(MEM0_PACKAGE);
    patchVectorStoreFactory(mem0);
    _currentEmbedFn = resolved.embed; // feed the patched 'custom' embedder (mem0 strips config.embed)
    patchEmbedderFactory(mem0);
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
