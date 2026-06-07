/**
 * Host seam for `@papercusp/memory`.
 *
 * The store core carries no operator coupling. Everything operator-
 * specific — where the embedded-pg lives, which LLM/embedder credentials
 * are configured, how the embedder cascade resolves, and the optional
 * adaptive-instruction feed — is injected once via `configureMemory()`.
 *
 * mem0 owns its own `pg.Client`, so (unlike backup/search) we do NOT
 * inject a postgres-js `sql` handle. The PG seam is just an admin-URL
 * resolver; mem0-connection parses it into the discrete fields mem0's
 * PGVector provider wants.
 *
 * Part of papercusp-systems-abstraction-2026-05-29 (P-021).
 */

/** An embedder: text in, a fixed-dimension vector out. */
export type EmbedFn = (text: string) => Promise<number[]>;

/**
 * The embedder resolved for the *current* user preference. The host runs
 * the openai → local → disabled cascade and hands back a pre-built embed
 * function plus the metadata the store needs (the `mode` drives the
 * per-model vec table; `dims` sizes the canonical column).
 */
export type ResolvedEmbedder =
  | { mode: 'openai' | 'local'; dims: number; embed: EmbedFn }
  | { mode: 'disabled'; reason?: string };

/** LLM credentials for mem0's fact-extraction step. */
export interface MemoryCredentials {
  openai_api_key?: string;
  anthropic_api_key?: string;
}

export interface MemoryHost {
  /**
   * Resolve the harness-admin Postgres URL the store should connect to.
   * Replaces the operator's `@/lib/embedded-pg-discovery`. May be sync
   * or async; mem0-connection awaits it.
   */
  getAdminUrl: () => string | Promise<string>;

  /**
   * Read the LLM credentials for mem0's fact-extraction cascade
   * (Anthropic Haiku → OpenAI gpt-4o-mini). Embedder keys are NOT read
   * here — they live behind `resolveEmbedder` / `buildEmbedderForMode`.
   */
  getCredentials: () => Promise<MemoryCredentials>;

  /**
   * Resolve the embedder for the current user preference (the
   * openai/local/disabled cascade). Called on every client (re)build.
   */
  resolveEmbedder: () => Promise<ResolvedEmbedder>;

  /**
   * Build an embedder for an *explicit* mode — used by the re-embed pass,
   * which must embed under the target model's space regardless of the
   * current preference. Throws if that mode's credentials/packages aren't
   * available.
   */
  buildEmbedderForMode: (mode: 'openai' | 'local') => Promise<EmbedFn>;

  /**
   * Optional: adaptive extraction instructions fed to mem0's
   * `customInstructions`. The operator's learning loop supplies these;
   * the package treats it as a black box. Default: none.
   */
  getLearningInstructions?: () => Promise<string | undefined>;

  /**
   * Optional: a host-provided fact-extraction LLM — cascade rung #1,
   * ahead of the API-key rungs (mem0-extraction-via-claude-session
   * D-002/D-003). The operator supplies `SessionExtractionLlm` riding
   * the Claude-session `anthropic-direct` transport; other hosts can
   * supply anything implementing mem0ai's LLM shape. Return `null`
   * when unavailable (no session, liveness probe failed, rung demoted)
   * — the key cascade then resolves exactly as before. Called on every
   * client (re)build, so a mid-process demotion takes effect at the
   * next TTL rebuild too.
   */
  getExtractionLlm?: () => Promise<import('./extraction-llm').ExtractionLlm | null>;

  /**
   * Postgres schema holding the memory tables (`memory_canonical` +
   * `memory_vec_*`). Default `'public'`. The host's migration must create
   * the tables in this schema. The operator passes `'harness_shared'`.
   */
  schema?: string;

  /**
   * Fallback Postgres database name when the admin URL has no path
   * component. Default `'postgres'`. The operator passes `'papercusp'`.
   */
  defaultDbName?: string;

  /**
   * Directory for mem0's local SQLite event-history file. Default the OS
   * tmpdir. The operator passes `~/.papercusp` so the log survives across
   * restarts. Set to `null` to force the in-memory (`:memory:`) history.
   */
  localStoreDir?: string | null;

  /**
   * Which `MemoryBackend` `getMemoryBackend()` serves — a registered
   * name (`'mem0'` / `'noop'` / anything added via
   * `registerMemoryBackend()`), a direct instance, or a **thunk** that
   * returns one of those (re-evaluated on every `getMemoryBackend()`
   * call). Default `'mem0'`. The operator feeds a thunk reading the
   * live operator setting (mem0-revive-or-retire) so the active backend
   * can be switched from the UI without a restart, falling back to
   * `PAPERCUSP_MEMORY_BACKEND` then `'mem0'`; the static string/instance
   * forms keep the store a config flip (generalize-memory-backend-swappable
   * D-004). The inline type-only import keeps config.ts free of a
   * runtime circular dependency on backend.ts.
   */
  backend?:
    | string
    | import('./backend').MemoryBackend
    | (() => string | import('./backend').MemoryBackend | undefined);
}

/** Resolved memory-table schema — host config, defaulting to `public`. */
export function memorySchema(): string {
  return memoryHost().schema ?? 'public';
}

// The host is stored on a process-global slot, NOT a module-level `let`.
// Under the operator's tsx runtime this module loads via a node_modules
// SYMLINK (node_modules/@papercusp/memory → packages/memory), and tsx's
// ESM loader resolves the symlink inconsistently — some import sites get
// the symlink path, others the realpath — so a module-level singleton
// FORKS into two instances: `configureMemory()` (called by the operator's
// lib/memory/configure.ts) sets `_host` on one, while the store's own
// `await import('./mem0-connection')` reads `memoryHost()` from the other
// → "@papercusp/memory is not configured" and memory is dead at runtime.
// `Symbol.for` keys the global registry, so every forked instance shares
// this one slot. (require.resolve canonicalizes to one realpath, which is
// why this only bit the ESM/tsx path, not CJS.)
const HOST_KEY = Symbol.for('@papercusp/memory:host');
type HostGlobal = typeof globalThis & { [HOST_KEY]?: MemoryHost | null };

/**
 * Wire the operator host seams. Call once at module load (the operator's
 * `lib/memory/configure.ts` does this for its side-effect). Idempotent —
 * last call wins.
 */
export function configureMemory(host: MemoryHost): void {
  (globalThis as HostGlobal)[HOST_KEY] = host;
}

/** Internal accessor — throws if the host hasn't been configured yet. */
export function memoryHost(): MemoryHost {
  const host = (globalThis as HostGlobal)[HOST_KEY];
  if (!host) {
    throw new Error(
      '@papercusp/memory is not configured — call configureMemory({ … }) before using the store (the operator does this in lib/memory/configure.ts).',
    );
  }
  return host;
}

/** Test/diagnostic helper: is a host wired? */
export function isMemoryConfigured(): boolean {
  return (globalThis as HostGlobal)[HOST_KEY] != null;
}
