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
}

let _host: MemoryHost | null = null;

/**
 * Wire the operator host seams. Call once at module load (the operator's
 * `lib/memory/configure.ts` does this for its side-effect). Idempotent —
 * last call wins.
 */
export function configureMemory(host: MemoryHost): void {
  _host = host;
}

/** Internal accessor — throws if the host hasn't been configured yet. */
export function memoryHost(): MemoryHost {
  if (!_host) {
    throw new Error(
      '@papercusp/memory is not configured — call configureMemory({ … }) before using the store (the operator does this in lib/memory/configure.ts).',
    );
  }
  return _host;
}

/** Test/diagnostic helper: is a host wired? */
export function isMemoryConfigured(): boolean {
  return _host !== null;
}
