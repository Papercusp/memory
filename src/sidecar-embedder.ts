/**
 * Sidecar-first embedder client (P-003, plan
 * shared-embedding-sidecar-and-enrichment-2026-07-10).
 *
 * The consumer-side seam for the shared embedding sidecar: an EmbedFn that
 * tries the loopback sidecar (D-004 wire — POST {url}/embed { model, kind,
 * texts }) and AUTOMATICALLY falls back to the caller-supplied in-process
 * embedder when the sidecar is down. Embedding is never 'disabled' by a
 * sidecar outage (D-003) — worst case is exactly today's in-process behavior.
 *
 * Vectors from the two paths are BIT-IDENTICAL by construction (D-002): the
 * sidecar wraps the SAME @papercusp/memory builders (model, prompts, MRL
 * truncation, ORT runtime) the fallback uses — same space, so a mid-stream
 * failover never mixes spaces.
 *
 * Down-handling: any sidecar failure (connect refused, timeout, non-200, bad
 * shape) marks the sidecar down for `reprobeAfterMs` (default 30s); embeds in
 * that window go straight to the fallback with zero sidecar round-trips. After
 * the cooldown the next embed simply tries the sidecar again — the embed IS
 * the probe (a separate healthz round-trip would buy nothing the embed itself
 * doesn't prove). The default per-embed budget is generous (15s) because a
 * freshly-spawned sidecar may still be warm-loading the model on its first
 * request; a budget trip just means one cooldown on the fallback path.
 *
 * Plain `fetch`, zero new dependencies — this library is deliberately dep-free
 * (pg only); the server lives in operator-core, never here.
 */

import type { GemmaEmbedKind } from './gemma-embedder';

type EmbedFn = (text: string) => Promise<number[]>;

/** Consumers point at a sidecar by setting this (e.g. http://127.0.0.1:3384). */
export const EMBED_SIDECAR_URL_ENV = 'PAPERCUSP_EMBED_SIDECAR_URL';

/** The sidecar base URL this process should use, or null when none is
 *  configured (→ pure in-process embedding). */
export function resolveEmbedSidecarUrl(env: NodeJS.ProcessEnv = process.env): string | null {
  const url = env[EMBED_SIDECAR_URL_ENV]?.trim();
  return url ? url.replace(/\/$/, '') : null;
}

export interface SidecarEmbedBatchOpts {
  model: string;
  kind: GemmaEmbedKind;
  texts: string[];
  timeoutMs?: number;
  fetchFn?: typeof fetch;
}

export interface SidecarEmbedResponse {
  vectors: number[][];
  dims: number;
  runtime: string;
  modelRev: string;
}

export const DEFAULT_SIDECAR_TIMEOUT_MS = 15_000;
export const DEFAULT_REPROBE_AFTER_MS = 30_000;

/**
 * One D-004 wire call. Throws on ANY failure (network, timeout, non-200,
 * malformed/mismatched response) — callers own the fallback decision.
 * Exported for batch consumers (embed-backfill, P-004) that want the
 * texts[] amortization the single-text EmbedFn seam can't express.
 */
export async function sidecarEmbedBatch(url: string, opts: SidecarEmbedBatchOpts): Promise<SidecarEmbedResponse> {
  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_SIDECAR_TIMEOUT_MS;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchFn(`${url.replace(/\/$/, '')}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: opts.model, kind: opts.kind, texts: opts.texts }),
      signal: ctl.signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`sidecar_embed_${res.status}: ${detail.slice(0, 200)}`);
    }
    const body = (await res.json()) as SidecarEmbedResponse;
    if (
      !Array.isArray(body.vectors) ||
      body.vectors.length !== opts.texts.length ||
      body.vectors.some((v) => !Array.isArray(v) || v.length === 0 || typeof v[0] !== 'number')
    ) {
      throw new Error('sidecar_embed_bad_shape: vectors missing/mismatched');
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

export interface SidecarFirstEmbedderOpts {
  /** Sidecar-side model name ('gemma' | 'local'). */
  model: string;
  /** Asymmetric task side — the sidecar owns the actual prompt text (D-004). */
  kind: GemmaEmbedKind;
  /** Lazy builder for the in-process fallback embedder (built at most once,
   *  on first need — never eagerly, so the sidecar-served happy path loads no
   *  local model). */
  fallback: () => EmbedFn | Promise<EmbedFn>;
  /** Sidecar base URL; defaults to resolveEmbedSidecarUrl(). null/absent ⇒
   *  pure fallback. */
  url?: string | null;
  timeoutMs?: number;
  /** How long a sidecar failure parks embeds on the fallback before retrying. */
  reprobeAfterMs?: number;
  fetchFn?: typeof fetch;
  /** Clock seam for tests. */
  now?: () => number;
  /** Down/up transition logging seam (default console.warn, transition-only). */
  onTransition?: (state: 'down' | 'up', detail: string) => void;
}

/**
 * Build a sidecar-first EmbedFn: sidecar when it answers, in-process fallback
 * when it doesn't, cooldown between retries. Same closure shape as
 * buildGemmaEmbedder/buildLocalEmbedder so it drops into every existing
 * embedder seam.
 */
export function buildSidecarFirstEmbedder(opts: SidecarFirstEmbedderOpts): EmbedFn {
  const url = opts.url === undefined ? resolveEmbedSidecarUrl() : opts.url;
  const now = opts.now ?? Date.now;
  const reprobeAfterMs = opts.reprobeAfterMs ?? DEFAULT_REPROBE_AFTER_MS;
  const onTransition =
    opts.onTransition ??
    ((state: 'down' | 'up', detail: string) =>
      console.warn(`[sidecar-embedder] ${opts.model}:${opts.kind} sidecar ${state}: ${detail}`));

  let fallbackPromise: Promise<EmbedFn> | null = null;
  const getFallback = (): Promise<EmbedFn> => {
    // A failed fallback build is not memoized — the next embed retries it.
    if (!fallbackPromise) {
      fallbackPromise = Promise.resolve()
        .then(() => opts.fallback())
        .catch((e) => {
          fallbackPromise = null;
          throw e;
        });
    }
    return fallbackPromise;
  };

  if (!url) {
    // No sidecar configured: the seam degenerates to the plain in-process
    // embedder with zero per-call overhead.
    return async (text: string) => (await getFallback())(text);
  }

  let downUntil = 0;
  let wasDown = false;

  return async (text: string): Promise<number[]> => {
    if (now() >= downUntil) {
      try {
        const res = await sidecarEmbedBatch(url, {
          model: opts.model,
          kind: opts.kind,
          texts: [text],
          timeoutMs: opts.timeoutMs,
          fetchFn: opts.fetchFn,
        });
        if (wasDown) {
          wasDown = false;
          onTransition('up', 'sidecar answering again');
        }
        return res.vectors[0];
      } catch (e) {
        downUntil = now() + reprobeAfterMs;
        if (!wasDown) {
          wasDown = true;
          onTransition(
            'down',
            `${e instanceof Error ? e.message : String(e)} — embedding continues IN-PROCESS (D-003), retry in ${Math.round(reprobeAfterMs / 1000)}s`,
          );
        }
      }
    }
    return (await getFallback())(text);
  };
}
