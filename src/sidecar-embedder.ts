/**
 * Sidecar-first embedder client (P-003, plan
 * shared-embedding-sidecar-and-enrichment-2026-07-10; sidecar-REQUIRED since
 * WI-4021, owner directive 2026-07-11).
 *
 * The consumer-side seam for the shared embedding sidecar: an EmbedFn over the
 * loopback sidecar (D-004 wire — POST {url}/embed { model, kind, texts }).
 *
 * AVAILABILITY CONTRACT (v2 — D-003 retired): when a sidecar URL is
 * configured, the sidecar is REQUIRED. A failure is retried briefly (the
 * sidecar is systemd-supervised with Restart=always, so a crash window is
 * seconds) and then THROWN — never silently absorbed by an in-process model
 * load. The old D-003 in-process failover let a stalling sidecar drag every
 * embedding host into duplicate in-process model loads (the 2026-07-11
 * "mem0 down" flap incident) and hid sidecar sickness instead of surfacing
 * it; memory writes survive a real outage via the write-ahead journal
 * (memory-write-journal-auto-recovery-2026-07-11), reads fail loudly.
 *
 * The caller-supplied in-process builder is used ONLY when NO sidecar is
 * configured (url null — desktop installs, tests, bench rigs): there it is
 * the sole engine, not a fallback. Vectors are BIT-IDENTICAL either way
 * (D-002): the sidecar wraps the SAME @papercusp/memory builders (model,
 * prompts, MRL truncation, ORT runtime) — same space on every path.
 *
 * Retry shape: one total `timeoutMs` budget (default 15s — a freshly
 * restarted sidecar may still be warm-loading) spans ALL attempts, so a
 * slow-but-alive sidecar is never hammered past its budget; fast failures
 * (connect refused during a restart) get up to `maxAttempts` tries with short
 * linear backoff inside that budget.
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
/** Attempts per embed when the sidecar is required (fast failures only — the
 *  shared `timeoutMs` budget caps total wall time regardless). */
export const DEFAULT_SIDECAR_MAX_ATTEMPTS = 3;

/** The sidecar server's own per-text char cap (embed-sidecar-server.ts
 *  MAX_TEXT_CHARS — kept in sync here since the client, not the server, is
 *  the one place every embed caller funnels through). A text over this limit
 *  is truncated client-side BEFORE it is ever sent (EI-14101): the sidecar's
 *  own 400 for this condition says "truncate before embedding", so truncating
 *  at the ONE shared client seam means no individual caller (embed-backfill,
 *  write-journal, memory writes, …) has to reimplement the cap. */
export const SIDECAR_MAX_TEXT_CHARS = 32_000;

/** Thrown by sidecarEmbedBatch on a non-2xx sidecar response. Carries the
 *  HTTP status so a retry policy can tell a DETERMINISTIC client-side
 *  rejection (4xx — the exact same request can never succeed on retry) apart
 *  from a transient server/network failure (5xx, connect refused, timeout) —
 *  see isNonRetryableSidecarError. */
export class SidecarEmbedHttpError extends Error {
  readonly status: number;
  constructor(status: number, detail: string) {
    super(`sidecar_embed_${status}: ${detail}`);
    this.name = 'SidecarEmbedHttpError';
    this.status = status;
  }
}

/** True for a sidecar failure that retrying can never fix (a 4xx: the
 *  sidecar looked at THIS request and rejected it — bad shape, unknown
 *  model, oversized text pre-truncation, …). False for anything else
 *  (network/connect/timeout/5xx), which IS worth a bounded retry — the
 *  sidecar may just be mid-restart. */
export function isNonRetryableSidecarError(e: unknown): boolean {
  return e instanceof SidecarEmbedHttpError && e.status >= 400 && e.status < 500;
}

/**
 * One D-004 wire call. Throws on ANY failure (network, timeout, non-200,
 * malformed/mismatched response) — callers own the fallback decision.
 * Exported for batch consumers (embed-backfill, P-004) that want the
 * texts[] amortization the single-text EmbedFn seam can't express.
 *
 * Texts over SIDECAR_MAX_TEXT_CHARS are truncated before submission (the
 * sidecar rejects them outright otherwise — a deterministic 400 that no
 * amount of retrying would fix).
 */
export async function sidecarEmbedBatch(url: string, opts: SidecarEmbedBatchOpts): Promise<SidecarEmbedResponse> {
  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_SIDECAR_TIMEOUT_MS;
  const texts = opts.texts.map((t) => (t.length > SIDECAR_MAX_TEXT_CHARS ? t.slice(0, SIDECAR_MAX_TEXT_CHARS) : t));
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchFn(`${url.replace(/\/$/, '')}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: opts.model, kind: opts.kind, texts }),
      signal: ctl.signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new SidecarEmbedHttpError(res.status, detail.slice(0, 200));
    }
    const body = (await res.json()) as SidecarEmbedResponse;
    if (
      !Array.isArray(body.vectors) ||
      body.vectors.length !== texts.length ||
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
  /** Sidecar-side model name ('gemma' | 'local' | 'harrier'). */
  model: string;
  /** Asymmetric task side — the sidecar owns the actual prompt text (D-004). */
  kind: GemmaEmbedKind;
  /** Lazy builder for the in-process embedder — used ONLY when no sidecar is
   *  configured (url null), where it is the sole engine. When a url is set it
   *  is never built: the sidecar is required (WI-4021, D-003 retired). */
  fallback: () => EmbedFn | Promise<EmbedFn>;
  /** Sidecar base URL; defaults to resolveEmbedSidecarUrl(). null/absent ⇒
   *  pure in-process. */
  url?: string | null;
  /** TOTAL budget per embed across every attempt (default 15s). */
  timeoutMs?: number;
  /** Attempts within the budget on sidecar failure (default 3). */
  maxAttempts?: number;
  fetchFn?: typeof fetch;
  /** Clock seam for tests. */
  now?: () => number;
  /** Backoff-sleep seam for tests. */
  sleepFn?: (ms: number) => Promise<void>;
  /** Down/up/rejected transition logging seam (default console.warn,
   *  transition-only). 'rejected' = the sidecar is UP and correctly
   *  refusing a bad request (deterministic 4xx) — distinct from 'down' so
   *  the log never claims the sidecar is unavailable when it isn't. */
  onTransition?: (state: 'down' | 'up' | 'rejected', detail: string) => void;
}

/**
 * Build a sidecar-first EmbedFn. With a url: sidecar-REQUIRED — brief retries
 * inside one total budget, then throw (`sidecar_required_unavailable`); the
 * in-process builder is never touched. Without a url: the plain in-process
 * embedder. Same closure shape as buildGemmaEmbedder/buildLocalEmbedder so it
 * drops into every existing embedder seam.
 */
export function buildSidecarFirstEmbedder(opts: SidecarFirstEmbedderOpts): EmbedFn {
  const url = opts.url === undefined ? resolveEmbedSidecarUrl() : opts.url;
  const now = opts.now ?? Date.now;
  const sleep = opts.sleepFn ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const timeoutMs = opts.timeoutMs ?? DEFAULT_SIDECAR_TIMEOUT_MS;
  const maxAttempts = Math.max(1, opts.maxAttempts ?? DEFAULT_SIDECAR_MAX_ATTEMPTS);
  const onTransition =
    opts.onTransition ??
    ((state: 'down' | 'up' | 'rejected', detail: string) =>
      console.warn(`[sidecar-embedder] ${opts.model}:${opts.kind} sidecar ${state}: ${detail}`));

  if (!url) {
    // No sidecar configured: the plain in-process embedder is the sole engine
    // (desktop installs, tests, bench rigs) with zero per-call overhead. A
    // failed build is not memoized — the next embed retries it.
    let fallbackPromise: Promise<EmbedFn> | null = null;
    const getFallback = (): Promise<EmbedFn> => {
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
    return async (text: string) => (await getFallback())(text);
  }

  let wasDown = false;

  return async (text: string): Promise<number[]> => {
    const deadline = now() + timeoutMs;
    let lastErr: unknown;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      const remaining = deadline - now();
      if (remaining <= 0) break;
      try {
        const res = await sidecarEmbedBatch(url, {
          model: opts.model,
          kind: opts.kind,
          texts: [text],
          timeoutMs: remaining,
          fetchFn: opts.fetchFn,
        });
        if (wasDown) {
          wasDown = false;
          onTransition('up', `sidecar answering again (attempt ${attempt})`);
        }
        return res.vectors[0];
      } catch (e) {
        lastErr = e;
        if (isNonRetryableSidecarError(e)) {
          // Deterministic 4xx: the sidecar is UP and correctly rejecting
          // THIS exact request (bad shape / oversized text / unknown model)
          // — retrying the same payload can never succeed, so stop right
          // away instead of burning the retry budget (EI-14101). Logged as
          // 'rejected', never 'down' — the sidecar isn't unavailable.
          onTransition(
            'rejected',
            `${e instanceof Error ? e.message : String(e)} — sidecar correctly rejected the request (non-retryable, not a downtime issue)`,
          );
          break;
        }
        if (!wasDown) {
          wasDown = true;
          onTransition(
            'down',
            `${e instanceof Error ? e.message : String(e)} — sidecar is REQUIRED (no in-process fallback, WI-4021); retrying within budget`,
          );
        }
        // Short linear backoff before the next attempt — but only when enough
        // budget remains for the backoff AND a meaningful retry.
        const backoffMs = 250 * attempt;
        if (attempt < maxAttempts && deadline - now() > backoffMs + 250) await sleep(backoffMs);
        else break;
      }
    }
    throw isNonRetryableSidecarError(lastErr)
      ? new Error(
          `sidecar_rejected_request: ${lastErr instanceof Error ? lastErr.message : String(lastErr)} ` +
            `(${url}, ${opts.model}:${opts.kind}) — the sidecar rejected this request (non-retryable); ` +
            'check payload shape/size — this is not a downtime issue',
        )
      : new Error(
          `sidecar_required_unavailable: ${lastErr instanceof Error ? lastErr.message : String(lastErr)} ` +
            `(${url}, ${opts.model}:${opts.kind}, budget ${timeoutMs}ms) — embedding requires the sidecar; ` +
            'writes are parked in the memory write journal and auto-recover when it returns',
        );
  };
}
