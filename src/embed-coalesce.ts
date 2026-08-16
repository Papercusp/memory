/**
 * Embed-call coalescing + short-TTL memoization (WI-5094).
 *
 * The operator's pre-turn memory injection issues THREE backend searches with
 * the SAME query text (user / harness / hive pools — injection.ts), and each
 * cosine leg embeds that text independently. On an in-process embedder (the
 * desktop-spawned operator, the staging host — no sidecar cache anywhere on
 * the path) that is 3 identical ONNX embeds per turn, largely serialized on
 * the embedder's concurrency cap: measured 2.6–4.5s of `searchMs` per turn,
 * which was the DOMINANT cost of the whole prompt build (prompt-phases,
 * 2026-07-16).
 *
 * This wrapper collapses those to one real embed at the single choke point
 * every embed already flows through (`_currentEmbedFn` in mem0-client.ts —
 * both the EI-12962 batched path and mem0's patched 'custom' embedder):
 *
 *  - IN-FLIGHT COALESCING: concurrent calls with identical text share one
 *    underlying promise (the injection's three legs pay one embed).
 *  - SHORT-TTL LRU: a just-computed vector is served for `ttlMs` so
 *    back-to-back turns over the same context skip the embedder entirely.
 *
 * Safety: embedding is deterministic for a given fn (same model + kind baked
 * into the closure), so same-text memoization is behavior-preserving. The fn
 * is re-wrapped whenever the embedder is rebuilt (tryLoad reassigns
 * `_currentEmbedFn`), so a backend/model flip never serves stale vectors.
 * Failures are NEVER cached: a rejection propagates to every coalesced
 * waiter and the next call retries the real embedder.
 */

import { createHash } from 'node:crypto';

export interface CoalesceEmbedOptions {
  /** How long a computed vector may be served from cache. Default 60s. */
  ttlMs?: number;
  /** Max cached vectors (LRU eviction). Default 64 (≲0.5MB at 768 dims). */
  maxEntries?: number;
  /** Clock seam for tests. */
  now?: () => number;
}

/** Test/telemetry handle returned alongside the wrapped fn by `coalesceEmbedFnWithStats`. */
export interface CoalesceEmbedStats {
  hits: number;
  coalesced: number;
  misses: number;
  size: () => number;
}

export function coalesceEmbedFn(
  fn: (text: string) => Promise<number[]>,
  opts: CoalesceEmbedOptions = {},
): (text: string) => Promise<number[]> {
  return coalesceEmbedFnWithStats(fn, opts).embed;
}

export function coalesceEmbedFnWithStats(
  fn: (text: string) => Promise<number[]>,
  opts: CoalesceEmbedOptions = {},
): { embed: (text: string) => Promise<number[]>; stats: CoalesceEmbedStats } {
  const ttlMs = opts.ttlMs ?? 60_000;
  const maxEntries = opts.maxEntries ?? 64;
  const now = opts.now ?? Date.now;

  const inFlight = new Map<string, Promise<number[]>>();
  // Map iteration order = insertion order; a fresh hit is re-inserted so the
  // first key is always the least-recently-used one.
  const done = new Map<string, { vector: number[]; at: number }>();
  const stats: CoalesceEmbedStats = { hits: 0, coalesced: 0, misses: 0, size: () => done.size };

  const embed = (text: string): Promise<number[]> => {
    const key = createHash('sha256').update(text).digest('base64');

    const hit = done.get(key);
    if (hit) {
      if (now() - hit.at < ttlMs) {
        stats.hits += 1;
        done.delete(key);
        done.set(key, hit);
        return Promise.resolve(hit.vector);
      }
      done.delete(key); // expired
    }

    const pending = inFlight.get(key);
    if (pending) {
      stats.coalesced += 1;
      return pending;
    }

    stats.misses += 1;
    const p = (async () => {
      const vector = await fn(text);
      // Cache only a real vector — an empty/degenerate result must not stick.
      if (Array.isArray(vector) && vector.length > 0) {
        done.set(key, { vector, at: now() });
        while (done.size > maxEntries) {
          const oldest = done.keys().next().value;
          if (oldest === undefined) break;
          done.delete(oldest);
        }
      }
      return vector;
    })();
    // Settled promises leave the in-flight table either way; a rejection is
    // therefore never coalesced onto later calls (they retry the embedder).
    const tracked = p.finally(() => {
      inFlight.delete(key);
    });
    inFlight.set(key, tracked);
    return tracked;
  };

  return { embed, stats };
}
