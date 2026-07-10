/**
 * P-003 (shared-embedding-sidecar-and-enrichment-2026-07-10): the sidecar-first
 * client seam. Contracts under test:
 *  - up-path: vectors come from the sidecar over the exact D-004 wire body;
 *  - down-path: ANY sidecar failure falls back to the in-process embedder
 *    (D-003 — never throws, never 'disabled'), with a cooldown so the down
 *    window costs ZERO extra sidecar round-trips;
 *  - re-probe: after the cooldown the next embed retries the sidecar;
 *  - identical-vector pin (D-002, seam-level): the two paths return the SAME
 *    vector for the same text — the failover is invisible to consumers. (The
 *    live bit-identical proof against a real model runs in the P-005 smoke.)
 */
import { afterEach, describe, expect, it } from 'vitest';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import {
  buildSidecarFirstEmbedder,
  resolveEmbedSidecarUrl,
  sidecarEmbedBatch,
  EMBED_SIDECAR_URL_ENV,
} from './sidecar-embedder';

const PINNED_VECTOR = [0.25, -0.5, 0.75];

/** Stub sidecar: answers /embed with one PINNED_VECTOR per text and records
 *  request bodies. Real loopback HTTP — the transport under test is fetch. */
async function startStubSidecar(
  handler?: (body: unknown, res: http.ServerResponse) => void,
): Promise<{ url: string; requests: unknown[]; close: () => Promise<void> }> {
  const requests: unknown[] = [];
  const server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', (c) => (raw += c));
    req.on('end', () => {
      const body = raw ? JSON.parse(raw) : undefined;
      requests.push(body);
      if (handler) {
        handler(body, res);
        return;
      }
      const texts = (body as { texts: string[] }).texts;
      res.setHeader('Content-Type', 'application/json');
      res.end(
        JSON.stringify({
          vectors: texts.map(() => PINNED_VECTOR),
          dims: PINNED_VECTOR.length,
          runtime: 'stub',
          modelRev: 'stub-model',
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

const closers: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});

describe('resolveEmbedSidecarUrl', () => {
  it('returns null when unset and strips a trailing slash when set', () => {
    expect(resolveEmbedSidecarUrl({})).toBeNull();
    expect(resolveEmbedSidecarUrl({ [EMBED_SIDECAR_URL_ENV]: '' })).toBeNull();
    expect(resolveEmbedSidecarUrl({ [EMBED_SIDECAR_URL_ENV]: 'http://127.0.0.1:3384/' })).toBe(
      'http://127.0.0.1:3384',
    );
  });
});

describe('sidecarEmbedBatch', () => {
  it('speaks the D-004 wire: {model, kind, texts} → {vectors, dims, runtime, modelRev}', async () => {
    const stub = await startStubSidecar();
    closers.push(stub.close);
    const res = await sidecarEmbedBatch(stub.url, { model: 'gemma', kind: 'document', texts: ['a', 'b'] });
    expect(res.vectors).toEqual([PINNED_VECTOR, PINNED_VECTOR]);
    expect(res.modelRev).toBe('stub-model');
    expect(stub.requests).toEqual([{ model: 'gemma', kind: 'document', texts: ['a', 'b'] }]);
  });

  it('throws on non-200 and on a vectors/texts count mismatch', async () => {
    const bad500 = await startStubSidecar((_body, res) => {
      res.statusCode = 500;
      res.end(JSON.stringify({ error: 'embed_failed: boom' }));
    });
    closers.push(bad500.close);
    await expect(
      sidecarEmbedBatch(bad500.url, { model: 'gemma', kind: 'query', texts: ['x'] }),
    ).rejects.toThrow(/sidecar_embed_500/);

    const badShape = await startStubSidecar((_body, res) => {
      res.end(JSON.stringify({ vectors: [], dims: 0, runtime: 'stub', modelRev: 'stub' }));
    });
    closers.push(badShape.close);
    await expect(
      sidecarEmbedBatch(badShape.url, { model: 'gemma', kind: 'query', texts: ['x'] }),
    ).rejects.toThrow(/sidecar_embed_bad_shape/);
  });
});

describe('buildSidecarFirstEmbedder', () => {
  it('up-path: serves vectors from the sidecar and never builds the fallback', async () => {
    const stub = await startStubSidecar();
    closers.push(stub.close);
    let fallbackBuilds = 0;
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: stub.url,
      fallback: () => {
        fallbackBuilds++;
        return async () => [9, 9, 9];
      },
      onTransition: () => {},
    });
    expect(await embed('hello')).toEqual(PINNED_VECTOR);
    expect(stub.requests).toEqual([{ model: 'gemma', kind: 'query', texts: ['hello'] }]);
    expect(fallbackBuilds).toBe(0);
  });

  it('down-path: falls back in-process, and the cooldown window skips the sidecar entirely', async () => {
    let fetchAttempts = 0;
    const failingFetch: typeof fetch = async () => {
      fetchAttempts++;
      throw new Error('connect ECONNREFUSED');
    };
    let t = 1_000_000;
    const transitions: string[] = [];
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'document',
      url: 'http://127.0.0.1:1', // never reached — fetchFn injected
      fallback: () => async (text: string) => [text.length, 0, 0],
      fetchFn: failingFetch,
      now: () => t,
      reprobeAfterMs: 30_000,
      onTransition: (state) => transitions.push(state),
    });

    expect(await embed('abc')).toEqual([3, 0, 0]); // failure #1 → fallback
    t += 1_000; // still inside the cooldown
    expect(await embed('abcd')).toEqual([4, 0, 0]);
    expect(fetchAttempts).toBe(1); // cooldown = zero extra sidecar round-trips
    expect(transitions).toEqual(['down']); // transition-only, not per-embed
  });

  it('re-probe: after the cooldown the sidecar is retried and recovery is reported', async () => {
    const stub = await startStubSidecar();
    closers.push(stub.close);
    let failNext = true;
    const flakyFetch: typeof fetch = async (...args) => {
      if (failNext) {
        failNext = false;
        throw new Error('connect ECONNREFUSED');
      }
      return fetch(...args);
    };
    let t = 1_000_000;
    const transitions: string[] = [];
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: stub.url,
      fallback: () => async () => [7, 7, 7],
      fetchFn: flakyFetch,
      now: () => t,
      reprobeAfterMs: 30_000,
      onTransition: (state) => transitions.push(state),
    });

    expect(await embed('x')).toEqual([7, 7, 7]); // down → fallback
    t += 30_001; // cooldown elapsed
    expect(await embed('x')).toEqual(PINNED_VECTOR); // re-probe succeeds
    expect(transitions).toEqual(['down', 'up']);
  });

  it('no url configured: pure fallback, no fetch at all', async () => {
    let fetched = 0;
    const spyFetch: typeof fetch = async () => {
      fetched++;
      throw new Error('must not be called');
    };
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'document',
      url: null,
      fallback: () => async () => [1, 2, 3],
      fetchFn: spyFetch,
    });
    expect(await embed('x')).toEqual([1, 2, 3]);
    expect(fetched).toBe(0);
  });

  it('identical-vector pin (D-002 at the seam): sidecar path and fallback path return the same vector', async () => {
    // Both paths are backed by the SAME embedder here — exactly the production
    // arrangement (the sidecar wraps the same builder the fallback uses). The
    // seam must preserve that equality bit-for-bit through the wire.
    const sharedEmbed = async (text: string): Promise<number[]> =>
      [...text].map((c, i) => (c.charCodeAt(0) * (i + 1)) / 255);
    const stub = await startStubSidecar((body, res) => {
      const texts = (body as { texts: string[] }).texts;
      void Promise.all(texts.map(sharedEmbed)).then((vectors) => {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ vectors, dims: vectors[0].length, runtime: 'stub', modelRev: 'stub' }));
      });
    });
    closers.push(stub.close);

    const viaSidecar = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: stub.url,
      fallback: () => sharedEmbed,
      onTransition: () => {},
    });
    const viaFallback = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: null,
      fallback: () => sharedEmbed,
    });

    const text = 'space discipline: same model, same vectors';
    expect(await viaSidecar(text)).toEqual(await viaFallback(text));
  });
});
