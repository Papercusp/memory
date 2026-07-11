/**
 * P-003 (shared-embedding-sidecar-and-enrichment-2026-07-10; sidecar-REQUIRED
 * since WI-4021): the sidecar-first client seam. Contracts under test:
 *  - up-path: vectors come from the sidecar over the exact D-004 wire body;
 *  - required-path: with a url configured, ANY sidecar failure is retried
 *    briefly inside ONE total budget and then THROWN
 *    (sidecar_required_unavailable) — the in-process builder is NEVER built
 *    (D-003 retired: no silent in-process failover);
 *  - blip recovery: a restart-window failure recovers on a retry attempt
 *    within the same call, with down/up transitions reported once each;
 *  - no-url path: the in-process embedder is the sole engine (not a fallback);
 *  - identical-vector pin (D-002, seam-level): the two paths return the SAME
 *    vector for the same text. (The live bit-identical proof against a real
 *    model runs in the P-005 smoke.)
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

  it('required-path: persistent failure retries within the budget then THROWS — fallback never built', async () => {
    let fetchAttempts = 0;
    const failingFetch: typeof fetch = async () => {
      fetchAttempts++;
      throw new Error('connect ECONNREFUSED');
    };
    let fallbackBuilds = 0;
    const transitions: string[] = [];
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'document',
      url: 'http://127.0.0.1:1', // never reached — fetchFn injected
      fallback: () => {
        fallbackBuilds++;
        return async (text: string) => [text.length, 0, 0];
      },
      fetchFn: failingFetch,
      sleepFn: async () => {}, // no real backoff waits in tests
      onTransition: (state) => transitions.push(state),
    });

    await expect(embed('abc')).rejects.toThrow(/sidecar_required_unavailable/);
    expect(fetchAttempts).toBe(3); // DEFAULT_SIDECAR_MAX_ATTEMPTS
    expect(fallbackBuilds).toBe(0); // in-process is NEVER a fallback (WI-4021)
    expect(transitions).toEqual(['down']); // transition-only, not per-attempt
  });

  it('blip recovery: a restart-window failure recovers on a retry within the same call', async () => {
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
    const transitions: string[] = [];
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: stub.url,
      fallback: () => {
        throw new Error('must not build the in-process embedder');
      },
      fetchFn: flakyFetch,
      sleepFn: async () => {},
      onTransition: (state) => transitions.push(state),
    });

    expect(await embed('x')).toEqual(PINNED_VECTOR); // attempt 2 serves it
    expect(transitions).toEqual(['down', 'up']);
  });

  it('stops retrying once the budget cannot fit another attempt', async () => {
    let fetchAttempts = 0;
    const failingFetch: typeof fetch = async () => {
      fetchAttempts++;
      throw new Error('boom');
    };
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: 'http://127.0.0.1:1',
      fallback: () => async () => [1],
      fetchFn: failingFetch,
      timeoutMs: 100, // too small for the 250ms backoff + a meaningful retry
      onTransition: () => {},
    });
    await expect(embed('x')).rejects.toThrow(/sidecar_required_unavailable/);
    expect(fetchAttempts).toBe(1);
  });

  it('no url configured: pure in-process engine, no fetch at all', async () => {
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
