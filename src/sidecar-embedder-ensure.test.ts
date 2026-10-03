/**
 * P-531 (plan agent-capacity-and-cost-gcp-2026-09-30, WI-10005523): the
 * sidecar-first embedder's per-attempt `ensure` hook.
 *
 * A sidecar this process spawned can exit after an idle period. The client
 * must re-establish it before an attempt instead of failing until a crash
 * respawn that never comes (an announced idle exit is deliberately NOT
 * crash-respawned). These tests pin:
 *  - ensure runs before EVERY attempt, so a sidecar that went away between
 *    attempts is re-launched on the retry;
 *  - a hanging ensure is bounded by the same total budget (no unbounded wait);
 *  - without the hook nothing changes (explicit-URL sidecars are owned
 *    elsewhere).
 */
import { describe, expect, it } from 'vitest';
import { buildSidecarFirstEmbedder, settleSidecarEnsureWithin } from './sidecar-embedder';

const VECTOR = [0.25, 0.5, 0.75];

function okResponse(): Response {
  return new Response(JSON.stringify({ vectors: [VECTOR], dims: 3, runtime: 'stub', modelRev: 'stub' }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

const noFallback = () => {
  throw new Error('in-process fallback must never be built when a url is set');
};

describe('buildSidecarFirstEmbedder ensure hook (P-531)', () => {
  it('re-ensures before the retry, so a sidecar that exited is re-launched and the embed succeeds', async () => {
    const events: string[] = [];
    let sidecarUp = false;
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: 'http://127.0.0.1:1',
      fallback: noFallback,
      sleepFn: async () => {},
      onTransition: (state) => events.push(`transition:${state}`),
      // The first ensure finds the (stale) sidecar "running" and does nothing;
      // the fetch then fails because it already exited. The second ensure
      // re-launches it.
      ensure: async () => {
        events.push('ensure');
        if (events.filter((e) => e === 'ensure').length >= 2) sidecarUp = true;
      },
      fetchFn: (async () => {
        events.push('fetch');
        if (!sidecarUp) throw new Error('connect ECONNREFUSED');
        return okResponse();
      }) as typeof fetch,
    });

    await expect(embed('hello')).resolves.toEqual(VECTOR);
    expect(events).toEqual(['ensure', 'fetch', 'transition:down', 'ensure', 'fetch', 'transition:up']);
  });

  it('calls ensure exactly once per attempt on the happy path', async () => {
    let ensures = 0;
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'document',
      url: 'http://127.0.0.1:1',
      fallback: noFallback,
      ensure: async () => {
        ensures++;
      },
      fetchFn: (async () => okResponse()) as typeof fetch,
    });
    await embed('a');
    await embed('b');
    expect(ensures).toBe(2);
  });

  it('a hanging ensure is bounded by the total budget and surfaces as sidecar_required_unavailable', async () => {
    let fetches = 0;
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: 'http://127.0.0.1:1',
      fallback: noFallback,
      timeoutMs: 60,
      sleepFn: async () => {},
      onTransition: () => {},
      ensure: () => new Promise<void>(() => {}), // never settles
      fetchFn: (async () => {
        fetches++;
        return okResponse();
      }) as typeof fetch,
    });
    const started = Date.now();
    await expect(embed('x')).rejects.toThrow(/sidecar_required_unavailable: sidecar_ensure_timeout/);
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(fetches).toBe(0); // never sent work to a sidecar that was not re-established
  });

  it('without an ensure hook the client behaves exactly as before (one fetch, no extra work)', async () => {
    let fetches = 0;
    const embed = buildSidecarFirstEmbedder({
      model: 'harrier',
      kind: 'document',
      url: 'http://127.0.0.1:1',
      fallback: noFallback,
      fetchFn: (async () => {
        fetches++;
        return okResponse();
      }) as typeof fetch,
    });
    await expect(embed('y')).resolves.toEqual(VECTOR);
    expect(fetches).toBe(1);
  });
});

describe('settleSidecarEnsureWithin', () => {
  it('resolves when the work settles in time', async () => {
    // P-530: the settle passes the ensure's value through (it may be the URL).
    await expect(settleSidecarEnsureWithin(Promise.resolve('ok'), 1_000)).resolves.toBe('ok');
  });

  it('rejects with sidecar_ensure_timeout when the work outlives the bound', async () => {
    await expect(settleSidecarEnsureWithin(new Promise(() => {}), 10)).rejects.toThrow(/sidecar_ensure_timeout/);
  });

  it('a rejection that lands after the timeout is not an unhandled rejection', async () => {
    let rejectLate!: (e: Error) => void;
    const late = new Promise<void>((_, reject) => {
      rejectLate = reject;
    });
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      await expect(settleSidecarEnsureWithin(late, 5)).rejects.toThrow(/sidecar_ensure_timeout/);
      rejectLate(new Error('spawn failed later'));
      await new Promise((r) => setTimeout(r, 20));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });

  it('honours the caller abort signal', async () => {
    const ac = new AbortController();
    const pending = settleSidecarEnsureWithin(new Promise(() => {}), 10_000, ac.signal);
    ac.abort(new Error('caller gave up'));
    await expect(pending).rejects.toThrow(/caller gave up/);
  });
});

/**
 * WI-10005932: a sidecar this process is configured to spawn whose first start
 * timed out has no URL yet. Before the fix the client took a null url as "no
 * sidecar" and built the in-process model in the main Server, which then held
 * a second copy next to the sidecar for good (P-532d pct3: +1.8 GB anon, 150
 * .onnx maps in the main process). With an ensure hook the client must stay
 * sidecar-only and take the address from a later ensure.
 */
describe('buildSidecarFirstEmbedder with a null url and an ensure hook (WI-10005932)', () => {
  it('never builds the fallback and uses the URL a later ensure reports', async () => {
    const events: string[] = [];
    let ensures = 0;
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'document',
      url: null,
      fallback: noFallback,
      sleepFn: async () => {},
      onTransition: (state) => events.push(`transition:${state}`),
      ensure: async () => {
        ensures++;
        return ensures >= 2 ? 'http://127.0.0.1:41223/' : null;
      },
      fetchFn: (async (input: string | URL | Request) => {
        events.push(`fetch:${String(input)}`);
        return okResponse();
      }) as typeof fetch,
    });

    await expect(embed('hello')).resolves.toEqual(VECTOR);
    expect(ensures).toBe(2);
    expect(events).toEqual(['transition:down', 'fetch:http://127.0.0.1:41223/embed', 'transition:up']);
  });

  it('keeps the reported URL for later calls', async () => {
    const fetched: string[] = [];
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'query',
      url: null,
      fallback: noFallback,
      sleepFn: async () => {},
      onTransition: () => {},
      // Reports the address once, then (like an already-running sidecar) nothing new.
      ensure: (() => {
        let first = true;
        return async () => {
          const out = first ? 'http://127.0.0.1:41224' : undefined;
          first = false;
          return out;
        };
      })(),
      fetchFn: (async (input: string | URL | Request) => {
        fetched.push(String(input));
        return okResponse();
      }) as typeof fetch,
    });

    await embed('a');
    await embed('b');
    expect(fetched).toEqual(['http://127.0.0.1:41224/embed', 'http://127.0.0.1:41224/embed']);
  });

  it('fails within the budget when no address ever arrives, without fetching or loading a model', async () => {
    let fetches = 0;
    const embed = buildSidecarFirstEmbedder({
      model: 'gemma',
      kind: 'document',
      url: null,
      fallback: noFallback,
      sleepFn: async () => {},
      maxAttempts: 3,
      onTransition: () => {},
      ensure: async () => null,
      fetchFn: (async () => {
        fetches++;
        return okResponse();
      }) as typeof fetch,
    });

    await expect(embed('hello')).rejects.toThrow(/sidecar_required_unavailable: sidecar_not_ready/);
    expect(fetches).toBe(0);
  });
});
