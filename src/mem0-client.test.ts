import { afterEach, describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { satisfies } from '../../../../scripts/check-peer-dep-conflicts.mjs';
import { patchEmbedderFactory, _setCurrentEmbedFnForTest } from './mem0-client';
import { Mem0Backend } from './mem0-backend';
import type { MemoryClient } from './mem0-client';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('installed mem0 peer compatibility', () => {
  const require = createRequire(import.meta.url);
  const manifest = JSON.parse(readFileSync(
    resolve(dirname(require.resolve('mem0ai/oss')), '../../package.json'), 'utf8',
  )) as {
    peerDependencies: Record<string, string>;
    peerDependenciesMeta: Record<string, { optional?: boolean }>;
  };

  it('accepts the host pg and type packages without forcing obsolete exact versions', () => {
    // mem0 3.1.8 pinned pg 8.11.3 and @types/pg 8.11.0, rejecting the
    // compatible versions used by our injected PostgreSQL memory store.
    for (const name of ['pg', '@types/pg']) {
      const installed = JSON.parse(readFileSync(require.resolve(`${name}/package.json`), 'utf8'));
      expect(satisfies(installed.version, manifest.peerDependencies[name])).toBe(true);
    }
  });

  it('keeps unused upstream stores and test tooling out of required runtime peers', () => {
    // The host supplies its own store; it does not use mem0's natural/PGVector
    // adapters. These peers are optional upstream, and Jest is test-only.
    for (const name of ['pg', '@types/pg', 'natural']) {
      expect(manifest.peerDependenciesMeta[name]?.optional).toBe(true);
    }
    expect(manifest.peerDependencies['@types/jest']).toBeUndefined();
  });

  it('resolves the host SDK using the tested mem0-specific compatibility override', () => {
    const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
    const host = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
    const installed = JSON.parse(readFileSync(
      resolve(dirname(require.resolve('@anthropic-ai/sdk')), 'package.json'), 'utf8',
    ));
    expect(satisfies(installed.version, host.overrides.mem0ai['@anthropic-ai/sdk'])).toBe(true);
  });

  it.each(['ESM', 'CommonJS'] as const)('runs the real mem0 %s Anthropic adapter with the host SDK', async (format) => {
    // mem0's optional SDK peer predates the host's Agent SDK requirement. Prove
    // its messages.create contract before widening that one package's edge.
    const oss = format === 'ESM' ? await import('mem0ai/oss') : require('mem0ai/oss');
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      id: 'msg_mem0_sdk', type: 'message', role: 'assistant', model: 'claude-test',
      content: [{ type: 'text', text: 'compatible' }], stop_reason: 'end_turn',
      stop_sequence: null, usage: { input_tokens: 1, output_tokens: 1 },
    }), { status: 200, headers: { 'content-type': 'application/json' } }));
    const llm = oss.LLMFactory.create('anthropic', {
      apiKey: 'test-key', baseURL: 'http://sdk-patch.test', model: 'claude-test', maxTokens: 8,
    });
    await expect(llm.generateResponse([
      { role: 'system', content: 'Remember this.' }, { role: 'user', content: 'hello' },
    ])).resolves.toBe('compatible');
    expect(fetch).toHaveBeenCalledOnce();
    const [url, init] = fetch.mock.calls[0];
    expect(String(url)).toContain('/v1/messages');
    expect(JSON.parse(String(init?.body))).toMatchObject({
      model: 'claude-test', max_tokens: 8, system: 'Remember this.',
      messages: [{ role: 'user', content: 'hello' }],
    });
  });
});

/**
 * Regression guard for the mem0ai 3.x compatibility fix (commit 879fc733).
 *
 * Before the fix, getMemoryClient() ALWAYS returned null: mem0ai 3.0.3's
 * EmbedderFactory rejects the 'custom' provider the store uses ("Unsupported
 * embedder provider: custom") and only VectorStoreFactory was patched, so the
 * Memory constructor threw and was swallowed — every memory:* tool silently
 * returned mem0_unavailable.
 *
 * This asserts the fix at its seam — patchEmbedderFactory must make mem0ai's own
 * EmbedderFactory accept the 'custom' provider — which is the right level for
 * THIS regression regardless of reachability.
 *
 * ⚠ The reason originally given here is no longer true, and it mattered: the
 * store used to load mem0ai through a bare `new Function('return import(s)')`
 * (dodging bundler static analysis) whose realm has no import callback under
 * vitest's module runner, so `getMemoryClient()` was unreachable from any test.
 * That is fixed — `./dynamic-import` keeps the bundler blind while working in a
 * VM realm (context-injection-audit-2026-07-28 P-002), so an end-to-end drive is
 * now possible if a future regression warrants one. Don't cite the old
 * limitation as a reason not to write one.
 */
describe('patchEmbedderFactory — mem0ai 3.x custom-embedder compatibility', () => {
  it('teaches mem0ai EmbedderFactory the custom provider (was unsupported)', async () => {
    const oss = (await import('mem0ai/oss')) as unknown as {
      EmbedderFactory: { create: (p: string, c: Record<string, unknown>) => unknown };
    };

    // Upstream gap this fix exists for: mem0ai 3.x has no 'custom' embedder.
    expect(() => oss.EmbedderFactory.create('custom', {})).toThrow(/custom/i);

    patchEmbedderFactory(oss);

    // After the patch, 'custom' yields an embedder implementing mem0's
    // interface (embed + embedBatch) instead of throwing.
    const emb = oss.EmbedderFactory.create('custom', {}) as {
      embed: unknown;
      embedBatch: unknown;
    };
    expect(typeof emb.embed).toBe('function');
    expect(typeof emb.embedBatch).toBe('function');

    // The deepest invariant: .embed() / .embedBatch() must route to the
    // INJECTED fn (set by tryLoad from the host's resolveEmbedder), NOT to
    // `config.embed` — which mem0's mergeConfig strips during Zod validation.
    // The bare typeof checks above would still pass if the embedder called a
    // dropped/stale fn; THIS is the exact failure mode the fix addresses.
    const injected = vi.fn(async (t: string) => [t.length, 7]);
    _setCurrentEmbedFnForTest(injected);
    const live = oss.EmbedderFactory.create('custom', {
      embed: () => { throw new Error('config.embed must NOT be used — mem0 strips it'); },
    }) as { embed: (t: string) => Promise<number[]>; embedBatch: (t: string[]) => Promise<number[][]> };
    await expect(live.embed('hi')).resolves.toEqual([2, 7]);
    expect(injected).toHaveBeenCalledWith('hi');
    await expect(live.embedBatch(['a', 'bbb'])).resolves.toEqual([[1, 7], [3, 7]]);
    _setCurrentEmbedFnForTest(null);

    // Non-custom providers still route through to the real factory (unknown
    // ones still throw — we didn't swallow the original behavior).
    expect(() => oss.EmbedderFactory.create('not-a-real-provider', {})).toThrow();
  });

  it('single-scope search forwards cancellation to query embeds and never starts an obsolete entity embed', async () => {
    const oss = await import('mem0ai/oss');
    patchEmbedderFactory(oss);
    const live = oss.EmbedderFactory.create('custom', {}) as {
      embed: (text: string) => Promise<number[]>;
    };
    let release!: () => void;
    let started!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const ready = new Promise<void>((resolve) => { started = resolve; });
    const injected = vi.fn(async (_text: string, _signal?: AbortSignal) => {
      started();
      await gate;
      return [0.5];
    });
    _setCurrentEmbedFnForTest(injected);
    const client = { search: async () => {
      await live.embed('query');
      await live.embed('entity');
      return { results: [] };
    } } as unknown as MemoryClient;
    const controller = new AbortController();
    const be = new Mem0Backend({ getClient: async () => client });
    const search = be.search('query', { scope: 'user', signal: controller.signal });
    const outcome = expect(search).rejects.toThrow('obsolete');
    await ready;
    const upstream = injected.mock.calls[0][1];
    controller.abort(new Error('obsolete'));
    release();
    try {
      await outcome;
      expect(upstream).toBe(controller.signal);
      expect(injected.mock.calls.map(([text]) => text)).toEqual(['query']);
    } finally {
      _setCurrentEmbedFnForTest(null);
    }
  });
});

describe('resolveExtractionLlmConfig — stale-key cascade (memory-backend-benchmark D-007)', async () => {
  const { resolveExtractionLlmConfig, _resetAnthropicKeyProbeCacheForTest } = await import('./mem0-client');

  it('prefers anthropic when the key probes usable', async () => {
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-good', openaiKey: 'sk-oai' },
      async () => true,
    );
    expect(cfg).toMatchObject({ provider: 'anthropic' });
    expect((cfg!.config as { apiKey: string }).apiKey).toBe('sk-ant-good');
  });

  it('falls back to openai when the anthropic key is auth-rejected', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-stale', openaiKey: 'sk-oai' },
      async () => false,
    );
    expect(cfg).toMatchObject({ provider: 'openai' });
    expect((cfg!.config as { model: string }).model).toBe('gpt-4o-mini');
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('falling back to OpenAI gpt-4o-mini for fact extraction'),
    );
  });

  it('keeps the dead anthropic key when nothing else exists (search/verbatim still work)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-stale', openaiKey: '' },
      async () => false,
    );
    expect(cfg).toMatchObject({ provider: 'anthropic' });
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('mem0 fact extraction WILL fail until the key is rotated'),
    );
  });

  it('openai-only and no-keys cases unchanged', async () => {
    const probe = vi.fn(async () => true);
    expect(
      await resolveExtractionLlmConfig({ anthropicKey: '', openaiKey: 'sk-oai' }, probe),
    ).toMatchObject({ provider: 'openai' });
    expect(probe).not.toHaveBeenCalled(); // no anthropic key → no probe
    expect(await resolveExtractionLlmConfig({ anthropicKey: '', openaiKey: '' }, probe)).toBeNull();
  });

  it('probe-cache hook clears without throwing', () => {
    _resetAnthropicKeyProbeCacheForTest();
  });
});

/**
 * EI-15687: getMemoryClient() used to have NO in-flight-build dedup, so
 * every caller racing a cold cache (e.g. injection.ts's Promise.all of 3
 * concurrent scope pulls per chat turn, or a fresh :3170 post-restart burst)
 * independently paid the FULL cold-build cost — a thundering herd that both
 * wasted work and produced the observed wildly-variable 2.7s-57.7s timings
 * (N-way contention on the same PG/network legs).
 *
 * We can't drive the full mem0ai-import path here (same constraint as above:
 * no import callback under vitest's module runner) — but `resolveEmbedder`
 * is the FIRST host call `buildClient()` makes, and a `{mode:'disabled'}`
 * result short-circuits before ever touching the dynamic import. That's
 * enough surface to prove the dedup: N concurrent getMemoryClient() calls
 * against a cold cache must invoke the host's resolveEmbedder exactly ONCE.
 */
describe('getMemoryClient — in-flight build de-dup (EI-15687)', async () => {
  const { configureMemory } = await import('./config');
  const { getMemoryClient } = await import('./mem0-client');

  it('shares ONE in-flight build across concurrent cold-cache callers', async () => {
    let resolveEmbedderCalls = 0;
    let releaseBuild!: () => void;
    const buildGate = new Promise<void>((r) => {
      releaseBuild = r;
    });
    configureMemory({
      getAdminUrl: () => 'postgres://unused/unused',
      getCredentials: async () => ({}),
      resolveEmbedder: async () => {
        resolveEmbedderCalls += 1;
        // Hold every concurrent caller here until they've all queued up
        // behind the SAME in-flight promise, then let the (single) real
        // build proceed to its 'disabled' short-circuit.
        await buildGate;
        return { mode: 'disabled' as const, reason: 'test' };
      },
      buildEmbedderForMode: async () => {
        throw new Error('must not be reached — resolveEmbedder short-circuits first');
      },
    });

    const calls = [getMemoryClient(), getMemoryClient(), getMemoryClient()];
    // Let the microtask queue settle so all 3 calls have observed the
    // in-flight promise (or started the build) before releasing it.
    await Promise.resolve();
    await Promise.resolve();
    releaseBuild();
    const results = await Promise.all(calls);

    expect(resolveEmbedderCalls).toBe(1);
    expect(results).toEqual([null, null, null]);
  });
});
