import { describe, it, expect, vi } from 'vitest';
import { patchEmbedderFactory, _setCurrentEmbedFnForTest } from './mem0-client';

/**
 * Regression guard for the mem0ai 3.x compatibility fix (commit 879fc733).
 *
 * Before the fix, getMemoryClient() ALWAYS returned null: mem0ai 3.0.3's
 * EmbedderFactory rejects the 'custom' provider the store uses ("Unsupported
 * embedder provider: custom") and only VectorStoreFactory was patched, so the
 * Memory constructor threw and was swallowed — every memory:* tool silently
 * returned mem0_unavailable.
 *
 * We can't drive getMemoryClient() end-to-end here: the store loads mem0ai via
 * a `new Function('return import(s)')` trick (to dodge bundler static analysis)
 * which has no import callback under vitest's module runner. So we assert the
 * fix at its seam — patchEmbedderFactory must make mem0ai's own EmbedderFactory
 * accept the 'custom' provider.
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
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-stale', openaiKey: 'sk-oai' },
      async () => false,
    );
    expect(cfg).toMatchObject({ provider: 'openai' });
    expect((cfg!.config as { model: string }).model).toBe('gpt-4o-mini');
  });

  it('keeps the dead anthropic key when nothing else exists (search/verbatim still work)', async () => {
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-stale', openaiKey: '' },
      async () => false,
    );
    expect(cfg).toMatchObject({ provider: 'anthropic' });
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
