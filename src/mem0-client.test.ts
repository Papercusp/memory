import { describe, it, expect } from 'vitest';
import { patchEmbedderFactory } from './mem0-client';

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

    // Non-custom providers still route through to the real factory (unknown
    // ones still throw — we didn't swallow the original behavior).
    expect(() => oss.EmbedderFactory.create('not-a-real-provider', {})).toThrow();
  });
});
