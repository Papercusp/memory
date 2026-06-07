import { describe, it, expect, vi } from 'vitest';
import {
  patchLlmFactory,
  patchEmbedderFactory,
  _setCurrentExtractionLlmForTest,
  _setCurrentEmbedFnForTest,
  resolveExtractionLlmConfig,
} from './mem0-client';
import {
  ExtractionAuthError,
  FallbackExtractionLlm,
  type ExtractionLlm,
  type ExtractionLlmMessage,
} from './extraction-llm';

// mem0's OSS telemetry reads this at module load — keep tests offline.
process.env.MEM0_TELEMETRY = 'false';

type OssModule = {
  LLMFactory: { create: (p: string, c: Record<string, unknown>) => unknown };
  Memory: new (config: Record<string, unknown>) => {
    add: (
      messages: string,
      config: Record<string, unknown>,
    ) => Promise<{ results: Array<{ id: string; memory: string; metadata?: { event?: string } }> }>;
    getAll: (opts: Record<string, unknown>) => Promise<{ results: Array<{ memory?: string }> }>;
  };
};

function fakeEmbed(dims = 8) {
  // Deterministic, text-sensitive — enough for the in-process store.
  return async (text: string): Promise<number[]> => {
    const v = Array.from({ length: dims }, (_, i) => {
      let h = i + 1;
      for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) % 997;
      return h / 997;
    });
    return v;
  };
}

/**
 * P-001 conformance suite (mem0-extraction-via-claude-session D-003):
 * pins mem0ai 3.x's custom-LLM interface so an upstream change fails
 * HERE, not silently in production. Mirrors the patchEmbedderFactory
 * regression test above it in spirit.
 */
describe('patchLlmFactory — mem0ai 3.x custom-LLM compatibility', () => {
  it('teaches mem0ai LLMFactory the custom provider (was unsupported)', async () => {
    const oss = (await import('mem0ai/oss')) as unknown as OssModule;

    // Upstream gap the patch exists for: mem0ai 3.x has no 'custom' LLM.
    expect(() => oss.LLMFactory.create('custom', {})).toThrow(/custom/i);

    patchLlmFactory(oss);

    const llm = oss.LLMFactory.create('custom', {}) as ExtractionLlm;
    expect(typeof llm.generateResponse).toBe('function');
    expect(typeof llm.generateChat).toBe('function');

    // The deepest invariant: calls route to the INJECTED instance (set by
    // tryLoad), NOT anything riding the config object — mem0's mergeConfig
    // strips non-scalar config fields during Zod validation.
    const injected: ExtractionLlm = {
      generateResponse: vi.fn(async () => '{"memory":[]}'),
      generateChat: vi.fn(async () => ({ content: 'hi', role: 'assistant' })),
    };
    _setCurrentExtractionLlmForTest(injected);
    const live = oss.LLMFactory.create('custom', {}) as ExtractionLlm;
    await expect(
      live.generateResponse([{ role: 'user', content: 'x' }], { type: 'json_object' }),
    ).resolves.toBe('{"memory":[]}');
    expect(injected.generateResponse).toHaveBeenCalledWith(
      [{ role: 'user', content: 'x' }],
      { type: 'json_object' },
      undefined,
    );
    await expect(live.generateChat([{ role: 'user', content: 'y' }])).resolves.toEqual({
      content: 'hi',
      role: 'assistant',
    });
    _setCurrentExtractionLlmForTest(null);

    // Non-custom providers still route through to the real factory.
    expect(() => oss.LLMFactory.create('not-a-real-provider', {})).toThrow();
  });

  it('drives a REAL mem0 Memory end-to-end through a custom extraction LLM', async () => {
    const oss = (await import('mem0ai/oss')) as unknown as OssModule;
    patchLlmFactory(oss);
    patchEmbedderFactory(oss as never);

    const calls: Array<{ messages: ExtractionLlmMessage[]; responseFormat?: { type: string } }> = [];
    const fact = 'User prefers tabs over spaces in the papercusp repo';
    const fakeLlm: ExtractionLlm = {
      generateResponse: async (messages, responseFormat) => {
        calls.push({ messages: messages as ExtractionLlmMessage[], responseFormat });
        // The AdditiveExtractionSchema shape mem0 3.x parses.
        return JSON.stringify({ memory: [{ id: 'new-1', text: fact }] });
      },
      generateChat: async () => ({ content: '', role: 'assistant' }),
    };
    _setCurrentExtractionLlmForTest(fakeLlm);
    _setCurrentEmbedFnForTest(fakeEmbed());

    try {
      const memory = new oss.Memory({
        embedder: { provider: 'custom', config: {} },
        // dbPath ':memory:' keeps the test hermetic — mem0's 'memory'
        // provider otherwise persists to ~/.mem0/vector_store.db, and a
        // prior run's row makes the hash-dedup skip this ADD.
        vectorStore: {
          provider: 'memory',
          config: { collectionName: 'conformance', dimension: 8, dbPath: ':memory:' },
        },
        llm: { provider: 'custom', config: {} },
        disableHistory: true,
      });

      const res = await memory.add('btw I prefer tabs over spaces', { userId: 'conformance' });

      // The extraction call contract: (system+user messages, json_object).
      expect(calls.length).toBeGreaterThanOrEqual(1);
      const extraction = calls[0];
      expect(extraction.responseFormat).toEqual({ type: 'json_object' });
      expect(extraction.messages[0]?.role).toBe('system');
      expect(extraction.messages[1]?.role).toBe('user');
      expect(extraction.messages[1]?.content).toContain('tabs over spaces');

      // The returned JSON string drove a real ADD.
      expect(res.results.some((r) => r.memory === fact)).toBe(true);

      // And it landed in the store.
      const all = await memory.getAll({ filters: { user_id: 'conformance' } });
      expect(all.results.some((r) => r.memory === fact)).toBe(true);
    } finally {
      _setCurrentExtractionLlmForTest(null);
      _setCurrentEmbedFnForTest(null);
    }
  });
});

describe('resolveExtractionLlmConfig — session rung #1 (D-002)', () => {
  const sessionLlm: ExtractionLlm = {
    generateResponse: async () => '{}',
    generateChat: async () => ({ content: '', role: 'assistant' }),
  };

  it('session adapter present → custom provider, key probe never runs', async () => {
    const probe = vi.fn(async () => true);
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-good', openaiKey: 'sk-oai' },
      probe,
      sessionLlm,
    );
    expect(cfg).toMatchObject({ provider: 'custom' });
    expect(probe).not.toHaveBeenCalled();
  });

  it('session adapter absent → key rungs unchanged', async () => {
    const cfg = await resolveExtractionLlmConfig(
      { anthropicKey: 'sk-ant-good', openaiKey: 'sk-oai' },
      async () => true,
      null,
    );
    expect(cfg).toMatchObject({ provider: 'anthropic' });
  });
});

describe('FallbackExtractionLlm — loud demotion cascade (D-004/D-005)', () => {
  const msg: ExtractionLlmMessage[] = [{ role: 'user', content: 'extract' }];

  function okLlm(tag: string): ExtractionLlm {
    return {
      generateResponse: vi.fn(async () => `${tag}-response`),
      generateChat: vi.fn(async () => ({ content: `${tag}-chat`, role: 'assistant' })),
    };
  }

  it('healthy primary serves; fallback never built', async () => {
    const primary = okLlm('primary');
    const buildFallback = vi.fn(async () => okLlm('fallback'));
    const warn = vi.fn();
    const llm = new FallbackExtractionLlm(primary, buildFallback, { warn });
    await expect(llm.generateResponse(msg)).resolves.toBe('primary-response');
    expect(buildFallback).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(llm.isDemoted).toBe(false);
  });

  it('ExtractionAuthError → warn + STICKY demote + fallback serves this and later calls', async () => {
    const primary: ExtractionLlm = {
      generateResponse: vi.fn(async () => {
        throw new ExtractionAuthError('401 after refresh+retry');
      }),
      generateChat: vi.fn(async () => ({ content: '', role: 'assistant' })),
    };
    const fallback = okLlm('fallback');
    const warn = vi.fn();
    const llm = new FallbackExtractionLlm(primary, async () => fallback, { warn });

    await expect(llm.generateResponse(msg)).resolves.toBe('fallback-response');
    expect(llm.isDemoted).toBe(true);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('demoting'));

    // Later calls never touch the primary again (process-lifetime demotion).
    await expect(llm.generateResponse(msg)).resolves.toBe('fallback-response');
    await expect(llm.generateChat(msg)).resolves.toMatchObject({ content: 'fallback-chat' });
    expect(primary.generateResponse).toHaveBeenCalledTimes(1);
  });

  it('non-auth failure → fallback serves THIS call; primary retried next call', async () => {
    let failures = 0;
    const primary: ExtractionLlm = {
      generateResponse: vi.fn(async () => {
        if (failures++ === 0) throw new Error('malformed JSON after repair retry');
        return 'primary-recovered';
      }),
      generateChat: vi.fn(async () => ({ content: '', role: 'assistant' })),
    };
    const fallback = okLlm('fallback');
    const warn = vi.fn();
    const llm = new FallbackExtractionLlm(primary, async () => fallback, { warn });

    await expect(llm.generateResponse(msg)).resolves.toBe('fallback-response');
    expect(llm.isDemoted).toBe(false);
    expect(warn).toHaveBeenCalledTimes(1);

    // Transient — the primary gets another shot and recovers.
    await expect(llm.generateResponse(msg)).resolves.toBe('primary-recovered');
  });

  it('no fallback available → rethrows AND the warning fired (never silent)', async () => {
    const primary: ExtractionLlm = {
      generateResponse: vi.fn(async () => {
        throw new ExtractionAuthError('dead token');
      }),
      generateChat: vi.fn(async () => ({ content: '', role: 'assistant' })),
    };
    const warn = vi.fn();
    const llm = new FallbackExtractionLlm(primary, async () => null, { warn });

    await expect(llm.generateResponse(msg)).rejects.toThrow('dead token');
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('demoting'));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('LOST'));
  });

  it('fallback builder throwing is contained (warn + original error rethrown)', async () => {
    const primary: ExtractionLlm = {
      generateResponse: vi.fn(async () => {
        throw new Error('primary down');
      }),
      generateChat: vi.fn(async () => ({ content: '', role: 'assistant' })),
    };
    const warn = vi.fn();
    const llm = new FallbackExtractionLlm(
      primary,
      async () => {
        throw new Error('factory exploded');
      },
      { warn },
    );
    await expect(llm.generateResponse(msg)).rejects.toThrow('primary down');
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('factory exploded'));
  });
});
