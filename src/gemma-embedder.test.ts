/**
 * EmbeddingGemma-300m embedder pure-function tests (the default local embedder,
 * owner ask 2026-07-10). These cover the two decisions that make Gemma reuse the
 * existing 384-dim infrastructure correctly WITHOUT loading the ~300M model:
 *   - MRL truncation 768→384 + L2-renorm (mrlTruncate)
 *   - the asymmetric task prompts (gemmaPrompt)
 * The model-loading path (buildGemmaEmbedder) is exercised by the live memory
 * suite; here we pin the math + the prompt contract.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mrlTruncate, gemmaPrompt, GEMMA_TARGET_DIMS, GEMMA_MODEL } from './gemma-embedder';

describe('mrlTruncate (MRL slice + renormalize; the target is now the NATIVE 768)', () => {
  it('truncates a longer vector to the target length', () => {
    const v = Array.from({ length: 768 }, (_, i) => (i % 7) - 3 || 0.5);
    expect(mrlTruncate(v, GEMMA_TARGET_DIMS)).toHaveLength(768);
  });

  it('defaults to the declared target width', () => {
    const v = Array.from({ length: 768 }, () => 1);
    // 768 (native) since D-005 — so the default call is a no-op slice that
    // still renormalizes, which is exactly what the storage path needs given
    // the pipeline is invoked with normalize:false.
    expect(mrlTruncate(v)).toHaveLength(768);
  });

  it('L2-normalizes the truncated slice to unit length', () => {
    const v = Array.from({ length: 768 }, (_, i) => i + 1); // non-trivial magnitudes
    const out = mrlTruncate(v, 384);
    const norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it('normalizes the PREFIX, not the full vector (prefix of a unit vector is not unit)', () => {
    // A vector already unit-norm over 768 dims: its first-384 slice must be
    // RE-normalized, i.e. its own norm becomes 1 (not the ~0.7 a raw prefix has).
    const raw = Array.from({ length: 768 }, () => 1);
    const fullNorm = Math.sqrt(raw.reduce((s, x) => s + x * x, 0));
    const unit768 = raw.map((x) => x / fullNorm); // unit over 768
    const out = mrlTruncate(unit768, 384);
    const norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it('does not divide by zero on a degenerate zero vector', () => {
    const out = mrlTruncate(new Array(768).fill(0), 384);
    expect(out).toHaveLength(384);
    expect(out.every((x) => x === 0)).toBe(true);
  });

  it('leaves a vector already at/under the target untouched in length', () => {
    expect(mrlTruncate([3, 4], 384)).toHaveLength(2); // shorter than target → sliced-noop
  });

  /**
   * TRUNCATION IS TRANSITIVE — the property the P-002 MRL sweep is built on.
   *
   * Scoring gemma@768/512/384/256 as four independent legs would cost four
   * ~16-minute corpus passes over prose-corpus.v1. It costs ONE, because
   * re-truncating an already-truncated vector equals truncating the raw
   * vector to the same width: mrlTruncate renormalizes, and normalization
   * cancels any positive scalar, so the intermediate renorm leaves no trace.
   *
   * If this ever stopped holding, the sweep would silently be comparing
   * doubly-normalized vectors against singly-normalized ones and attributing
   * the difference to DIMENSION — a wrong verdict that no metric would flag.
   * Hence a test rather than a comment.
   */
  it('is TRANSITIVE: truncating via an intermediate width equals truncating directly', () => {
    // Deliberately not smooth/monotonic — a decaying sinusoid gives each
    // prefix a genuinely different norm, so an accidental no-op would show.
    const raw = Array.from({ length: 768 }, (_, i) => Math.sin(i * 0.7) * Math.exp(-i / 400) + 0.01 * i);

    for (const [via, target] of [
      [768, 384],
      [768, 512],
      [768, 256],
      [512, 256],
    ] as const) {
      const direct = mrlTruncate(raw, target);
      const derived = mrlTruncate(mrlTruncate(raw, via), target);
      expect(derived).toHaveLength(target);
      for (let i = 0; i < target; i++) expect(derived[i]).toBeCloseTo(direct[i], 10);
      // And the pair is cosine-identical, which is what retrieval actually sees.
      const cos = direct.reduce((s, x, i) => s + x * derived[i], 0);
      expect(cos).toBeCloseTo(1, 10);
    }
  });
});

describe('gemmaPrompt (asymmetric task prompts)', () => {
  it('applies the query prompt for search queries', () => {
    expect(gemmaPrompt('query', 'red shoes')).toBe('task: search result | query: red shoes');
  });

  it('applies the document prompt for stored text', () => {
    expect(gemmaPrompt('document', 'hello world')).toBe('title: none | text: hello world');
  });

  it('query and document prompts differ (dual-encoder)', () => {
    expect(gemmaPrompt('query', 'x')).not.toBe(gemmaPrompt('document', 'x'));
  });
});

describe('buildGemmaEmbedder — worker-only, non-sticky (EI-16184, EI-28961860650735669)', () => {
  beforeEach(() => {
    vi.resetModules();
  });
  afterEach(() => {
    vi.doUnmock('./local-embedder-worker');
    vi.doUnmock('./dynamic-import');
  });

  // dynamicImport is the first step of any inline (main-thread) model load.
  // It throws here, so a reintroduced inline path can neither download a real
  // model nor pass: the call would reject with this error, not the worker's.
  const stubInlineModel = () => {
    const dynamicImport = vi.fn(async () => {
      throw new Error('inline-path-entered');
    });
    vi.doMock('./dynamic-import', () => ({ dynamicImport }));
    return { dynamicImport };
  };

  const mockWorker = (embedViaWorker: unknown, disabled = false) => {
    const warnEmbedFallback = vi.fn();
    vi.doMock('./local-embedder-worker', () => ({
      embedViaWorker,
      embedWorkerDisabledError: (model: string) => new Error(`worker disabled: ${model}`),
      getWorkerState: () => ({ alive: !disabled, disabled, pendingCount: 0 }),
      warnEmbedFallback,
      isEmbedWorkerShutdownError: () => false,
    }));
    return { warnEmbedFallback };
  };

  it('rejects a failed call with the worker error (no inline retry), then retries the worker on the NEXT call', async () => {
    const inline = stubInlineModel();
    const failure = new Error('transient worker hiccup');
    const embedViaWorker = vi.fn().mockRejectedValueOnce(failure).mockResolvedValueOnce(new Array(768).fill(0.1));
    const { warnEmbedFallback } = mockWorker(embedViaWorker);

    const { buildGemmaEmbedder } = await import('./gemma-embedder');
    const embed = buildGemmaEmbedder({ kind: 'query' });

    await expect(embed('hello')).rejects.toBe(failure);
    expect(warnEmbedFallback).toHaveBeenCalledWith('gemma', failure);
    expect(inline.dynamicImport).not.toHaveBeenCalled();

    // Call 2 MUST re-attempt the worker — the old sticky `workerDisabled`
    // flag skipped it forever after one failure.
    const result = await embed('hello again');
    expect(embedViaWorker).toHaveBeenCalledTimes(2);
    expect(result).toHaveLength(GEMMA_TARGET_DIMS);
    expect(warnEmbedFallback).toHaveBeenCalledTimes(1);
    expect(inline.dynamicImport).not.toHaveBeenCalled();
  });

  it('rejects without touching the worker or the main thread once the module reports permanent unavailability', async () => {
    const inline = stubInlineModel();
    const embedViaWorker = vi.fn().mockResolvedValue(new Array(768).fill(0.1));
    const { warnEmbedFallback } = mockWorker(embedViaWorker, true);

    const { buildGemmaEmbedder } = await import('./gemma-embedder');
    const embed = buildGemmaEmbedder({ kind: 'query' });
    await expect(embed('hello')).rejects.toThrow('worker disabled: gemma');

    expect(embedViaWorker).not.toHaveBeenCalled();
    expect(warnEmbedFallback).toHaveBeenCalledWith('gemma', 'worker-disabled');
    expect(inline.dynamicImport).not.toHaveBeenCalled();
  });
});

describe('constants', () => {
  it('targets the NATIVE 768 — no truncation, so no untrained cut is possible (D-005)', () => {
    expect(GEMMA_TARGET_DIMS).toBe(768);
  });

  it('points at the Transformers.js/ONNX EmbeddingGemma-300m build', () => {
    expect(GEMMA_MODEL).toBe('onnx-community/embeddinggemma-300m-ONNX');
  });
});
