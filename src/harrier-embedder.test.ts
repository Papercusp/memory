/**
 * Harrier-oss-0.6b embedder pure-function tests (P-013). These pin the
 * contracts that make harrier land in its OWN space correctly WITHOUT loading
 * the ~0.6B model:
 *   - the asymmetric instruct-query / raw-document prompts
 *   - native-1024 vs exploratory truncated-384 widths (via mrlTruncate)
 *   - the graph-output contract (pooling+normalize are IN the ONNX graph;
 *     the sole output is 'sentence_embedding')
 * The model-loading path (buildHarrierEmbedder) is exercised by the P-006
 * bake-off smoke; here we pin the math + the prompt contract.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  harrierPrompt,
  HARRIER_MODEL,
  HARRIER_NATIVE_DIMS,
  HARRIER_GRAPH_OUTPUT,
  HARRIER_QUERY_TASK,
} from './harrier-embedder';
import { mrlTruncate } from './gemma-embedder';

describe('harrierPrompt (asymmetric instruct prompts)', () => {
  it('queries get the Instruct+Query prefix with the web_search_query task', () => {
    expect(harrierPrompt('query', 'red shoes')).toBe(
      `Instruct: ${HARRIER_QUERY_TASK}\nQuery: red shoes`,
    );
  });

  it('documents are embedded RAW — no prefix', () => {
    expect(harrierPrompt('document', 'hello world')).toBe('hello world');
  });

  it('query and document prompts differ (dual-encoder)', () => {
    expect(harrierPrompt('query', 'x')).not.toBe(harrierPrompt('document', 'x'));
  });
});

describe('width handling (native 1024 / exploratory 384)', () => {
  it('mrlTruncate at native width is a plain L2 norm (no truncation)', () => {
    const v = Array.from({ length: 1024 }, (_, i) => i + 1);
    const out = mrlTruncate(v, HARRIER_NATIVE_DIMS);
    expect(out).toHaveLength(1024);
    const norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it('truncated-384 is truncate-THEN-normalize (unit norm over the slice)', () => {
    const v = Array.from({ length: 1024 }, (_, i) => i + 1);
    const out = mrlTruncate(v, 384);
    expect(out).toHaveLength(384);
    const norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it('truncating an already-normalized vector equals truncating the raw one (scale invariance)', () => {
    // The graph L2-normalizes sentence_embedding; truncate-then-normalize on
    // that must equal the canonical MRL procedure on the raw vector.
    const raw = Array.from({ length: 1024 }, (_, i) => Math.sin(i + 1) * (i + 1));
    const unit = mrlTruncate(raw, 1024); // plain L2 norm
    const a = mrlTruncate(raw, 384);
    const b = mrlTruncate(unit, 384);
    for (let i = 0; i < 384; i++) expect(b[i]).toBeCloseTo(a[i], 10);
  });
});

describe('buildHarrierEmbedder — worker-only, non-sticky (EI-16184, EI-28961860650735669)', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  /**
   * dynamicImport is the first step of any inline (main-thread) model load. It
   * throws here, so a reintroduced inline path can neither load the ~0.6B model
   * (a real load took 5-7s and once red-pinned the gate on candidate a94547c8)
   * nor pass: the call would reject with this error, not the worker's. It must
   * be `./dynamic-import`, not '@huggingface/transformers': dynamicImport()
   * resolves the package through `new Function('s','return import(s)')`, which
   * vitest's module interception cannot see.
   */
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
    const embedViaWorker = vi
      .fn()
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce(new Array(HARRIER_NATIVE_DIMS).fill(0.1));
    const { warnEmbedFallback } = mockWorker(embedViaWorker);

    const { buildHarrierEmbedder } = await import('./harrier-embedder');
    const embed = buildHarrierEmbedder({ kind: 'query' });

    await expect(embed('hello')).rejects.toBe(failure);
    const result = await embed('hello again');
    expect(embedViaWorker).toHaveBeenCalledTimes(2);
    expect(result).toHaveLength(HARRIER_NATIVE_DIMS);
    expect(warnEmbedFallback).toHaveBeenCalledTimes(1);
    expect(inline.dynamicImport).not.toHaveBeenCalled();
  });

  it('rejects without touching the worker or the main thread once the module reports permanent unavailability', async () => {
    const inline = stubInlineModel();
    const embedViaWorker = vi.fn().mockResolvedValue(new Array(HARRIER_NATIVE_DIMS).fill(0.1));
    const { warnEmbedFallback } = mockWorker(embedViaWorker, true);

    const { buildHarrierEmbedder } = await import('./harrier-embedder');
    const embed = buildHarrierEmbedder({ kind: 'query' });
    await expect(embed('hello')).rejects.toThrow('worker disabled: harrier');

    expect(embedViaWorker).not.toHaveBeenCalled();
    expect(warnEmbedFallback).toHaveBeenCalledWith('harrier', 'worker-disabled');
    expect(inline.dynamicImport).not.toHaveBeenCalled();
  });
});

describe('constants', () => {
  it('points at the Transformers.js/ONNX harrier-oss-v1-0.6b build', () => {
    expect(HARRIER_MODEL).toBe('onnx-community/harrier-oss-v1-0.6b-ONNX');
  });

  it('is natively 1024-dim (own vector(1024) table when adopted — P-014)', () => {
    expect(HARRIER_NATIVE_DIMS).toBe(1024);
  });

  it('reads the pre-pooled graph output (no JS-side pooling exists for this export)', () => {
    expect(HARRIER_GRAPH_OUTPUT).toBe('sentence_embedding');
  });
});
