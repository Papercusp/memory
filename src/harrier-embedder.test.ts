/**
 * Harrier-oss-0.6b embedder pure-function tests (P-013). These pin the three
 * contracts that make harrier land in its OWN space correctly WITHOUT loading
 * the ~0.6B model:
 *   - last-token pooling over the flattened [seq, dims] per-token output
 *   - the asymmetric instruct-query / raw-document prompts
 *   - native-1024 vs exploratory truncated-384 widths (via mrlTruncate)
 * The model-loading path (buildHarrierEmbedder) is exercised by the P-006
 * bake-off smoke; here we pin the math + the prompt contract.
 */
import { describe, it, expect } from 'vitest';
import {
  harrierPrompt,
  lastTokenPool,
  HARRIER_MODEL,
  HARRIER_NATIVE_DIMS,
  HARRIER_QUERY_TASK,
} from './harrier-embedder';
import { mrlTruncate } from './gemma-embedder';

describe('lastTokenPool (per-token [seq, dims] → final token slice)', () => {
  it('returns the LAST dims-wide slice, not the first', () => {
    // 3 tokens × 4 dims; each token's vector is constant-valued by index.
    const flat = [...Array(4).fill(1), ...Array(4).fill(2), ...Array(4).fill(3)];
    expect(lastTokenPool(flat, 4)).toEqual([3, 3, 3, 3]);
  });

  it('a single-token sequence is returned whole', () => {
    const flat = [1, 2, 3, 4];
    expect(lastTokenPool(flat, 4)).toEqual([1, 2, 3, 4]);
  });

  it('defaults dims to the native 1024', () => {
    const flat = Array.from({ length: HARRIER_NATIVE_DIMS * 2 }, (_, i) => i);
    expect(lastTokenPool(flat)).toHaveLength(1024);
    expect(lastTokenPool(flat)[0]).toBe(HARRIER_NATIVE_DIMS); // second token's first cell
  });

  it('throws loudly on a length that is not a whole number of tokens', () => {
    expect(() => lastTokenPool([1, 2, 3, 4, 5], 4)).toThrow(/not a multiple/);
    expect(() => lastTokenPool([1, 2], 4)).toThrow(/not a multiple/);
  });
});

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
});

describe('constants', () => {
  it('points at the Transformers.js/ONNX harrier-oss-v1-0.6b build', () => {
    expect(HARRIER_MODEL).toBe('onnx-community/harrier-oss-v1-0.6b-ONNX');
  });

  it('is natively 1024-dim (own vector(1024) table when adopted — P-014)', () => {
    expect(HARRIER_NATIVE_DIMS).toBe(1024);
  });
});
