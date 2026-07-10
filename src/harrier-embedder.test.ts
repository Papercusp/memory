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
import { describe, it, expect } from 'vitest';
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
