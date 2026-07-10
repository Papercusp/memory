/**
 * EmbeddingGemma-300m embedder pure-function tests (the default local embedder,
 * owner ask 2026-07-10). These cover the two decisions that make Gemma reuse the
 * existing 384-dim infrastructure correctly WITHOUT loading the ~300M model:
 *   - MRL truncation 768→384 + L2-renorm (mrlTruncate)
 *   - the asymmetric task prompts (gemmaPrompt)
 * The model-loading path (buildGemmaEmbedder) is exercised by the live memory
 * suite; here we pin the math + the prompt contract.
 */
import { describe, it, expect } from 'vitest';
import { mrlTruncate, gemmaPrompt, GEMMA_TARGET_DIMS, GEMMA_MODEL } from './gemma-embedder';

describe('mrlTruncate (MRL 768→384 + renormalize)', () => {
  it('truncates a longer vector to the target length', () => {
    const v = Array.from({ length: 768 }, (_, i) => (i % 7) - 3 || 0.5);
    expect(mrlTruncate(v, GEMMA_TARGET_DIMS)).toHaveLength(384);
  });

  it('defaults to 384 dims', () => {
    const v = Array.from({ length: 768 }, () => 1);
    expect(mrlTruncate(v)).toHaveLength(384);
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

describe('constants', () => {
  it('targets 384 dims (reuses the existing vector(384) columns)', () => {
    expect(GEMMA_TARGET_DIMS).toBe(384);
  });

  it('points at the Transformers.js/ONNX EmbeddingGemma-300m build', () => {
    expect(GEMMA_MODEL).toBe('onnx-community/embeddinggemma-300m-ONNX');
  });
});
