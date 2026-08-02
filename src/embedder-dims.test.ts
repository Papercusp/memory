/**
 * The recurrence guard for EI-19301722864393687 — the prose surfaces' untrained
 * MRL-384 cut.
 *
 * The original bug was not a wrong line of code; it was an ASSERTION NOBODY
 * MADE. A docblock claimed any prefix of an MRL model was "a valid lower-dim
 * embedding", the target dim was a bare `384` next to the truncation call, and
 * nothing anywhere disagreed — so the cut shipped, degraded quality silently
 * (an untrained cut still yields a usable vector), and no test could have
 * caught it. This file is the disagreement.
 *
 * Three properties, and all three are load-bearing:
 *
 *   1. every declared spec is sound (the guard passes today);
 *   2. the guard can actually FAIL — each rule is shown rejecting a spec that
 *      violates it. A guard whose failure path is never exercised is
 *      indistinguishable from a guard that always returns "fine", and that is
 *      precisely the failure mode being guarded against;
 *   3. the constants the builders truncate to are the constants declared here,
 *      so a sound declaration cannot describe code that does something else.
 */

import { describe, expect, it } from 'vitest';
import {
  EMBEDDER_DIM_SPECS,
  CANDIDATE_DIM_SPECS,
  dimSpecFor,
  isTrainedDim,
  validateEmbedderDimSpec,
  type EmbedderDimSpec,
  type EmbedderMode,
} from './embedder-dims';
import { GEMMA_TARGET_DIMS } from './gemma-embedder';
import { HARRIER_NATIVE_DIMS } from './harrier-embedder';
import { graniteNativeDims, GRANITE_SPEC_KEYS } from './granite-embedder';
import { QWEN3_NATIVE_DIMS } from './qwen3-embedder';

const MODES: EmbedderMode[] = ['openai', 'local', 'gemma', 'harrier'];
const CANDIDATES = ['granite97', 'granite311', 'qwen3'] as const;

/** A sound baseline to mutate one field at a time. */
const OK: EmbedderDimSpec = {
  model: 'test/model',
  nativeDims: 768,
  mrl: 'discrete',
  trainedDims: [768, 512, 256],
  targetDims: 512,
};

describe('every declared embedder spec is sound', () => {
  it('covers every embedder mode — a new model must declare its dims', () => {
    // The Record<EmbedderMode, …> type makes an omission a compile error; this
    // asserts the runtime table matches, so the two can never disagree.
    expect(Object.keys(EMBEDDER_DIM_SPECS).sort()).toEqual([...MODES].sort());
  });

  it.each(MODES)('%s declares a valid dim spec', (mode) => {
    expect(validateEmbedderDimSpec(mode, EMBEDDER_DIM_SPECS[mode])).toEqual([]);
  });

  it.each(CANDIDATES)('bake-off candidate %s declares a valid dim spec', (key) => {
    // Candidates are held to the SAME validator as shipped modes. A model
    // measured at a width nobody declared is exactly how the original bug got
    // in — being "only a candidate" is not an exemption, it is the moment the
    // declaration is cheapest to make.
    expect(validateEmbedderDimSpec(key, CANDIDATE_DIM_SPECS[key])).toEqual([]);
  });

  it('keeps candidates OUT of the production mode table', () => {
    // A candidate silently becoming a mode would skip the resolver/table/
    // migration work a real mode needs. The separation is the point.
    for (const key of CANDIDATES) expect(Object.keys(EMBEDDER_DIM_SPECS)).not.toContain(key);
  });
});

describe('every candidate can produce 384 without an untrained cut', () => {
  // The prose surfaces store vector(384), so a candidate is only CHEAP to
  // adopt if it emits 384 without a hand-rolled prefix. That is the property
  // asserted here, and it is a claim about MIGRATION COST, not about quality.
  //
  // ⚠ Do not extend these into a quality ranking. D-003 measured the
  // temptation false: gemma@384 (untrained) scores ABOVE the trained 512/256
  // interpolation, and the TRAINED 256 is dramatically WORSE than it
  // (-.0889 MRR). Trainedness predicted quality in neither direction, and
  // "prefer a trained dim" — D-001's original advice — would have selected
  // 256 and degraded retrieval. Which model ships is decided by the prose
  // gold set (P-003), never by this file.

  it('gemma@384 is an untrained cut, and only passes because it SAYS so', () => {
    expect(isTrainedDim(EMBEDDER_DIM_SPECS.gemma, 384)).toBe(false);
    expect(EMBEDDER_DIM_SPECS.gemma.untrainedCut).toBeDefined();
    // The acknowledgement is what makes the tradeoff reviewable — it is not a
    // verdict that the cut is bad. D-003 found the cut itself blameless; the
    // loss it flagged was ordinary width loss.
  });

  it('granite97 is natively 384, so no truncation happens at all', () => {
    expect(CANDIDATE_DIM_SPECS.granite97.nativeDims).toBe(384);
    expect(isTrainedDim(CANDIDATE_DIM_SPECS.granite97, 384)).toBe(true);
    expect(CANDIDATE_DIM_SPECS.granite97.untrainedCut).toBeUndefined();
  });

  it('granite311 declares 384 among its published nesting points', () => {
    expect(isTrainedDim(CANDIDATE_DIM_SPECS.granite311, 384)).toBe(true);
    // No acknowledgement needed, and the validator's stale-ack rule means one
    // cannot be added later "to be safe" without failing the suite.
    expect(CANDIDATE_DIM_SPECS.granite311.untrainedCut).toBeUndefined();
    // Trained at 384 but NOT at an arbitrary neighbouring width — otherwise
    // this assertion would pass for a spec that simply claimed everything.
    expect(isTrainedDim(CANDIDATE_DIM_SPECS.granite311, 400)).toBe(false);
  });

  it('qwen3 reduces continuously, so 384 is supported', () => {
    expect(isTrainedDim(CANDIDATE_DIM_SPECS.qwen3, 384)).toBe(true);
    expect(CANDIDATE_DIM_SPECS.qwen3.untrainedCut).toBeUndefined();
  });
});

describe('dimSpecFor spans both tables', () => {
  it('resolves production modes and candidates alike', () => {
    expect(dimSpecFor('gemma')).toBe(EMBEDDER_DIM_SPECS.gemma);
    expect(dimSpecFor('granite311')).toBe(CANDIDATE_DIM_SPECS.granite311);
  });

  it('returns undefined for an unknown family rather than throwing', () => {
    // The eval CLI turns this into a per-leg refusal; throwing would abort a
    // multi-leg sweep over a single typo.
    expect(dimSpecFor('not-a-model')).toBeUndefined();
  });
});

describe('the guard FAILS on the mistakes it exists to catch', () => {
  it('rejects an untrained target dim with no acknowledgement — THE original bug', () => {
    // gemma@384 exactly: an intermediate prefix of a discrete-MRL model.
    const problems = validateEmbedderDimSpec('gemma-like', { ...OK, targetDims: 384 });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain('NOT a trained dim');
    expect(problems[0]).toContain('untrainedCut');
  });

  it('ACCEPTS that same cut once it is acknowledged in writing', () => {
    expect(
      validateEmbedderDimSpec('gemma-like', {
        ...OK,
        targetDims: 384,
        untrainedCut: { reason: 'fits vector(384)', trackedBy: 'some-plan' },
      }),
    ).toEqual([]);
  });

  it('rejects an acknowledgement that says nothing — you cannot silence it with a stub', () => {
    const problems = validateEmbedderDimSpec('gemma-like', {
      ...OK,
      targetDims: 384,
      untrainedCut: { reason: '  ', trackedBy: '' },
    });
    expect(problems).toEqual([
      expect.stringContaining('reason must not be empty'),
      expect.stringContaining('trackedBy must not be empty'),
    ]);
  });

  it('rejects a STALE acknowledgement on a now-trained dim', () => {
    // Left behind after someone fixes the dim, a stale ack silently
    // pre-authorizes the NEXT untrained cut — so it is itself a violation.
    const problems = validateEmbedderDimSpec('gemma-like', {
      ...OK,
      targetDims: 512,
      untrainedCut: { reason: 'no longer applies', trackedBy: 'old-plan' },
    });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain('stale');
  });

  it('rejects truncating UP, and a trainedDims set omitting the native width', () => {
    expect(validateEmbedderDimSpec('bad', { ...OK, targetDims: 1024 })[0]).toContain('cannot be truncated UP');
    expect(validateEmbedderDimSpec('bad', { ...OK, trainedDims: [512, 256] })[0]).toContain('omits nativeDims');
  });

  it('rejects a no-MRL model claiming extra trained dims', () => {
    const problems = validateEmbedderDimSpec('bad', {
      ...OK,
      mrl: 'none',
      trainedDims: [768, 512],
      targetDims: 768,
    });
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain("mrl:'none'");
  });

  it('reports ALL violations at once, not just the first', () => {
    // One re-run per problem is how a multi-problem new embedder turns into
    // four sequential CI failures.
    expect(
      validateEmbedderDimSpec('bad', { ...OK, nativeDims: -1, targetDims: -5, trainedDims: [512] }).length,
    ).toBeGreaterThan(2);
  });
});

describe('isTrainedDim reflects how each real model actually reduces', () => {
  it('gemma: the trained MRL points are supported, 384 is not', () => {
    const gemma = EMBEDDER_DIM_SPECS.gemma;
    for (const trained of [768, 512, 256, 128]) expect(isTrainedDim(gemma, trained)).toBe(true);
    // The configured target is knowingly untrained — acknowledged, not hidden.
    expect(isTrainedDim(gemma, 384)).toBe(false);
    expect(gemma.untrainedCut?.trackedBy).toContain('prose-embedding-384-untrained-mrl-fix');
  });

  it('harrier: publishes no MRL, so its exploratory 384 variant IS an untrained cut', () => {
    const harrier = EMBEDDER_DIM_SPECS.harrier;
    expect(isTrainedDim(harrier, 1024)).toBe(true);
    expect(isTrainedDim(harrier, 384)).toBe(false);
  });

  it('openai: reduction is first-class, so 384 is genuinely supported', () => {
    // The contrast case. A guard that flagged every non-native width would be
    // noise, and would have been switched off long before it caught anything.
    const openai = EMBEDDER_DIM_SPECS.openai;
    expect(isTrainedDim(openai, 384)).toBe(true);
    expect(isTrainedDim(openai, 1536)).toBe(true);
    expect(isTrainedDim(openai, 2048)).toBe(false); // still cannot exceed native
  });

  it('local: natively 384, so it never truncates at all', () => {
    expect(isTrainedDim(EMBEDDER_DIM_SPECS.local, 384)).toBe(true);
  });
});

describe('the declared dims are the dims the builders actually use', () => {
  it('gemma truncates to its declared target', () => {
    // Derived, not restated — a docblock that merely AGREED with the code is
    // what shipped the bug. This asserts they are the same constant.
    expect(GEMMA_TARGET_DIMS).toBe(EMBEDDER_DIM_SPECS.gemma.targetDims);
    expect(GEMMA_TARGET_DIMS).toBe(768);
  });

  it('harrier stores at its declared native width', () => {
    expect(HARRIER_NATIVE_DIMS).toBe(EMBEDDER_DIM_SPECS.harrier.nativeDims);
    expect(HARRIER_NATIVE_DIMS).toBe(1024);
  });

  it('granite derives BOTH variant widths from the declared specs', () => {
    // The 97m/311m widths differ (384 vs 768) and are easy to transpose; the
    // builder reads them from the spec rather than hard-coding either.
    expect(graniteNativeDims('97m')).toBe(CANDIDATE_DIM_SPECS.granite97.nativeDims);
    expect(graniteNativeDims('311m')).toBe(CANDIDATE_DIM_SPECS.granite311.nativeDims);
    expect(graniteNativeDims('97m')).toBe(384);
    expect(graniteNativeDims('311m')).toBe(768);
    // And the variant→spec mapping is not crossed.
    expect(GRANITE_SPEC_KEYS['97m']).toBe('granite97');
    expect(GRANITE_SPEC_KEYS['311m']).toBe('granite311');
  });

  it('qwen3 embeds at its declared native width', () => {
    expect(QWEN3_NATIVE_DIMS).toBe(CANDIDATE_DIM_SPECS.qwen3.nativeDims);
    expect(QWEN3_NATIVE_DIMS).toBe(1024);
  });
});
