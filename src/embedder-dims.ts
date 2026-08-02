/**
 * Declared embedding dimensions per embedder — the recurrence guard for
 * EI-19301722864393687 (the prose surfaces' untrained MRL-384 cut).
 *
 * THE BUG CLASS. Every embedder here emits a vector of some NATIVE width, and
 * one may be tempted to truncate it to fit a narrower storage column.
 * Truncating a Matryoshka (MRL) model is legitimate *only at the dims it was
 * trained to nest at*; truncating anywhere else is an UNTRAINED cut. It
 * degrades gracefully rather than breaking — the space stays coherent, so
 * nothing throws, no test reds, and quality just quietly sits somewhere
 * unmeasured. That silence is the whole problem: gemma@384 shipped because a
 * docblock asserted any prefix was "a valid lower-dim embedding", and nothing
 * in the code disagreed. It cost ~19-21% of prose retrieval MRR for months
 * before anyone measured it (D-003).
 *
 * ⚠ THE LESSON IS *MEASURE*, NOT "PREFER A TRAINED DIM" (D-003 correction 1).
 * Trainedness turned out to predict quality in NEITHER direction: the untrained
 * 384 scored ABOVE the trained-512/256 interpolation, and the TRAINED 256 was
 * dramatically WORSE than it (-.0889 MRR). A reflex to "just pick a trained
 * dim" would have selected 256 and made retrieval worse. What this file
 * guarantees is that a width is DECLARED and REVIEWABLE — the gold set, never
 * this table, says which width is actually good.
 *
 * WHAT THIS FILE CHANGES. The target dim is no longer a bare number sitting
 * next to the truncation call — it is DECLARED here alongside the dims the
 * model actually supports, and `validateEmbedderDimSpec` fails the build's
 * unit test when the two disagree without an explicit, justified
 * acknowledgement. A model swap that truncates to an untrained dim now has to
 * say so out loud, in writing, with a tracking ref.
 *
 * SINGLE SOURCE, SO IT CANNOT DRIFT. The builders derive their constants FROM
 * this table (`GEMMA_TARGET_DIMS`, `HARRIER_NATIVE_DIMS`) rather than
 * restating them, so the value that is declared here is the value the model
 * is actually truncated to. A declaration that could drift from the code it
 * describes would just be a second docblock — which is exactly what already
 * failed.
 *
 * ⚠ This is a QUALITY guard, not a correctness one. An untrained cut still
 * produces a usable vector; the acknowledgement exists to make the tradeoff
 * deliberate and reviewable, never to imply the alternative is broken.
 */

export type EmbedderMode = 'openai' | 'local' | 'gemma' | 'harrier';

/**
 * How a model supports being shortened below its native width:
 *
 * - `none`       — no MRL training. ONLY the native width is trained; any
 *                  truncation is an untrained cut.
 * - `discrete`   — MRL trained at SPECIFIC nesting points (the common case).
 *                  Only those exact dims are trained; the gaps between them
 *                  are not, and interpolating across them is unsound.
 * - `continuous` — reduction is a supported, first-class operation at any
 *                  width (e.g. OpenAI's `dimensions` API parameter, which
 *                  applies the trained MRL + renormalizes server-side).
 */
export type MrlSupport = 'none' | 'discrete' | 'continuous';

/**
 * A deliberate, reviewed decision to truncate to a dim the model was NOT
 * trained for. Required to be explicit — you cannot silence the guard without
 * writing down why and where it is tracked.
 */
export interface UntrainedCutAck {
  /** WHY this cut is being taken despite being untrained. */
  reason: string;
  /** Where the decision is tracked (a plan slug, work-item, or issue id). */
  trackedBy: string;
}

export interface EmbedderDimSpec {
  /** The model this spec describes, for error messages. */
  model: string;
  /** The width the model natively emits, before any truncation. */
  nativeDims: number;
  mrl: MrlSupport;
  /**
   * Dims this model was TRAINED to emit. Always includes `nativeDims` (a
   * model is by definition trained at its own output width). For `mrl: 'none'`
   * that is the only entry; for `discrete`, the published nesting points.
   * Ignored when `mrl` is `continuous`.
   */
  trainedDims: readonly number[];
  /** The width we actually configure this embedder to produce. */
  targetDims: number;
  /** Required iff `targetDims` is not a trained dim. */
  untrainedCut?: UntrainedCutAck;
}

/**
 * Is `dims` a width this model was actually trained to emit?
 *
 * Exported so a caller taking a non-default width (harrier's exploratory
 * 384 variant) can ask the same question the test asks, rather than
 * re-deriving the rule.
 */
export function isTrainedDim(spec: EmbedderDimSpec, dims: number): boolean {
  if (dims === spec.nativeDims) return true;
  if (dims <= 0 || dims > spec.nativeDims) return false;
  if (spec.mrl === 'continuous') return true;
  if (spec.mrl === 'none') return false;
  return spec.trainedDims.includes(dims);
}

/**
 * Structural violations in one spec — empty means the spec is sound.
 *
 * Returns ALL violations rather than throwing on the first, so a failing test
 * reports everything wrong with a new embedder in one run instead of one
 * problem per re-run.
 */
export function validateEmbedderDimSpec(mode: string, spec: EmbedderDimSpec): string[] {
  const problems: string[] = [];
  const at = `${mode} (${spec.model})`;

  if (spec.nativeDims <= 0) problems.push(`${at}: nativeDims must be positive, got ${spec.nativeDims}`);
  if (spec.targetDims <= 0) problems.push(`${at}: targetDims must be positive, got ${spec.targetDims}`);
  if (spec.targetDims > spec.nativeDims) {
    problems.push(
      `${at}: targetDims ${spec.targetDims} exceeds nativeDims ${spec.nativeDims} — a model cannot be truncated UP`,
    );
  }
  if (!spec.trainedDims.includes(spec.nativeDims)) {
    problems.push(
      `${at}: trainedDims ${JSON.stringify(spec.trainedDims)} omits nativeDims ${spec.nativeDims} — a model is trained at its own output width by definition`,
    );
  }
  if (spec.mrl === 'none' && spec.trainedDims.length !== 1) {
    problems.push(
      `${at}: mrl:'none' means only the native width is trained, so trainedDims must be exactly [${spec.nativeDims}], got ${JSON.stringify(spec.trainedDims)}`,
    );
  }

  const trained = isTrainedDim(spec, spec.targetDims);
  if (!trained && !spec.untrainedCut) {
    problems.push(
      `${at}: targetDims ${spec.targetDims} is NOT a trained dim (native ${spec.nativeDims}, mrl '${spec.mrl}'` +
        `${spec.mrl === 'discrete' ? `, trained at ${spec.trainedDims.join('/')}` : ''}). ` +
        `Prefer a trained dim; if the cut is deliberate, declare untrainedCut { reason, trackedBy } so the tradeoff is reviewable.`,
    );
  }
  if (trained && spec.untrainedCut) {
    // A stale ack is worse than none: it silences the guard for a dim that no
    // longer needs silencing, so the NEXT bad change inherits the exemption.
    problems.push(
      `${at}: targetDims ${spec.targetDims} IS a trained dim, so the untrainedCut acknowledgement is stale — remove it, or it will silently pre-authorize a future untrained cut`,
    );
  }
  if (spec.untrainedCut) {
    if (!spec.untrainedCut.reason.trim()) problems.push(`${at}: untrainedCut.reason must not be empty`);
    if (!spec.untrainedCut.trackedBy.trim()) problems.push(`${at}: untrainedCut.trackedBy must not be empty`);
  }

  return problems;
}

/**
 * Every embedder mode's declared dimensions.
 *
 * Keyed by the `EmbedderMode` union so adding a mode without declaring its
 * dims is a TYPE error, and the accompanying test additionally fails if a
 * declared spec is unsound. Both halves matter: the type catches the omission,
 * the test catches the bad declaration.
 */
export const EMBEDDER_DIM_SPECS: Record<EmbedderMode, EmbedderDimSpec> = {
  /**
   * BGE-small is natively 384 — no truncation happens at all, which is why
   * this mode never had the bug.
   */
  local: {
    model: 'Xenova/bge-small-en-v1.5',
    nativeDims: 384,
    mrl: 'none',
    trainedDims: [384],
    targetDims: 384,
  },

  /**
   * OpenAI text-embedding-3-small. Reduction is first-class here: the
   * `dimensions` API parameter applies the trained MRL and renormalizes
   * server-side, so 384 is a SUPPORTED width, not a hand-rolled prefix.
   * This is the contrast case that keeps the guard honest — it must not
   * flag every non-native width, only untrained ones.
   */
  openai: {
    model: 'text-embedding-3-small',
    nativeDims: 1536,
    mrl: 'continuous',
    trainedDims: [1536],
    targetDims: 384,
  },

  /**
   * EmbeddingGemma-300m at its NATIVE 768 — the case this guard was built for,
   * resolved by REMOVING the truncation rather than relocating it to a
   * different trained point (P-005 / D-005).
   *
   * The former `targetDims: 384` was an intermediate, untrained MRL cut taken
   * to fit the pre-existing `vector(384)` columns. Measured (D-003), it cost
   * ~19-21% of prose retrieval MRR: @384 scores .2889 against @512 .3435 and
   * @768 .3492, and the paired bootstrap vs @384 excludes zero for both
   * (@512 +.0546 CI [.0165,.0935]; @768 +.0603 CI [.0210,.1022]).
   *
   * WHY NATIVE 768 AND NOT THE TRAINED 512 (D-005). @768 is better on both the
   * point estimate and the CI lower bound, and — the decisive part — costs the
   * SAME compute: the model always runs a native-768 forward pass, so an MRL
   * "truncation" is only a slice afterwards. 512 would buy ~400MB of disk at
   * the price of keeping the truncation machinery, and therefore keeping this
   * bug class one config typo away. At the native width `targetDims ===
   * nativeDims`, so `isTrainedDim` is trivially true and no acknowledgement can
   * ever be needed here again.
   *
   * ⚠ Do NOT "optimize" this back down to a narrower prefix without re-reading
   * D-005: the re-embed corpus is ~395k vectors, so a width change is a
   * multi-hour, all-at-once migration (pgvector cannot cast between widths).
   * That cost is why the width chosen here is the model's terminal one.
   */
  gemma: {
    model: 'onnx-community/embeddinggemma-300m-ONNX',
    nativeDims: 768,
    mrl: 'discrete',
    trainedDims: [768, 512, 256, 128],
    targetDims: 768,
  },

  /**
   * harrier-oss-v1-0.6b at its NATIVE 1024 (its own vector(1024) table), so
   * the configured target needs no acknowledgement. Harrier publishes no MRL
   * at all, which is why the exploratory `dims: 384` variant is an untrained
   * cut — see `isTrainedDim`, which reports it as such, and the P-001 gate it
   * must win before any prose-surface use.
   */
  harrier: {
    model: 'microsoft/harrier-oss-v1-0.6b',
    nativeDims: 1024,
    mrl: 'none',
    trainedDims: [1024],
    targetDims: 1024,
  },
};

/**
 * BAKE-OFF CANDIDATES — models being MEASURED as replacements, not yet wired
 * as production `EmbedderMode`s.
 *
 * Deliberately a separate table rather than new `EmbedderMode` entries. A mode
 * implies a resolver branch, a vector table and a migration; these have earned
 * none of that until they win the prose gold-set gate
 * (prose-embedding-384-untrained-mrl-fix-2026-08-02, P-003). Keeping them out
 * of `EMBEDDER_DIM_SPECS` preserves that table's real guarantee — its
 * `Record<EmbedderMode, …>` key type is what makes shipping a mode without
 * declaring its dims a TYPE error — while still holding candidates to the same
 * validator, so a candidate cannot be measured under a width nobody checked.
 *
 * ⚠ READ THE ACKNOWLEDGEMENTS AS HYGIENE, NOT AS A QUALITY RANKING. It is
 * tempting to notice that gemma@384 carries an `untrainedCut` while the
 * granite entries do not, and conclude the granite widths are therefore
 * better. That inference is MEASURED FALSE (P-002 / D-003): gemma@384 scores
 * ABOVE the trained 512/256 interpolation, and the TRAINED 256 is dramatically
 * WORSE than the untrained 384 (-.0889 MRR). Trainedness predicted quality in
 * neither direction. What these specs assert is only that a width is declared
 * and reviewable — the gold set, not this table, says which model to ship.
 */
export const CANDIDATE_DIM_SPECS: Record<'granite97' | 'granite311' | 'qwen3', EmbedderDimSpec> = {
  /**
   * Granite-Embedding-97M-Multilingual-R2 — natively 384. The cheapest
   * possible resolution of the untrained-cut bug: at the width the prose
   * columns already store, there is no truncation happening at all, so the
   * question "is this cut trained?" does not arise.
   */
  granite97: {
    model: 'onnx-community/granite-embedding-97m-multilingual-r2-ONNX',
    nativeDims: 384,
    mrl: 'none',
    trainedDims: [384],
    targetDims: 384,
  },

  /**
   * Granite-Embedding-311M-Multilingual-R2 — 768 native, with card-published
   * Matryoshka nesting at 768/512/384/256/128. 384 is among them, so
   * `targetDims: 384` needs no `untrainedCut` acknowledgement (and if someone
   * later adds one anyway, the validator's stale-ack rule rejects it).
   *
   * That is a statement about DECLARABILITY, not quality — see the warning on
   * this table. Whether 311m@384 actually beats gemma@512 is P-003's
   * measurement to make.
   */
  granite311: {
    model: 'onnx-community/granite-embedding-311m-multilingual-r2-ONNX',
    nativeDims: 768,
    mrl: 'discrete',
    trainedDims: [768, 512, 384, 256, 128],
    targetDims: 384,
  },

  /**
   * Qwen3-Embedding-0.6B — 1024 native, card-documented MRL across
   * user-defined widths from 32 to 1024, so reduction is first-class the way
   * OpenAI's `dimensions` parameter is (hence `continuous`).
   *
   * ⚠ `continuous` here means "trained at any width in the published range",
   * and that range has a FLOOR of 32 which this model shares with no other
   * entry. `isTrainedDim` cannot express a floor, so it would call dims 1..31
   * trained. Our target of 384 sits far inside the range; a future caller
   * asking for a sub-32 width would need that floor checked by hand.
   */
  qwen3: {
    model: 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
    nativeDims: 1024,
    mrl: 'continuous',
    trainedDims: [1024],
    targetDims: 384,
  },
};

/**
 * The declared spec for a production mode OR a bake-off candidate, by key.
 *
 * The eval CLI scores both from one code path (`gemma@512` and `granite311@384`
 * are the same kind of question), so it needs one lookup spanning both tables.
 * Returns `undefined` for an unknown key rather than throwing — the CLI turns
 * that into a per-leg "unknown leg" refusal, which is more useful than
 * aborting a multi-leg sweep over one typo.
 */
export function dimSpecFor(key: string): EmbedderDimSpec | undefined {
  const production = (EMBEDDER_DIM_SPECS as Record<string, EmbedderDimSpec | undefined>)[key];
  if (production) return production;
  return (CANDIDATE_DIM_SPECS as Record<string, EmbedderDimSpec | undefined>)[key];
}
