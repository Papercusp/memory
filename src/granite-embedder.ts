/**
 * IBM Granite-Embedding-R2 local embedders (prose-embedding-384-untrained-mrl-fix
 * P-003) — BAKE-OFF CANDIDATES, not defaults: adoption is gated on a prose
 * gold-set win outside noise, judged against gemma@512/@768 per that plan's
 * D-003. Siblings of `buildGemmaEmbedder`, sharing the same worker-thread
 * isolation via `embedViaWorker`, differing in three ways granite requires:
 *
 * 1. MODELS — two sizes, and they are NOT interchangeable in width:
 *      97m  → 384-dim NATIVELY (no truncation happens at all)
 *      311m → 768-dim, with TRAINED Matryoshka nesting at 768/512/384/256/128
 *
 * 2. POOLING — **CLS**, not mean. Verified from each model's
 *    `1_Pooling/config.json` (`pooling_mode_cls_token: true`, every other mode
 *    false), not from a summary of the card. This is the single most important
 *    line in this file: pooling a CLS-trained model with `mean` returns a
 *    perfectly plausible vector from the wrong space, so the bake-off would
 *    read a measurement bug as a fair loss for granite.
 *
 * 3. PROMPTS — **symmetric**: no task prefix on either side. Verified from
 *    `config_sentence_transformers.json`, which declares BOTH prompts as the
 *    empty string (`{"query": "", "document": ""}`). Contrast gemma and Qwen3,
 *    which are asymmetric dual-encoders and are understated without their
 *    prefixes. Do not "helpfully" add one here.
 *
 * WHY THESE TWO ARE THE INTERESTING CANDIDATES. The prose surfaces store
 * `vector(384)`, and the incumbent gemma@384 reaches that width by an
 * UNTRAINED MRL cut (EI-19301722864393687) — measured on the prose gold set as
 * a real 19-21% MRR loss versus gemma@512/@768 (that plan's D-003). Both
 * granite options reach 384 legitimately, and therefore need NO wide-column
 * migration:
 *   - 97m is natively 384 — there is no cut to be untrained.
 *   - 311m publishes 384 as a TRAINED nesting point — the exact property gemma
 *     lacks. `CANDIDATE_DIM_SPECS.granite311` declares this, so the dims guard
 *     asserts it mechanically rather than leaving it as a claim in prose.
 *
 * SPACE: granite vectors are a DISTINCT embedding space from gemma/BGE/OpenAI
 * (the embedding-space-vs-dimension scar / EI-8913) — same 384 dims,
 * incomparable cosine. Adoption means a full re-embed, never a mixed table.
 */

import { embedViaWorker, getWorkerState, ORT_SESSION_OPTIONS, warnEmbedFallback } from './local-embedder-worker';
import { mrlTruncate } from './gemma-embedder';
import { CANDIDATE_DIM_SPECS } from './embedder-dims';

/** The two R2 multilingual sizes under evaluation. */
export type GraniteVariant = '97m' | '311m';

/** Transformers.js/ONNX builds of the granite-embedding-*-multilingual-r2 pair. */
export const GRANITE_MODELS: Record<GraniteVariant, string> = {
  '97m': 'onnx-community/granite-embedding-97m-multilingual-r2-ONNX',
  '311m': 'onnx-community/granite-embedding-311m-multilingual-r2-ONNX',
};

/** Bench-leg family key per variant — the `CANDIDATE_DIM_SPECS` entry, so the
 *  width a leg claims is checked against the declared spec (D-001). */
export const GRANITE_SPEC_KEYS: Record<GraniteVariant, 'granite97' | 'granite311'> = {
  '97m': 'granite97',
  '311m': 'granite311',
};

/** Native output width per variant — DERIVED from the declared spec so it
 *  cannot drift from the dims guard. */
export function graniteNativeDims(variant: GraniteVariant): number {
  return CANDIDATE_DIM_SPECS[GRANITE_SPEC_KEYS[variant]].nativeDims;
}

/**
 * Whether a text is being embedded as a stored document or a search query.
 *
 * Accepted for interface symmetry with the asymmetric embedders (gemma,
 * harrier, Qwen3) so a caller can plumb one `kind` through all of them —
 * granite DELIBERATELY ignores it, because its published prompts are empty on
 * both sides. Kept as a named type rather than dropped so that this is a
 * documented fact about granite, not an omission someone later "fixes".
 */
export type GraniteEmbedKind = 'document' | 'query';

/**
 * Build a granite-r2 embedder closure `(text) => number[]` for one variant.
 *
 * `dims` truncates (MRL) and re-normalizes the prefix — legitimate for 311m at
 * its trained nesting points, and a no-op width for 97m, which is natively 384.
 * As with gemma, the correct MRL procedure is truncate-THEN-normalize, so the
 * pipeline is asked for an UN-normalized vector and `mrlTruncate` normalizes
 * the slice. At the native width that reduces to a plain L2 normalize, which
 * is exactly granite's own `2_Normalize` module.
 */
export function buildGraniteEmbedder(opts: {
  variant: GraniteVariant;
  /** Ignored — granite is symmetric. Present so callers can pass one uniformly. */
  kind?: GraniteEmbedKind;
  dims?: number;
}): (text: string) => Promise<number[]> {
  const { variant } = opts;
  const model = GRANITE_MODELS[variant];
  const dims = opts.dims ?? graniteNativeDims(variant);
  let pipelinePromise: Promise<(text: string, o: unknown) => Promise<{ data: Float32Array }>> | null = null;

  return async (text: string): Promise<number[]> => {
    // No prompt: granite r2 declares both prompts empty (see docblock).
    if (!getWorkerState().disabled) {
      try {
        // normalize:false — MRL requires truncate-then-normalize, done below.
        const full = await embedViaWorker(text, { model, pooling: 'cls', normalize: false });
        return mrlTruncate(full, dims);
      } catch (err) {
        warnEmbedFallback(`granite-${variant}`, err);
      }
    }

    // Inline (main-thread) fallback — same thread-cap rationale as the worker.
    if (!pipelinePromise) {
      const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
      pipelinePromise = transformers.pipeline('feature-extraction', model, {
        session_options: ORT_SESSION_OPTIONS,
      });
    }
    const pipe = await pipelinePromise;
    const result = await pipe(text, { pooling: 'cls', normalize: false });
    return mrlTruncate(Array.from(result.data), dims);
  };
}

/** The @huggingface/transformers package (lazily resolved — optional dep). */
const TRANSFORMERS_PACKAGE = '@huggingface/transformers';

// Dodge bundler static analysis — @huggingface/transformers is an optional,
// lazily-resolved dependency (only present when a local embedder is selected).
const dynamicImport = new Function('specifier', 'return import(specifier)') as <T>(
  specifier: string,
) => Promise<T>;

type TransformersModule = {
  pipeline: (
    task: string,
    model: string,
    opts?: Record<string, unknown>,
  ) => Promise<(text: string, o: unknown) => Promise<{ data: Float32Array }>>;
};
