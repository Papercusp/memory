/**
 * EmbeddingGemma-300m local embedder (the DEFAULT local embedder as of the
 * 2026-07-10 owner ask). A sibling of BGE-small's `buildLocalEmbedder`
 * (local-embedder-worker.ts); shares the same worker-thread isolation via
 * `embedViaWorker`, differing in three ways EmbeddingGemma requires:
 *
 * 1. MODEL — `onnx-community/embeddinggemma-300m-ONNX` (Transformers.js/ONNX
 *    build of Google's EmbeddingGemma-300m, Sept 2025; MTEB-Multilingual-v2 #1
 *    among <500M-param models).
 *
 * 2. DIMENSION — EmbeddingGemma is natively 768-dim with Matryoshka (MRL)
 *    representation. ⚠ MRL nesting is TRAINED AT SPECIFIC TARGET DIMS, and per
 *    google/embeddinggemma-300m's model card the officially supported ones are
 *    768, 512, 256 and 128 — **384 is NOT among them**. So do NOT read this as
 *    "any prefix is a valid lower-dim embedding": an intermediate prefix is an
 *    UNTRAINED cut. It degrades gracefully rather than breaking (the space is
 *    still coherent — this is a quality question, not a correctness one), but
 *    it is not optimized, and 384 sits between two trained points with NO
 *    published measurement. It may sit BELOW the 512/256 interpolation rather
 *    than on it. Published curve (MTEB Multilingual v2, Mean(Task)):
 *      768 -> 61.15 | 512 -> 60.71 | 256 -> 59.68 | 128 -> 58.23   (384: unmeasured)
 *
 *    We nonetheless truncate to 384 + L2-renormalize (`mrlTruncate`) so it
 *    reuses the entire existing 384-dim vector infrastructure (the vector(384)
 *    columns, incl. the 5 shared-column prose surfaces) with NO wide column
 *    migration. That is a deliberate infrastructure-fit tradeoff, NOT a claim
 *    that 384 is a supported MRL dim — the original docblock asserted the
 *    latter, and that is precisely what licensed this cut unexamined
 *    (EI-19301722864393687). The correct MRL procedure is
 *    truncate-THEN-normalize, so we ask the pipeline for an UN-normalized
 *    vector (`normalize: false`) and normalize the 384-slice ourselves.
 *
 *    If you are changing this target: prefer a TRAINED dim (512/256), or a
 *    natively-384 model (granite-embedding-97m-multilingual-r2), over another
 *    intermediate prefix. Full 768 remains a deferred max-quality follow-up
 *    (needs wider column migrations). Tracked by plan
 *    `prose-embedding-384-untrained-mrl-fix-2026-08-02`, which MEASURES this
 *    rather than assuming either way.
 *
 * 3. TASK PROMPTS — EmbeddingGemma is an asymmetric dual-encoder trained with
 *    task prefixes. Documents and queries get DIFFERENT prompts, and the two
 *    are designed to be cross-comparable, so storage uses the document prompt
 *    and search queries use the query prompt (`gemmaPrompt`). Passing the wrong
 *    prompt still lands in the same space (same model) — only slightly
 *    suboptimal — so this is a quality knob, never a correctness footgun.
 *
 * SPACE: EmbeddingGemma vectors are a DISTINCT embedding space from BGE and
 * OpenAI (the embedding-space-vs-dimension scar / EI-8913) — same 384 dims,
 * incomparable cosine. They live in their own `memory_vec_gemma` table
 * (migration 534), selected by `ResolvedEmbedder.mode === 'gemma'`.
 */

import { embedViaWorker, getWorkerState, ORT_SESSION_OPTIONS, warnEmbedFallback } from './local-embedder-worker';
import { EMBEDDER_DIM_SPECS } from './embedder-dims';

/** Transformers.js/ONNX build of EmbeddingGemma-300m. */
export const GEMMA_MODEL = 'onnx-community/embeddinggemma-300m-ONNX';
/** The @huggingface/transformers package (lazily resolved — optional dep). */
const TRANSFORMERS_PACKAGE = '@huggingface/transformers';
/** MRL truncation target — chosen to match the existing vector(384) columns,
 *  NOT because 384 is a trained MRL dim. EmbeddingGemma's supported MRL dims are
 *  768/512/256/128; 384 is an untrained intermediate cut. See the DIMENSION note
 *  in the file docblock before changing this (EI-19301722864393687).
 *
 *  DERIVED from the declared spec rather than restated, so the dim this file
 *  truncates to is by construction the dim `embedder-dims.ts` declares — and
 *  the guard there cannot be satisfied by a declaration that has drifted from
 *  the code it describes. Change the target THERE, where the trained dims and
 *  the untrained-cut acknowledgement sit next to it. */
export const GEMMA_TARGET_DIMS = EMBEDDER_DIM_SPECS.gemma.targetDims;

/** Whether a text is being embedded as a stored document or a search query.
 *  EmbeddingGemma applies a different task prompt to each. */
export type GemmaEmbedKind = 'document' | 'query';

/**
 * Prepend EmbeddingGemma's task prompt. The official prompts (Google model
 * card / sentence-transformers `encode_query`/`encode_document`):
 *   query:    "task: search result | query: {text}"
 *   document: "title: none | text: {text}"
 */
export function gemmaPrompt(kind: GemmaEmbedKind, text: string): string {
  return kind === 'query'
    ? `task: search result | query: ${text}`
    : `title: none | text: ${text}`;
}

/**
 * MRL-truncate a (Matryoshka) embedding to `dims` and L2-renormalize the
 * prefix. The renorm is on the TRUNCATED slice — a prefix of a unit vector is
 * not itself unit-norm — so cosine over the truncated space is well-behaved.
 * A zero vector (degenerate) is returned truncated but un-normalized rather
 * than divided by zero.
 */
export function mrlTruncate(vec: number[], dims: number = GEMMA_TARGET_DIMS): number[] {
  const sliced = vec.length > dims ? vec.slice(0, dims) : vec;
  let sumSq = 0;
  for (const v of sliced) sumSq += v * v;
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return sliced;
  return sliced.map((v) => v / norm);
}

// Dodge bundler static analysis — @huggingface/transformers is an optional,
// lazily-resolved dependency (only present when a local embedder is selected).
const dynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as <T>(specifier: string) => Promise<T>;

type TransformersModule = {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<Pipeline>;
};
type Pipeline = (text: string, opts: unknown) => Promise<{ data: Float32Array }>;

/**
 * Build an EmbeddingGemma-300m embedder closure `(text) => number[]` (384-dim,
 * MRL-truncated + renormalized) for a fixed embed `kind`. Storage/mem0 build
 * with `kind: 'document'`; search's query embedder builds with `kind: 'query'`.
 *
 * Prefers the worker-thread path (off the main event loop — EmbeddingGemma-300m
 * is ~10x BGE-small's params, so main-thread inference blocks noticeably);
 * falls back to an inline main-thread pipeline for THIS call when the worker
 * is unavailable or the call fails. EI-16184: this used to be sticky per
 * closure (a single worker failure permanently disabled the worker path,
 * silently, for the rest of that closure's — often hour-long cached — life,
 * making every subsequent embed block the main event loop). Every call now
 * re-checks the module's own liveness (`getWorkerState().disabled` — set
 * only for genuine, permanent unavailability; a crashed worker instead
 * self-heals via `ensureWorker`'s respawn), so a transient failure costs at
 * most one degraded call.
 *
 * `dims` overrides the truncation width for MEASUREMENT ONLY (the P-002 MRL
 * sweep scores 768/512/384/256 against one gold set), mirroring the `dims`
 * knob `buildHarrierEmbedder` already carries. It deliberately does NOT
 * weaken D-001: the DEFAULT is still derived from `EMBEDDER_DIM_SPECS`, so
 * every production caller — which passes no `dims` — remains bound to the
 * declared, guard-checked target, and a bench leg cannot change what ships.
 */
export function buildGemmaEmbedder(opts: {
  kind: GemmaEmbedKind;
  dims?: number;
}): (text: string) => Promise<number[]> {
  const { kind, dims = GEMMA_TARGET_DIMS } = opts;
  let pipelinePromise: Promise<Pipeline> | null = null;

  return async (text: string): Promise<number[]> => {
    const prompted = gemmaPrompt(kind, text);

    if (!getWorkerState().disabled) {
      try {
        // normalize:false — MRL requires truncate-then-normalize, done below.
        const full = await embedViaWorker(prompted, { model: GEMMA_MODEL, normalize: false });
        return mrlTruncate(full, dims);
      } catch (err) {
        warnEmbedFallback('gemma', err);
      }
    }

    // Inline (main-thread) fallback.
    if (!pipelinePromise) {
      const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
      // Same thread-cap rationale as the worker path (WI-3792 spin-pool storm).
      pipelinePromise = transformers.pipeline('feature-extraction', GEMMA_MODEL, {
        session_options: ORT_SESSION_OPTIONS,
      });
    }
    const pipe = await pipelinePromise;
    const result = await pipe(prompted, { pooling: 'mean', normalize: false });
    return mrlTruncate(Array.from(result.data), dims);
  };
}
