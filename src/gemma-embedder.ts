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
 *    representation, so a PREFIX of the vector is itself a valid lower-dim
 *    embedding. We truncate to 384 + L2-renormalize (`mrlTruncate`) so it
 *    reuses the entire existing 384-dim vector infrastructure (the vector(384)
 *    columns, incl. the 5 shared-column prose surfaces) with NO wide column
 *    migration. The correct MRL procedure is truncate-THEN-normalize, so we ask
 *    the pipeline for an UN-normalized vector (`normalize: false`) and
 *    normalize the 384-slice ourselves. Full 768 is a deferred max-quality
 *    follow-up (needs wider column migrations).
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

import { embedViaWorker, ORT_SESSION_OPTIONS } from './local-embedder-worker';

/** Transformers.js/ONNX build of EmbeddingGemma-300m. */
export const GEMMA_MODEL = 'onnx-community/embeddinggemma-300m-ONNX';
/** The @huggingface/transformers package (lazily resolved — optional dep). */
const TRANSFORMERS_PACKAGE = '@huggingface/transformers';
/** MRL truncation target — matches the existing vector(384) columns. */
export const GEMMA_TARGET_DIMS = 384;

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
 * falls back to an inline main-thread pipeline when worker_threads is
 * unavailable (older Node, some vitest envs). Fallback is sticky per closure.
 */
export function buildGemmaEmbedder(opts: { kind: GemmaEmbedKind }): (text: string) => Promise<number[]> {
  const { kind } = opts;
  let pipelinePromise: Promise<Pipeline> | null = null;
  let workerDisabled = false;

  return async (text: string): Promise<number[]> => {
    const prompted = gemmaPrompt(kind, text);

    if (!workerDisabled) {
      try {
        // normalize:false — MRL requires truncate-then-normalize, done below.
        const full = await embedViaWorker(prompted, { model: GEMMA_MODEL, normalize: false });
        return mrlTruncate(full, GEMMA_TARGET_DIMS);
      } catch {
        workerDisabled = true;
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
    return mrlTruncate(Array.from(result.data), GEMMA_TARGET_DIMS);
  };
}
