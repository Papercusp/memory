/**
 * Harrier-OSS-v1-0.6b local embedder (shared-embedding-sidecar-and-enrichment
 * P-013) — a BAKE-OFF CANDIDATE, not a default: adoption is gated on a P-001
 * gold-set win outside noise (P-006). A sibling of `buildGemmaEmbedder`
 * sharing the same worker-thread isolation via `embedViaWorker`, differing in
 * three ways harrier requires:
 *
 * 1. MODEL — `onnx-community/harrier-oss-v1-0.6b-ONNX` (Transformers.js/ONNX
 *    build of microsoft/harrier-oss-v1-0.6b), natively 1024-dim.
 *
 * 2. POOLING — LAST-TOKEN, not mean: the pipeline is asked for per-token
 *    output (`pooling: 'none'`) and `lastTokenPool` slices the final token's
 *    hidden state. Mean-pooling this model scores garbage — the pooling
 *    config is part of the space.
 *
 * 3. PROMPTS — asymmetric, Qwen3-embedding style: queries get
 *    `Instruct: {task}\nQuery: {q}` (model-card prompt_name=web_search_query);
 *    documents are embedded RAW (no prefix).
 *
 * DIMS: memory-side use is native 1024 (needs its own vector(1024) table —
 * P-014). Harrier has NO documented MRL, so the truncated-384 variant
 * (`dims: 384`, truncate-then-normalize like gemma) is EXPLORATORY: it must
 * WIN the P-001 gate before any prose-surface (vector(384)) use.
 *
 * SPACE: harrier vectors are a DISTINCT embedding space from gemma/BGE/OpenAI
 * (the embedding-space-vs-dimension scar) — adoption means a full re-embed.
 */

import { embedViaWorker, ORT_SESSION_OPTIONS } from './local-embedder-worker';
import { mrlTruncate } from './gemma-embedder';

/** Transformers.js/ONNX build of microsoft/harrier-oss-v1-0.6b. */
export const HARRIER_MODEL = 'onnx-community/harrier-oss-v1-0.6b-ONNX';
/** Native output width — memory-side storage uses this (P-014). */
export const HARRIER_NATIVE_DIMS = 1024;
/** The model-card web_search_query instruction (prompt_name=web_search_query). */
export const HARRIER_QUERY_TASK =
  'Given a web search query, retrieve relevant passages that answer the query';

/** Whether a text is being embedded as a stored document or a search query. */
export type HarrierEmbedKind = 'document' | 'query';

/**
 * Apply harrier's asymmetric prompting: queries carry the instruct prefix,
 * documents are raw. (Same cross-comparable dual-encoder contract as gemma:
 * wrong-prompt is a quality knob, not a correctness footgun.)
 */
export function harrierPrompt(kind: HarrierEmbedKind, text: string): string {
  return kind === 'query' ? `Instruct: ${HARRIER_QUERY_TASK}\nQuery: ${text}` : text;
}

/**
 * Last-token pooling over a FLATTENED per-token output ([seq, dims] row-major,
 * which is what `pooling: 'none'` yields for a single text): the last `dims`
 * slice is the final token's hidden state. Throws on a length that isn't a
 * whole number of tokens — a wrong-dims bug must be loud, never mis-sliced.
 */
export function lastTokenPool(flat: number[], dims: number = HARRIER_NATIVE_DIMS): number[] {
  if (flat.length < dims || flat.length % dims !== 0) {
    throw new Error(`lastTokenPool: length ${flat.length} is not a multiple of dims ${dims}`);
  }
  return flat.slice(flat.length - dims);
}

// Dodge bundler static analysis — @huggingface/transformers is an optional,
// lazily-resolved dependency (only present when a local embedder is selected).
const dynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as <T>(specifier: string) => Promise<T>;

const TRANSFORMERS_PACKAGE = '@huggingface/transformers';

type TransformersModule = {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<Pipeline>;
};
type Pipeline = (text: string, opts: unknown) => Promise<{ data: Float32Array }>;

/**
 * Build a harrier embedder closure `(text) => number[]` for a fixed embed
 * `kind`: last-token pooled, L2-normalized, `dims`-wide (native 1024 default;
 * pass 384 for the exploratory truncated variant — truncate-then-normalize
 * via `mrlTruncate`, which at native width is a plain L2 norm).
 *
 * Prefers the worker-thread path (0.6B params — inference blocks the main
 * loop far worse than gemma-300m); falls back to an inline main-thread
 * pipeline when worker_threads is unavailable. Fallback is sticky per closure.
 * Both paths cap ORT threads (the WI-3792 spin-pool storm).
 */
export function buildHarrierEmbedder(opts: {
  kind: HarrierEmbedKind;
  dims?: number;
}): (text: string) => Promise<number[]> {
  const { kind, dims = HARRIER_NATIVE_DIMS } = opts;
  let pipelinePromise: Promise<Pipeline> | null = null;
  let workerDisabled = false;

  return async (text: string): Promise<number[]> => {
    const prompted = harrierPrompt(kind, text);

    if (!workerDisabled) {
      try {
        const flat = await embedViaWorker(prompted, {
          model: HARRIER_MODEL,
          pooling: 'none',
          normalize: false,
        });
        return mrlTruncate(lastTokenPool(flat, HARRIER_NATIVE_DIMS), dims);
      } catch {
        workerDisabled = true;
      }
    }

    // Inline (main-thread) fallback.
    if (!pipelinePromise) {
      const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
      pipelinePromise = transformers.pipeline('feature-extraction', HARRIER_MODEL, {
        session_options: ORT_SESSION_OPTIONS,
      });
    }
    const pipe = await pipelinePromise;
    const result = await pipe(prompted, { pooling: 'none', normalize: false });
    return mrlTruncate(lastTokenPool(Array.from(result.data), HARRIER_NATIVE_DIMS), dims);
  };
}
