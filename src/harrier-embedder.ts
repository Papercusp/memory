/**
 * Harrier-OSS-v1-0.6b local embedder (shared-embedding-sidecar-and-enrichment
 * P-013) — ADOPTED: this is the intended DEFAULT local embedder for a fresh
 * desktop install (offline embeddings out of the box), live since WI-4082
 * (2026-07-11); the embed-sidecar warms both harrier and gemma at boot and
 * both stay bundled in a LIVE HYBRID by owner directive (owner,
 * WI-5638 2026-07-20). The "bake-off candidate, not a default" framing below
 * predates that adoption — kept only as history for the truncated-384
 * variant, which is still exploratory (see DIMS below). A sibling of
 * `buildGemmaEmbedder` sharing the same worker-thread isolation via
 * `embedViaWorker`, differing in three ways harrier requires:
 *
 * 1. MODEL — `onnx-community/harrier-oss-v1-0.6b-ONNX` (Transformers.js/ONNX
 *    build of microsoft/harrier-oss-v1-0.6b), natively 1024-dim.
 *
 * 2. POOLING — LAST-TOKEN + L2 norm, and this export bakes BOTH into the ONNX
 *    graph (sentence-transformers head: /model/st/pool_0/lasttoken_* +
 *    /model/st/normalize_1/*). The graph's ONLY output is the pre-pooled
 *    `sentence_embedding` — there is NO last_hidden_state output, so the
 *    pipeline's JS-side pooling path cannot run at all. We read the graph
 *    output directly via `embedViaWorker`'s `output` option.
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

import {
  embedViaWorker,
  embedWorkerDisabledError,
  getWorkerState,
  isEmbedWorkerShutdownError,
  warnEmbedFallback,
} from './local-embedder-worker';
import { mrlTruncate } from './gemma-embedder';
import { EMBEDDER_DIM_SPECS } from './embedder-dims';

/** Transformers.js/ONNX build of microsoft/harrier-oss-v1-0.6b. */
export const HARRIER_MODEL = 'onnx-community/harrier-oss-v1-0.6b-ONNX';
/** Native output width — memory-side storage uses this (P-014). DERIVED from
 *  the declared spec so it cannot drift from the dims guard; harrier publishes
 *  NO MRL, so any narrower `dims` (the exploratory 384 variant) is an untrained
 *  cut — `isTrainedDim` in embedder-dims.ts reports it as such. */
export const HARRIER_NATIVE_DIMS = EMBEDDER_DIM_SPECS.harrier.nativeDims;
/** The export's sole graph output: last-token pooled + L2-normalized. */
export const HARRIER_GRAPH_OUTPUT = 'sentence_embedding';
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

/** A wrong-width vector means the graph output isn't what we think it is —
 *  fail loudly rather than store vectors from a mis-read space. */
function assertNativeDims(vec: number[]): number[] {
  if (vec.length !== HARRIER_NATIVE_DIMS) {
    throw new Error(`harrier ${HARRIER_GRAPH_OUTPUT}: expected ${HARRIER_NATIVE_DIMS} dims, got ${vec.length}`);
  }
  return vec;
}

/**
 * Build a harrier embedder closure `(text) => number[]` for a fixed embed
 * `kind`: last-token pooled + L2-normalized (in-graph), `dims`-wide (native
 * 1024 default; pass 384 for the exploratory truncated variant —
 * truncate-then-normalize via `mrlTruncate`, which at native width is a
 * no-op renorm of an already-unit vector).
 *
 * Inference runs only on the worker thread (0.6B params: on the main loop it
 * blocks far worse than gemma-300m). A worker failure rejects THIS call and
 * the caller degrades; nothing retries on the main thread
 * (EI-28961860650735669, see warnEmbedFallback). Every call re-checks the
 * worker (EI-16184): a crashed worker self-heals via `ensureWorker`'s
 * respawn, so a transient failure costs one rejected call.
 */
export function buildHarrierEmbedder(opts: {
  kind: HarrierEmbedKind;
  dims?: number;
}): (text: string) => Promise<number[]> {
  const { kind, dims = HARRIER_NATIVE_DIMS } = opts;

  return async (text: string): Promise<number[]> => {
    const prompted = harrierPrompt(kind, text);
    if (getWorkerState().disabled) {
      warnEmbedFallback('harrier', 'worker-disabled');
      throw embedWorkerDisabledError('harrier');
    }
    let full: number[];
    try {
      full = await embedViaWorker(prompted, {
        model: HARRIER_MODEL,
        output: HARRIER_GRAPH_OUTPUT,
      });
    } catch (err) {
      // WI-10006602: a deliberate shutdown is not a worker failure.
      if (!isEmbedWorkerShutdownError(err)) warnEmbedFallback('harrier', err);
      throw err;
    }
    return mrlTruncate(assertNativeDims(full), dims);
  };
}
