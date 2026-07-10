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

import { embedViaWorker, ORT_SESSION_OPTIONS } from './local-embedder-worker';
import { mrlTruncate } from './gemma-embedder';

/** Transformers.js/ONNX build of microsoft/harrier-oss-v1-0.6b. */
export const HARRIER_MODEL = 'onnx-community/harrier-oss-v1-0.6b-ONNX';
/** Native output width — memory-side storage uses this (P-014). */
export const HARRIER_NATIVE_DIMS = 1024;
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

// Dodge bundler static analysis — @huggingface/transformers is an optional,
// lazily-resolved dependency (only present when a local embedder is selected).
const dynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as <T>(specifier: string) => Promise<T>;

const TRANSFORMERS_PACKAGE = '@huggingface/transformers';

type Tensor = { data: Float32Array };
/** The pipeline object doubles as tokenizer+model holder — the direct-call
 *  path uses those (mirroring the worker's `output` branch) since the
 *  pipeline's own pooling path cannot run on this export. */
type RawPipeline = {
  tokenizer: (text: string, opts?: Record<string, unknown>) => Record<string, unknown>;
  model: (inputs: Record<string, unknown>) => Promise<Record<string, Tensor>>;
};
type TransformersModule = {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<RawPipeline>;
};

/**
 * Build a harrier embedder closure `(text) => number[]` for a fixed embed
 * `kind`: last-token pooled + L2-normalized (in-graph), `dims`-wide (native
 * 1024 default; pass 384 for the exploratory truncated variant —
 * truncate-then-normalize via `mrlTruncate`, which at native width is a
 * no-op renorm of an already-unit vector).
 *
 * Prefers the worker-thread path (0.6B params — inference blocks the main
 * loop far worse than gemma-300m); falls back to an inline main-thread
 * model call when worker_threads is unavailable, with a transition-only
 * warn carrying the worker's real error (a swallowed model/output error
 * otherwise reads as a silent 2.4GB duplicate load). Fallback is sticky per
 * closure. Both paths cap ORT threads (the WI-3792 spin-pool storm).
 */
export function buildHarrierEmbedder(opts: {
  kind: HarrierEmbedKind;
  dims?: number;
}): (text: string) => Promise<number[]> {
  const { kind, dims = HARRIER_NATIVE_DIMS } = opts;
  let pipelinePromise: Promise<RawPipeline> | null = null;
  let workerDisabled = false;

  return async (text: string): Promise<number[]> => {
    const prompted = harrierPrompt(kind, text);

    if (!workerDisabled) {
      try {
        const full = await embedViaWorker(prompted, {
          model: HARRIER_MODEL,
          output: HARRIER_GRAPH_OUTPUT,
        });
        return mrlTruncate(assertNativeDims(full), dims);
      } catch (err) {
        workerDisabled = true;
        console.warn(
          `harrier embed worker path failed, falling back to inline: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    // Inline (main-thread) fallback — same direct model call as the worker.
    if (!pipelinePromise) {
      const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
      pipelinePromise = transformers.pipeline('feature-extraction', HARRIER_MODEL, {
        session_options: ORT_SESSION_OPTIONS,
      });
    }
    const pipe = await pipelinePromise;
    const enc = pipe.tokenizer(prompted, { padding: true, truncation: true });
    const out = await pipe.model(enc);
    const tensor = out[HARRIER_GRAPH_OUTPUT];
    if (!tensor) {
      throw new Error(
        `harrier model output '${HARRIER_GRAPH_OUTPUT}' missing (has: ${Object.keys(out).join(', ')})`,
      );
    }
    return mrlTruncate(assertNativeDims(Array.from(tensor.data)), dims);
  };
}
