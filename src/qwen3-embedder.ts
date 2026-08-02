/**
 * Qwen3-Embedding-0.6B local embedder (prose-embedding-384-untrained-mrl-fix
 * P-003) — a BAKE-OFF CANDIDATE, not a default: adoption is gated on a prose
 * gold-set win outside noise, judged against gemma@512/@768 per that plan's
 * D-003. A sibling of `buildGemmaEmbedder` sharing the same worker-thread
 * isolation via `embedViaWorker`, differing in three ways Qwen3 requires:
 *
 * 1. MODEL — `onnx-community/Qwen3-Embedding-0.6B-ONNX`, natively 1024-dim,
 *    with MRL supporting user-defined widths across 32..1024.
 *
 * 2. POOLING — **LAST TOKEN**, not mean and not CLS. Qwen3-Embedding is a
 *    decoder; its sentence representation is the final position's hidden state
 *    (the official `last_token_pool`). ⚠ Unlike harrier — which is also a
 *    last-token Qwen3-family model — this export does NOT bake pooling into
 *    the graph: its outputs are `last_hidden_state` plus the KV cache, with no
 *    `sentence_embedding`. So harrier's `output` trick does not transfer, and
 *    the pooling genuinely has to happen on the JS side.
 *
 *    THE POOLING IS ONLY CORRECT BECAUSE WE EMBED ONE TEXT PER CALL.
 *    transformers.js implements `last_token` as the final sequence position
 *    with no attention-mask check; over a single text nothing is padded, so
 *    that position is the last real token. See the batching warning in
 *    local-embedder-worker.script.mjs before ever batching this.
 *
 *    The last real token is the tokenizer's appended `<|endoftext|>` (verified:
 *    "hello world" tokenizes to [14990, 1879, 151643]) — which is exactly the
 *    representation Qwen3-Embedding is trained to pool, not an off-by-one.
 *
 * 3. PROMPTS — **asymmetric**: queries carry an instruct prefix, documents are
 *    embedded RAW. Taken verbatim from the model's own
 *    `config_sentence_transformers.json`, including the detail that there is NO
 *    space after `Query:` (matching the card's `get_detailed_instruct` helper).
 *    Scoring Qwen3 with one symmetric prompt understates it and voids the
 *    comparison — the same trap the eval CLI's docblock flags for gemma.
 *
 * SPACE: Qwen3 vectors are a DISTINCT embedding space from gemma/granite/BGE
 * (the embedding-space-vs-dimension scar / EI-8913) — a shared width is NOT a
 * shared space, and adoption means a full re-embed.
 */

import { embedViaWorker, getWorkerState, ORT_SESSION_OPTIONS, warnEmbedFallback } from './local-embedder-worker';
import { mrlTruncate } from './gemma-embedder';
import { CANDIDATE_DIM_SPECS } from './embedder-dims';

/** Transformers.js/ONNX build of Qwen/Qwen3-Embedding-0.6B. */
export const QWEN3_MODEL = 'onnx-community/Qwen3-Embedding-0.6B-ONNX';

/** Native output width — DERIVED from the declared spec so it cannot drift. */
export const QWEN3_NATIVE_DIMS = CANDIDATE_DIM_SPECS.qwen3.nativeDims;

/** The retrieval task string from the model's declared `query` prompt. */
export const QWEN3_QUERY_TASK = 'Given a web search query, retrieve relevant passages that answer the query';

/** Whether a text is being embedded as a stored document or a search query. */
export type Qwen3EmbedKind = 'document' | 'query';

/**
 * Apply Qwen3's asymmetric prompting: queries carry the instruct prefix,
 * documents are raw.
 *
 * ⚠ The lack of a space after `Query:` is deliberate and copied from the
 * model's own config (`"Instruct: ...\nQuery:"`), which sentence-transformers
 * concatenates directly onto the text. Harrier's sibling prompt DOES have a
 * space, per its own card — do not "fix" one to match the other.
 */
export function qwen3Prompt(kind: Qwen3EmbedKind, text: string): string {
  return kind === 'query' ? `Instruct: ${QWEN3_QUERY_TASK}\nQuery:${text}` : text;
}

/**
 * Build a Qwen3-Embedding-0.6B embedder closure `(text) => number[]` for a
 * fixed embed `kind`.
 *
 * `dims` MRL-truncates and re-normalizes the prefix (truncate-THEN-normalize,
 * as with gemma). Qwen3 documents MRL across 32..1024, so 384 — the width the
 * prose surfaces store — is a supported reduction rather than the untrained
 * cut gemma@384 is.
 */
export function buildQwen3Embedder(opts: {
  kind: Qwen3EmbedKind;
  dims?: number;
}): (text: string) => Promise<number[]> {
  const { kind, dims = QWEN3_NATIVE_DIMS } = opts;
  let pipelinePromise: Promise<(text: string, o: unknown) => Promise<{ data: Float32Array }>> | null = null;

  return async (text: string): Promise<number[]> => {
    const prompted = qwen3Prompt(kind, text);

    if (!getWorkerState().disabled) {
      try {
        // normalize:false — MRL requires truncate-then-normalize, done below.
        const full = await embedViaWorker(prompted, {
          model: QWEN3_MODEL,
          pooling: 'last_token',
          normalize: false,
        });
        return mrlTruncate(full, dims);
      } catch (err) {
        warnEmbedFallback('qwen3', err);
      }
    }

    // Inline (main-thread) fallback — same thread-cap rationale as the worker.
    if (!pipelinePromise) {
      const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
      pipelinePromise = transformers.pipeline('feature-extraction', QWEN3_MODEL, {
        session_options: ORT_SESSION_OPTIONS,
      });
    }
    const pipe = await pipelinePromise;
    const result = await pipe(prompted, { pooling: 'last_token', normalize: false });
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
