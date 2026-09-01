/**
 * @papercusp/memory — the mem0-backed persistent-memory store core.
 *
 * Host-agnostic: the embedded-pg admin URL, LLM credentials, the resolved
 * embedder (a pre-built embed fn + mode + dims), the explicit-mode
 * embedder builder, and the optional adaptive-instruction feed are all
 * injected via `configureMemory()`. mem0 owns its own `pg.Client`, so
 * no shared transaction handle is injected.
 *
 * The operator's curation pipeline (learning loop, pre-turn injection,
 * harness anchoring, dedup judge, insights index) stays in the operator
 * and is built ON TOP of this store via the `memory:*` tool wrappers.
 *
 * Extracted per papercusp-systems-abstraction-2026-05-29 (P-021).
 */

export {
  configureMemory,
  memoryHost,
  isMemoryConfigured,
  type MemoryHost,
  type MemoryCredentials,
  type ResolvedEmbedder,
  type EmbedFn,
} from './config';

export {
  getMemoryClient,
  getResolvedMode,
  invalidateMemoryClient,
  disposeMemoryClient,
} from './mem0-client';

// Injectable fact-extraction LLM seam (mem0-extraction-via-claude-session
// D-003): hosts implement `ExtractionLlm` (mem0ai's LLM shape) and hand it
// back from `MemoryHost.getExtractionLlm` to become cascade rung #1.
// `ExtractionAuthError` is the typed throw that triggers a STICKY demotion
// to the key rungs (D-004); `FallbackExtractionLlm` is the cascade wrapper
// (exported for hosts that compose their own cascades + for tests).
export {
  ExtractionAuthError,
  FallbackExtractionLlm,
  type ExtractionLlm,
  type ExtractionLlmMessage,
  type ExtractionLlmResponse,
} from './extraction-llm';

// The neutral, swappable store seam (generalize-memory-backend-swappable
// D-001/D-002/D-004). Consumers call getMemoryBackend() and the neutral
// verbs; which store serves them is the host's `backend` config flip.
export {
  MemoryUnavailableError,
  scopesOf,
  type ListOptions,
  type MemoryAvailability,
  type MemoryBackend,
  type LegRunStats,
  type MemoryEntry,
  type RetrievalProvenance,
  type ScoreScale,
  type SearchLegStats,
  type RememberOptions,
  type SearchFloorPolicy,
  type SearchOptions,
  type SearchOptionsCommon,
  type UpdatePatch,
} from './backend';
export {
  LEXICAL_SCOPE_CONCURRENCY,
  Mem0Backend,
  extractAddedIds,
  type Mem0BackendDeps,
} from './mem0-backend';
export { applyScoreFloor, type ScoreFloorOptions } from './score-floor';
/** The ONE runtime-opaque optional-dependency import — see ./dynamic-import. */
export { dynamicImport } from './dynamic-import';
export {
  createTextSimilarity,
  diversityDisabledByEnv,
  diversityRerank,
  lexicalSimilarity,
  textSimilarity,
  type DiversityRerankOptions,
} from './diversity-rerank';
export { HybridBackend, type HybridBackendOptions } from './hybrid-backend';
export { LexicalLegBackend } from './lexical-leg';
export {
  fuse,
  fuseCosineGated,
  DEFAULT_RRF_K,
  DEFAULT_MIN_LEX_SCORE,
  type FusionMode,
  type FusionOptions,
} from './hybrid-fusion';
export { NoopBackend, NOOP_DISABLED_REASON } from './noop-backend';
// The Claude Code topic-file bridge (generalize-memory-backend-swappable
// D-005 / claude-memory-projection-integration P-004): read/write the
// native Claude file memory through the neutral seam. Hosts register it:
//   registerMemoryBackend('claude-file', () => new ClaudeFileMemoryBackend({ memoryDir }));
export {
  ClaudeFileMemoryBackend,
  CLAUDE_FILE_BACKEND_NAME,
  MEMORY_DIR_MISSING_REASON,
  type ClaudeFileBackendOptions,
} from './claude-file-backend';
export {
  parseTopicFile,
  serializeTopicFile,
  typeForKind,
  slugify,
  deriveDescription,
  claudeProjectMemoryDir,
  CLAUDE_MEMORY_TYPES,
  type ClaudeMemoryType,
  type TopicFile,
} from './topic-file';
export {
  getMemoryBackend,
  registerMemoryBackend,
  registeredMemoryBackends,
  _resetMemoryBackendsForTest,
} from './backend-registry';

export {
  connectionString,
  pgFields,
  pgClientFields,
  type Mem0PgConnection,
} from './mem0-connection';

export {
  reembedMemories,
  type ReembedResult,
} from './reembed';

export {
  relinkEntities,
  type RelinkResult,
} from './relink-entities';

export {
  activeVecTable,
  embedAndUpsertVector,
  VEC_TABLE,
  MODE_DIMS,
  type ResolvedVecMode,
} from './vec-write';

export {
  embedViaWorker,
  buildLocalEmbedder,
  getWorkerState,
  _resetWorker,
  shutdownLocalEmbedder,
  LOCAL_EMBEDDER_MODEL,
} from './local-embedder-worker';

export {
  buildGemmaEmbedder,
  gemmaPrompt,
  mrlTruncate,
  GEMMA_MODEL,
  GEMMA_TARGET_DIMS,
  type GemmaEmbedKind,
} from './gemma-embedder';

// The declared dim table (D-001). Exported so callers that CHOOSE a width —
// the MRL bench sweep especially — can ask whether that width is one the model
// was actually trained at, instead of re-encoding the model card by hand and
// letting the two drift.
export {
  EMBEDDER_DIM_SPECS,
  CANDIDATE_DIM_SPECS,
  dimSpecFor,
  isTrainedDim,
  validateEmbedderDimSpec,
  type EmbedderMode,
  type EmbedderDimSpec,
  type MrlSupport,
  type UntrainedCutAck,
} from './embedder-dims';

// Bake-off candidates for the prose surfaces' untrained-MRL-384 cut (plan
// prose-embedding-384-untrained-mrl-fix-2026-08-02, P-003). Exported so the
// eval CLI measures the SAME builders a winner would ship with — measuring a
// bench-local reimplementation and then shipping a different one is how an
// embedding decision goes unexamined in the first place.
export {
  buildGraniteEmbedder,
  graniteNativeDims,
  GRANITE_MODELS,
  GRANITE_SPEC_KEYS,
  type GraniteVariant,
  type GraniteEmbedKind,
} from './granite-embedder';

export {
  buildQwen3Embedder,
  qwen3Prompt,
  QWEN3_MODEL,
  QWEN3_NATIVE_DIMS,
  QWEN3_QUERY_TASK,
  type Qwen3EmbedKind,
} from './qwen3-embedder';

export {
  buildHarrierEmbedder,
  harrierPrompt,
  HARRIER_MODEL,
  HARRIER_NATIVE_DIMS,
  HARRIER_GRAPH_OUTPUT,
  HARRIER_QUERY_TASK,
  type HarrierEmbedKind,
} from './harrier-embedder';

export {
  buildSidecarFirstEmbedder,
  sidecarEmbedBatch,
  resolveEmbedSidecarUrl,
  EMBED_SIDECAR_URL_ENV,
  DEFAULT_SIDECAR_TIMEOUT_MS,
  DEFAULT_SIDECAR_MAX_ATTEMPTS,
  SIDECAR_MAX_TEXT_CHARS,
  SIDECAR_BATCH_PER_TEXT_MS,
  SIDECAR_BATCH_TIMEOUT_CAP_MS,
  sidecarBatchTimeoutMs,
  SidecarEmbedHttpError,
  isNonRetryableSidecarError,
  type SidecarFirstEmbedderOpts,
  type SidecarEmbedBatchOpts,
  type SidecarEmbedResponse,
} from './sidecar-embedder';
export { normalizeEmbeddingText } from './embed-coalesce';

export { CanonicalVectorStore, LEXICAL_QUERY_CONCURRENCY } from './canonical-store';
/**
 * The lexical leg's own tokenizer. Exported because anything ASKING a question
 * of that leg needs to know what its query will actually be reduced to — the
 * 32-token cap keeps the HEAD, so a caller composing a query can only tell
 * whether the part it appended survived by running the real tokenizer.
 * Re-implementing it would defeat the point.
 */
export { lexicalTokens } from './canonical-store';
