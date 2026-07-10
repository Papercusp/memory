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
  type MemoryEntry,
  type RememberOptions,
  type SearchOptions,
  type UpdatePatch,
} from './backend';
export { Mem0Backend, extractAddedIds, type Mem0BackendDeps } from './mem0-backend';
export { applyScoreFloor, type ScoreFloorOptions } from './score-floor';
export { HybridBackend, type HybridBackendOptions } from './hybrid-backend';
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
  embedViaWorker,
  buildLocalEmbedder,
  getWorkerState,
  _resetWorker,
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

export {
  buildSidecarFirstEmbedder,
  sidecarEmbedBatch,
  resolveEmbedSidecarUrl,
  EMBED_SIDECAR_URL_ENV,
  DEFAULT_SIDECAR_TIMEOUT_MS,
  DEFAULT_REPROBE_AFTER_MS,
  type SidecarFirstEmbedderOpts,
  type SidecarEmbedBatchOpts,
  type SidecarEmbedResponse,
} from './sidecar-embedder';

export { CanonicalVectorStore } from './canonical-store';
