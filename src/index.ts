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
} from './mem0-client';

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
} from './local-embedder-worker';

export { CanonicalVectorStore } from './canonical-store';
