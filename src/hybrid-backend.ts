/**
 * HybridBackend (memory-backend-improve-and-hybrid P-020).
 *
 * Fuses a LEXICAL leg (exact-identifier — the claude-file token search) and a
 * COSINE leg (semantic/paraphrase — the mem0/canonical pgvector store) so one
 * backend captures BOTH columns the bench showed are complementary
 * (claude-file exact-id MRR 0.99 + mem0 paraphrase 5/6) instead of a lossy
 * either/or (D-001).
 *
 * - READS fuse via cosine-gated reciprocal-rank fusion (see hybrid-fusion): the
 *   cosine leg (FP-floored by SearchOptions.minScore) is the gate; the lexical
 *   leg only re-ranks, so a hard-negative still returns empty.
 * - WRITES (remember/forget/update/list/get) delegate to the COSINE backend —
 *   the canonical PG store — so the hybrid is cross-backend BY CONSTRUCTION: any
 *   client's `memory:*` call (claude/omp/codex all hit the same operator) lands
 *   in the one shared store, and a remember from any client is recallable from
 *   all three (D-002). The lexical leg searches the same memories once they are
 *   projected into its native form (the claude topic files). In the bench both
 *   legs are seeded identically, so fusion is exercised directly; in production
 *   that native-surface projection of the canonical store into each client's
 *   auto-inject surface rides the owner-deferred mem0 revive (P-022 /
 *   docs-and-memory-as-projections D-008) — until then the lexical leg only
 *   re-ranks what is already in the native store and the cosine gate carries
 *   recall, so the hybrid degrades cleanly to cosine-only for un-projected writes.
 * - `available()` tracks the cosine leg (the source of truth + write target); a
 *   missing/cold lexical leg just degrades search to cosine-only.
 */
import type {
  ListOptions,
  MemoryAvailability,
  MemoryBackend,
  MemoryEntry,
  RememberOptions,
  SearchOptions,
  UpdatePatch,
} from './backend';
import { DEFAULT_RRF_K, fuseCosineGated } from './hybrid-fusion';

export interface HybridBackendOptions {
  /** RRF damping constant (default 60). */
  rrfK?: number;
  /** How many lexical hits to pull for re-ranking (default 3× the search limit). */
  lexicalDepth?: number;
}

export class HybridBackend implements MemoryBackend {
  readonly name = 'hybrid';

  constructor(
    private readonly lexical: MemoryBackend,
    private readonly cosine: MemoryBackend,
    private readonly opts: HybridBackendOptions = {},
  ) {}

  available(): Promise<MemoryAvailability> {
    return this.cosine.available();
  }

  remember(text: string, opts: RememberOptions): Promise<{ ids: string[]; storedEvents?: number }> {
    return this.cosine.remember(text, opts);
  }

  get(id: string): Promise<MemoryEntry | null> {
    return this.cosine.get(id);
  }

  forget(id: string): Promise<void> {
    return this.cosine.forget(id);
  }

  update(id: string, patch: UpdatePatch): Promise<void> {
    return this.cosine.update(id, patch);
  }

  list(opts: ListOptions): Promise<MemoryEntry[]> {
    return this.cosine.list(opts);
  }

  async search(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    // The cosine leg carries the FP floor (opts.minScore) → its hits are the
    // gated candidate set; an empty set means the query matched nothing relevant.
    const cosineHits = await this.cosine.search(query, opts);
    if (cosineHits.length === 0) return [];
    const depth = this.opts.lexicalDepth ?? (opts.limit ?? 6) * 3;
    // Lexical leg is a re-rank enhancement; if it's unavailable, degrade cleanly.
    const lexicalHits = await this.lexical
      .search(query, { scope: opts.scope, limit: depth })
      .catch(() => [] as MemoryEntry[]);
    return fuseCosineGated(cosineHits, lexicalHits, this.opts.rrfK ?? DEFAULT_RRF_K);
  }
}
