/**
 * HybridBackend (memory-backend-improve-and-hybrid P-020).
 *
 * Fuses a LEXICAL leg (exact-identifier — the claude-file token search) and a
 * COSINE leg (semantic/paraphrase — the mem0/canonical pgvector store) so one
 * backend captures BOTH columns the bench showed are complementary
 * (claude-file exact-id MRR 0.99 + mem0 paraphrase 5/6) instead of a lossy
 * either/or (D-001).
 *
 * - READS fuse via reciprocal-rank fusion (see hybrid-fusion). DEFAULT mode is
 *   `floored-union`: the result is the FP-floored cosine hits (SearchOptions.minScore)
 *   UNION the lexical hits that clear the identifier-precision bar (minLexScore) —
 *   so an exact-id target the cosine leg missed is still captured (the lexical leg
 *   is a co-equal recall source, not just a re-ranker), while a hard-negative floors
 *   away in cosine AND fails the lexical bar → empty. `cosine-gated` mode (lexical
 *   re-ranks only, never admits) stays selectable for the strictest discipline.
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
import { DEFAULT_RRF_K, fuse, type FusionMode } from './hybrid-fusion';

export interface HybridBackendOptions {
  /** RRF damping constant (default 60). */
  rrfK?: number;
  /** How many lexical hits to pull for re-ranking / union (default 3× the search limit). */
  lexicalDepth?: number;
  /**
   * Fusion mode (default 'floored-union' — admits strong lexical-only hits so the
   * exact-identifier column is captured, not capped at the cosine leg's recall).
   */
  fusionMode?: FusionMode;
  /** Lexical admission bar for floored-union (normalized 0..1, default 0.5). */
  minLexScore?: number;
  /** Weight on the lexical leg's RRF contribution (default 1; >1 favors exact-id). */
  lexWeight?: number;
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

  async remember(text: string, opts: RememberOptions): Promise<{ ids: string[]; storedEvents?: number }> {
    // The CANONICAL write — the cosine (PG) store owns ids + durability.
    const result = await this.cosine.remember(text, opts);
    // Write-through PROJECTION into the lexical native surface so its leg can
    // serve exact-identifier reads over the SAME memories the cosine leg sees
    // (otherwise the lexical leg is empty and the hybrid degrades to cosine-only,
    // losing the exact-id column). Best-effort: a cold/missing lexical leg (e.g.
    // no ~/.claude dir) is non-fatal — the cosine gate still carries recall.
    // forget/update target the canonical leg; the lexical projection is
    // reconciled by re-projection, not per-delete id-mapping (P-022 / D-002).
    //
    // Stamp `link_id` = the canonical id onto the projection so fusion can DEDUPE
    // a memory that surfaces from BOTH legs (the legs assign different native ids
    // to the same fact; without a shared key it would appear twice in recall).
    const linkId = result.ids[0];
    const lexOpts: RememberOptions = linkId
      ? { ...opts, metadata: { ...(opts.metadata ?? {}), link_id: linkId } }
      : opts;
    try { await this.lexical.remember(text, lexOpts); } catch { /* projection is best-effort */ }
    return result;
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
    // Per-call overrides (the P-031 sweep) win over the constructor defaults.
    const mode = opts.fusionMode ?? this.opts.fusionMode ?? 'floored-union';
    const minLexScore = opts.minLexScore ?? this.opts.minLexScore;
    // The cosine leg carries the FP floor (opts.minScore). In cosine-gated mode an
    // empty cosine set means "nothing relevant" → return early; in floored-union
    // mode the lexical leg can still contribute strong identifier hits, so we run it.
    const cosineHits = await this.cosine.search(query, opts);
    if (cosineHits.length === 0 && mode === 'cosine-gated') return [];
    const depth = this.opts.lexicalDepth ?? (opts.limit ?? 6) * 3;
    // Lexical leg: a re-rank (gated mode) or a co-equal recall source (union mode);
    // if it's unavailable, degrade cleanly to cosine-only.
    const lexicalHits = await this.lexical
      .search(query, { scope: opts.scope, limit: depth })
      .catch(() => [] as MemoryEntry[]);
    const fused = fuse(cosineHits, lexicalHits, {
      k: this.opts.rrfK ?? DEFAULT_RRF_K,
      mode,
      ...(minLexScore !== undefined ? { minLexScore } : {}),
      ...(this.opts.lexWeight !== undefined ? { lexWeight: this.opts.lexWeight } : {}),
    });
    return opts.limit !== undefined ? fused.slice(0, opts.limit) : fused;
  }
}
