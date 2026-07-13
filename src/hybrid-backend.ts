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

  /**
   * Temporal-lite validity close — delegates to the COSINE (canonical PG)
   * leg like every lifecycle write; exposed only when that leg has the
   * capability. The lexical projection is reconciled by re-projection, not
   * per-id mirroring (same posture as forget/update above).
   */
  invalidateEntry(id: string, opts?: { supersededBy?: string }): Promise<boolean> {
    const impl = this.cosine.invalidateEntry?.bind(this.cosine);
    // A false here would read as "not found" upstream — a missing capability
    // must surface as an error, not a clean negative. (The live cosine leg is
    // the Mem0Backend, which always has it.)
    if (!impl) throw new Error('invalidateEntry: the cosine leg has no validity-window support');
    return impl(id, opts);
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
    const depth = this.opts.lexicalDepth ?? (opts.limit ?? 6) * 3;
    // ⚠ The two legs run CONCURRENTLY — they are INDEPENDENT (different rankings of the
    // same rows; neither's input depends on the other's output), so awaiting them in
    // sequence put the lexical leg's full cost on the critical path for no reason:
    // total = cosine + lexical instead of max(cosine, lexical).
    //
    // That was invisible while the lexical leg was a local file scan (~free), and it
    // became the dominant term the moment the leg became a PG query: it is most of why
    // `hybrid-pg` benched at p50 1182ms vs `hybrid`'s 821ms (memory-pg-lexical-own-
    // injection-2026-07-13 P-006, run 8). The cosine leg is embed-bound (a network
    // round-trip to the embedder); the lexical leg is embed-free and DB-bound. Overlapping
    // them hides the cheaper one entirely behind the one we cannot avoid.
    //
    // ⚠ ONLY the union path is overlapped. `cosine-gated` mode carries a DELIBERATE
    // short-circuit — an empty cosine set means "nothing relevant", so the lexical leg
    // must never be searched at all (that contract is pinned by hybrid-backend.test.ts
    // "cosine-gated early-return short-circuits the lexical leg entirely"). Racing the
    // leg eagerly would silently spend the very work that mode exists to avoid. So in
    // gated mode we keep the strict sequence; in floored-union mode (the default, and
    // what hybrid-pg runs) the leg ALWAYS runs anyway, so starting it early costs
    // nothing and removes it from the critical path.
    //
    // The leg is wrapped in an async IIFE with a try/catch — NOT a trailing `.catch()` on
    // the call — because `.catch()` only handles an async REJECTION, while a SYNCHRONOUS
    // throw from the method access itself (an undefined/misconfigured leg, or a leg missing
    // `.search`) escapes it. That asymmetry was EI-2777. Inside an `async` function a sync
    // throw becomes a rejection, so this catches BOTH and a broken lexical leg still
    // degrades search cleanly to cosine-only (header §"available()/degrades cleanly").
    const runLexical = (): Promise<MemoryEntry[]> =>
      (async () => {
        try {
          return await this.lexical.search(query, { scope: opts.scope, limit: depth });
        } catch {
          return [];
        }
      })();

    const gated = mode === 'cosine-gated';
    // Union mode: start the lexical leg NOW so it overlaps the embed-bound cosine call.
    const inFlightLexical = gated ? null : runLexical();

    // The cosine leg carries the FP floor (opts.minScore).
    const cosineHits = await this.cosine.search(query, opts);
    if (cosineHits.length === 0 && gated) return []; // lexical leg never started — as contracted
    const lexicalHits = await (inFlightLexical ?? runLexical());
    const fused = fuse(cosineHits, lexicalHits, {
      k: this.opts.rrfK ?? DEFAULT_RRF_K,
      mode,
      ...(minLexScore !== undefined ? { minLexScore } : {}),
      ...(this.opts.lexWeight !== undefined ? { lexWeight: this.opts.lexWeight } : {}),
    });
    return opts.limit !== undefined ? fused.slice(0, opts.limit) : fused;
  }

  /**
   * EMBED-FREE degraded-path fallback (WI-4214). Prefer the COSINE leg's
   * lexical capability — the canonical PG store covers ALL memories, while
   * the native lexical leg only sees projected writes (header §"in
   * production…"). If the cosine leg lacks/fails it, the native lexical
   * leg's search is itself embed-free, so it serves as the last resort. No
   * fusion here: this is an emergency recall path, not the ranked product.
   */
  async searchLexical(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    const cosineLex = this.cosine.searchLexical?.bind(this.cosine);
    if (cosineLex) {
      try {
        return await cosineLex(query, opts);
      } catch {
        /* fall through to the native lexical leg */
      }
    }
    return this.lexical.search(query, { scope: opts.scope, ...(opts.limit !== undefined ? { limit: opts.limit } : {}) });
  }
}
