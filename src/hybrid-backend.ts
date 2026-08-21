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
 * - The legs may be asked DIFFERENT questions: `SearchOptions.lexicalQuery`
 *   overrides the text the lexical leg searches while the cosine leg keeps
 *   `query` (context-injection-audit-2026-07-28 P-044). Omit it and both legs
 *   get `query`, exactly as before. Read that option's doc before using it —
 *   the lexical score is normalized by query-token count, so a longer lexical
 *   query lowers every hit's score against `minLexScore`.
 */
import type {
  ListOptions,
  MemoryAvailability,
  MemoryBackend,
  MemoryEntry,
  RememberOptions,
  SearchLegStats,
  SearchOptions,
  UpdatePatch,
} from "./backend";
import { DEFAULT_RRF_K, fuse, type FusionMode } from "./hybrid-fusion";
import {
  createTextSimilarity,
  diversityDisabledByEnv,
  diversityRerank,
} from "./diversity-rerank";

/**
 * How deep into the fused list the optional diversity re-rank looks, as a
 * multiple of the caller's `limit`. 3 matches `lexicalDepth`'s existing
 * limit×3 candidate-pool convention in this file — deep enough that a
 * near-duplicate can be demoted clear of the top-K and a distinct hit promoted
 * into it, shallow enough that the O(n²) pass stays trivial on the injection
 * critical path.
 */
const RERANK_DEPTH_FACTOR = 3;

/** Monotonic where available; Date.now keeps the generic package portable. */
const nowMs = (): number =>
  typeof performance !== "undefined" ? performance.now() : Date.now();

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
  /**
   * Lexical admission bar for floored-union (normalized 0..1).
   *
   * Default **0.3** — `DEFAULT_MIN_LEX_SCORE` in hybrid-fusion.ts, which is
   * where the value is actually applied (`opts.minLexScore ?? DEFAULT_MIN_LEX_SCORE`).
   * This doc-comment previously claimed 0.5, which was never the effective
   * default on any path; corrected under context-injection-audit-2026-07-28
   * P-029 (I-D), whose whole point is that a stale comment about a tuning
   * constant is the next agent's false lead — someone reading 0.5 here would
   * conclude the bar had been LOOSENED to 0.4 by the push path, when it is in
   * fact TIGHTENED from 0.3.
   *
   * Resolution order, both legs of it real in production:
   *   opts.minLexScore  → the push path (memory/injection.ts) passes 0.40
   *   this.opts.minLexScore → NOT set by the production wiring
   *                           (configure.ts constructs HybridBackend with only
   *                           `{ name: 'hybrid-pg' }`)
   *   DEFAULT_MIN_LEX_SCORE → 0.3, what the pull path therefore gets
   */
  minLexScore?: number;
  /** Weight on the lexical leg's RRF contribution (default 1; >1 favors exact-id). */
  lexWeight?: number;
  /**
   * Identity this instance reports as `backend.name` (default `'hybrid'`).
   *
   * ⚠ MUST be set to the REGISTERED name by every caller that registers a
   * distinct hybrid wiring — `hybrid-pg` passes `'hybrid-pg'`. This used to be
   * hardcoded `'hybrid'`, which made two structurally different backends
   * (claude-file+mem0 vs PG-lexical+mem0) indistinguishable downstream, and that
   * is not cosmetic: the live recall canary keys its "did the backend change?"
   * reseed off exactly this string (`set.backend !== backend.name` in
   * bench/recall-canary.ts). With both wirings answering `'hybrid'`, a flip
   * between them silently skipped the reseed and kept scoring new results against
   * a baseline seeded on the OTHER backend — a corrupt comparison that reports as
   * a healthy one. It also printed two identical "hybrid" columns in the
   * 2026-07-13 bench scorecard. (memory-declaude-and-defaults-2026-07-28 P-003.)
   */
  name?: string;
}

export class HybridBackend implements MemoryBackend {
  readonly name: string;

  /**
   * `search()` ALWAYS returns fused RRF scores — there is no branch that
   * returns a leg's own scores. `fuse()` recomputes `score` for every surviving
   * candidate, and the only other exit from `search()` is the `cosine-gated`
   * early return, which yields an EMPTY array (no scores at all). So this is a
   * constant, not a mode-dependent value.
   */
  readonly scoreScale = "rrf" as const;

  /** `searchLexical()` delegates to the cosine leg's own lexical capability
   *  and does NOT fuse — those scores stay on the raw lexical scale. */
  readonly lexicalScoreScale = "lexical" as const;

  /**
   * Shared-embed capability (EI-12992) — delegated to the COSINE leg, which is
   * the ONLY leg that embeds (the lexical leg is embed-free by construction).
   *
   * ⚠ It MUST delegate to the leg that will CONSUME the vector. `search()`
   * forwards `opts` — and therefore `opts.vector` — to `this.cosine` (all of it
   * except `diversify`, which is a POST-fusion concern; see search()), so the
   * vector is only meaningful in the cosine leg's embedding space.
   * Sourcing it from anywhere else (a second embedder, the lexical leg) would
   * silently compare vectors across spaces and return plausible-looking
   * garbage rather than failing loudly.
   *
   * ASSIGNED CONDITIONALLY so the capability feature-test callers perform
   * (`backend.embedQuery?.(…)`) stays HONEST: a cosine leg that cannot embed
   * (NoopBackend, a lexical-only store) leaves this undefined, and the caller
   * falls back to per-call embedding exactly as before. Without this
   * delegation the wrapper hid the leg's capability entirely, which silently
   * made the EI-12992 shared-embed a NO-OP under `hybrid-pg` — the production
   * backend — while it worked in the bare-Mem0Backend tests.
   */
  readonly embedQuery?: (text: string) => Promise<number[] | null>;

  constructor(
    private readonly lexical: MemoryBackend,
    private readonly cosine: MemoryBackend,
    private readonly opts: HybridBackendOptions = {},
  ) {
    this.name = opts.name ?? "hybrid";
    const cosineEmbed = cosine.embedQuery?.bind(cosine);
    if (cosineEmbed) this.embedQuery = cosineEmbed;
  }

  available(): Promise<MemoryAvailability> {
    return this.cosine.available();
  }

  async remember(
    text: string,
    opts: RememberOptions,
  ): Promise<{ ids: string[]; storedEvents?: number }> {
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
    try {
      await this.lexical.remember(text, lexOpts);
    } catch {
      /* projection is best-effort */
    }
    return result;
  }

  get(id: string): Promise<MemoryEntry | null> {
    return this.cosine.get(id);
  }

  /** Remove any separate-store lexical copies linked to a canonical id.
   * `LexicalLegBackend` (hybrid-pg) is a shared-store ranking adapter, so its
   * rows carry no link_id and this is naturally a no-op there. */
  private async removeLexicalProjections(id: string): Promise<void> {
    // The lexical leg stores a projection under its OWN id, linked back to the
    // canonical id through metadata.link_id. Mutating only the cosine row left
    // that copy recallable after either hard forget OR soft invalidation.
    // Resolve and remove projections FIRST: if cleanup fails, the canonical row
    // remains current and available for a safe retry.
    const canonical = await this.cosine.get(id);
    if (!canonical) return;
    if (!canonical.scope) {
      throw new Error(`memory ${id} has no scope for lexical projection cleanup`);
    }
    const projections = await this.lexical.list({ scope: canonical.scope });
    for (const projection of projections) {
      if (projection.metadata?.link_id === id) await this.lexical.forget(projection.id);
    }
  }

  async forget(id: string): Promise<void> {
    await this.removeLexicalProjections(id);
    await this.cosine.forget(id);
  }

  /**
   * Temporal-lite validity close — delegates to the COSINE (canonical PG)
   * leg like every lifecycle write; exposed only when that leg has the
   * capability. The lexical projection is reconciled by re-projection, not
   * per-id mirroring (same posture as forget/update above).
   */
  async invalidateEntry(
    id: string,
    opts?: { supersededBy?: string },
  ): Promise<boolean> {
    const impl = this.cosine.invalidateEntry?.bind(this.cosine);
    // A false here would read as "not found" upstream — a missing capability
    // must surface as an error, not a clean negative. (The live cosine leg is
    // the Mem0Backend, which always has it.)
    if (!impl)
      throw new Error(
        "invalidateEntry: the cosine leg has no validity-window support",
      );
    await this.removeLexicalProjections(id);
    return impl(id, opts);
  }

  update(id: string, patch: UpdatePatch): Promise<void> {
    return this.cosine.update(id, patch);
  }

  list(opts: ListOptions): Promise<MemoryEntry[]> {
    return this.cosine.list(opts);
  }

  async search(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    const totalStartedAt = nowMs();
    // Per-call overrides (the P-031 sweep) win over the constructor defaults.
    const mode = opts.fusionMode ?? this.opts.fusionMode ?? "floored-union";
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
    // P-044 (F-L): the legs may be asked DIFFERENT questions. `lexicalQuery`
    // overrides the text this leg searches; omitted, it is `query` — so a caller
    // that does not use the seam is byte-identical to before. See
    // `SearchOptions.lexicalQuery` for why the split exists (the cosine leg
    // embeds ONE vector, so identifiers dilute it; the lexical leg is the leg
    // that wins on them) and for the length warning that comes with it.
    const lexicalText = opts.lexicalQuery?.trim() ? opts.lexicalQuery : query;
    let lexicalDurationMs: number | undefined;
    const runLexical = (): Promise<MemoryEntry[]> =>
      (async () => {
        const startedAt = nowMs();
        try {
          return await this.lexical.search(lexicalText, {
            scope: opts.scope,
            limit: depth,
          });
        } catch {
          return [];
        } finally {
          lexicalDurationMs = Math.max(0, nowMs() - startedAt);
        }
      })();

    const gated = mode === "cosine-gated";
    // Union mode: start the lexical leg NOW so it overlaps the embed-bound cosine call.
    const inFlightLexical = gated ? null : runLexical();

    // The cosine leg carries the FP floor (opts.minScore).
    //
    // ⚠ `diversify` is STRIPPED from what the cosine leg receives, and it is not
    // an optimization — forwarding it is a CORRECTNESS bug (context-injection-
    // audit-2026-07-28 F-D / D-014). `Mem0Backend.search` honors `diversify` by
    // MMR-reordering its own result, and `fuse()` below scores every hit by its
    // RANK WITHIN THAT LEG (`1/(k + cosineRank)`). So a forwarded `diversify`
    // would not diversify this backend's output at all — it would silently
    // rewrite the fusion's INPUT ranking, moving a demoted near-duplicate's RRF
    // contribution rather than demoting the entry. The pass belongs on the fused
    // list, which is the only globally-comparable ranking on this path.
    //
    // `lexicalQuery` is stripped for the same reason and it is the SAME class of
    // bug, not tidiness (P-044): the cosine leg's `search` takes `(query, opts)`,
    // so an option naming a DIFFERENT query text has no honest meaning there —
    // and a leg that later learned to read it would silently embed the
    // identifier-bearing string this option exists to keep OUT of the vector.
    // Strip an option at the boundary of the leg it is not for.
    // `onLegStats` is stripped for the same reason as `diversify`/`lexicalQuery`
    // above — it is THIS backend's reporting seam, not the cosine leg's, and a
    // leg that later learned to read it would report its own half as the whole.
    const {
      diversify,
      lexicalQuery: _lexicalQuery,
      onLegStats,
      ...cosineOpts
    } = opts;
    // A reporter must never be able to fail a search (SearchOptions.onLegStats).
    const report = (stats: SearchLegStats): void => {
      if (!onLegStats) return;
      try {
        onLegStats(stats);
      } catch {
        /* swallow — telemetry is never load-bearing */
      }
    };
    const cosineStartedAt = nowMs();
    const cosineHits = await this.cosine.search(query, cosineOpts);
    const cosineDurationMs = Math.max(0, nowMs() - cosineStartedAt);
    if (cosineHits.length === 0 && gated) {
      // The lexical leg NEVER STARTED — contracted behaviour, not a failure.
      // Reported as `ran: false` so a reader cannot mistake the short-circuit
      // for a leg that ran and found nothing; those have opposite fixes.
      report({
        mode,
        cosine: {
          ran: true,
          candidates: 0,
          qualifying: 0,
          ...(opts.limit !== undefined ? { depth: opts.limit } : {}),
          durationMs: cosineDurationMs,
        },
        lexical: { ran: false },
        fused: 0,
        totalMs: Math.max(0, nowMs() - totalStartedAt),
      });
      return []; // lexical leg never started — as contracted
    }
    const lexicalHits = await (inFlightLexical ?? runLexical());
    const observedFusion: {
      current?: {
        cosineCandidates: number;
        lexicalCandidates: number;
        lexicalQualifying: number;
        fused: number;
      };
    } = {};
    const fusionStartedAt = nowMs();
    const fused = fuse(cosineHits, lexicalHits, {
      k: this.opts.rrfK ?? DEFAULT_RRF_K,
      mode,
      ...(minLexScore !== undefined ? { minLexScore } : {}),
      ...(this.opts.lexWeight !== undefined
        ? { lexWeight: this.opts.lexWeight }
        : {}),
      ...(onLegStats
        ? {
            onFusionStats: (s) => {
              observedFusion.current = s;
            },
          }
        : {}),
    });
    const fusionMs = Math.max(0, nowMs() - fusionStartedAt);
    const diversified = this.diversify(fused, diversify, opts.limit);
    const result =
      opts.limit !== undefined ? diversified.slice(0, opts.limit) : diversified;
    const fusionStats = observedFusion.current;
    if (fusionStats) {
      // The cosine leg carries no post-hoc admission bar of its own (its FP floor
      // is applied inside the leg), so qualifying === candidates there. The
      // lexical leg's bar is `minLexScore`, and its two numbers are what say
      // whether that bar is doing anything on this workload.
      //
      // ⚠ The two legs have DIFFERENT budgets and each records its own: the cosine
      // leg gets `opts.limit`, the lexical leg pulls `depth` (= lexicalDepth ??
      // limit*3) PER SCOPE. Recording one for both would make every saturation
      // read on the other leg wrong by a factor of three.
      report({
        mode,
        cosine: {
          ran: true,
          candidates: fusionStats.cosineCandidates,
          qualifying: fusionStats.cosineCandidates,
          ...(opts.limit !== undefined ? { depth: opts.limit } : {}),
          durationMs: cosineDurationMs,
        },
        lexical: {
          ran: true,
          candidates: fusionStats.lexicalCandidates,
          qualifying: fusionStats.lexicalQualifying,
          depth,
          durationMs: lexicalDurationMs,
        },
        fused: fusionStats.fused,
        fusionMs,
        totalMs: Math.max(0, nowMs() - totalStartedAt),
      });
    }
    return result;
  }

  /**
   * Optional MMR pass over the FUSED list (EI-10230; wired here by
   * context-injection-audit-2026-07-28 F-D). Off unless the caller asks.
   *
   * Applied BEFORE `slice(0, limit)` — deliberately, and this is the whole
   * point of the pass here. The fused pool is much larger than `limit` (the
   * cosine leg alone returns up to `limit` per scope and a multi-scope pull
   * merges them), so re-ranking AFTER the slice could only reorder a top-K that
   * pure relevance had already filled with near-copies. Selecting K diverse
   * entries out of the larger post-floor candidate pool is what actually
   * reclaims the budget. The QUALITY invariant still holds exactly: every
   * candidate here already cleared its leg's admission bar (the cosine floor /
   * `minLexScore`), so diversity re-ranks the admitted set and can never
   * re-admit a floored-out hit.
   *
   * Bounded to a `limit`-relative head for cost: MMR is O(n²) in pairwise
   * similarity and this runs on the per-turn injection critical path inside a
   * timeout. Anything below `RERANK_DEPTH_FACTOR × limit` cannot reach the
   * final top-K by diversity alone, so re-ranking the deep tail buys nothing and
   * costs quadratically. The tail is preserved in relevance order behind the
   * head, so an unlimited caller still gets every entry back.
   */
  private diversify(
    fused: MemoryEntry[],
    diversify: SearchOptions["diversify"],
    limit: number | undefined,
  ): MemoryEntry[] {
    if (!diversify || diversityDisabledByEnv() || fused.length <= 1)
      return fused;
    const depth =
      limit === undefined
        ? fused.length
        : Math.min(fused.length, limit * RERANK_DEPTH_FACTOR);
    if (depth <= 1) return fused;
    const head = diversityRerank(fused.slice(0, depth), {
      lambda: diversify.lambda,
      similarity: createTextSimilarity(),
    });
    return depth >= fused.length ? head : [...head, ...fused.slice(depth)];
  }

  /**
   * EMBED-FREE degraded-path fallback (WI-4214). Prefer the COSINE leg's
   * lexical capability — the canonical PG store covers ALL memories, while
   * the native lexical leg only sees projected writes (header §"in
   * production…"). If the cosine leg lacks/fails it, the native lexical
   * leg's search is itself embed-free, so it serves as the last resort. No
   * fusion here: this is an emergency recall path, not the ranked product.
   */
  async searchLexical(
    query: string,
    opts: SearchOptions,
  ): Promise<MemoryEntry[]> {
    const cosineLex = this.cosine.searchLexical?.bind(this.cosine);
    if (cosineLex) {
      try {
        return await cosineLex(query, opts);
      } catch {
        /* fall through to the native lexical leg */
      }
    }
    return this.lexical.search(query, {
      scope: opts.scope,
      ...(opts.limit !== undefined ? { limit: opts.limit } : {}),
    });
  }
}
