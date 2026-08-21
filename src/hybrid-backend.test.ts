import { describe, it, expect, vi, afterEach } from "vitest";
import { HybridBackend } from "./hybrid-backend";
import { Mem0Backend } from "./mem0-backend";
import { LexicalLegBackend } from "./lexical-leg";
import { NoopBackend, NOOP_DISABLED_REASON } from "./noop-backend";
import { MemoryUnavailableError } from "./backend";
import type {
  MemoryBackend,
  MemoryEntry,
  RememberOptions,
  SearchLegStats,
  SearchOptions,
} from "./backend";

const e = (id: string, score?: number): MemoryEntry => ({
  id,
  text: id,
  scope: "s",
  ...(score !== undefined ? { score } : {}),
});

function fakeBackend(
  name: string,
  searchResult: MemoryEntry[],
  over: Partial<MemoryBackend> = {},
): MemoryBackend {
  return {
    name,
    available: async () => ({ ok: true }),
    remember: async () => ({ ids: [] }),
    search: async () => searchResult,
    list: async () => [],
    get: async () => null,
    forget: async () => {},
    update: async () => {},
    ...over,
  } as MemoryBackend;
}

describe("HybridBackend identity (memory-declaude-and-defaults-2026-07-28 P-003)", () => {
  it("reports the name it was CONSTRUCTED with, so two hybrid wirings are distinguishable", () => {
    const mem0 = new Mem0Backend();
    const pg = new HybridBackend(new LexicalLegBackend(mem0), mem0, {
      name: "hybrid-pg",
    });
    expect(pg.name).toBe("hybrid-pg");
  });

  it('defaults to "hybrid" when no name is given (the legacy claude-file+mem0 wiring)', () => {
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", []),
    );
    expect(hy.name).toBe("hybrid");
  });

  it("two DIFFERENT wirings never collide on one name — the canary reseed guard depends on this", () => {
    // bench/recall-canary.ts decides whether to reseed its frozen pair set with
    // `set.backend !== backend.name`. While both wirings answered 'hybrid', a
    // flip between them skipped the reseed and scored new results against a
    // baseline seeded on the OTHER backend — a corrupt comparison reported as a
    // healthy one. Equal names here would restore exactly that bug.
    const mem0 = new Mem0Backend();
    const pg = new HybridBackend(new LexicalLegBackend(mem0), mem0, {
      name: "hybrid-pg",
    });
    const legacy = new HybridBackend(fakeBackend("claude-file", []), mem0);
    expect(pg.name).not.toBe(legacy.name);
  });
});

describe("HybridBackend (P-020)", () => {
  it("fuses: an exact-id hit (in both legs) ranks above a paraphrase (cosine-only)", async () => {
    const cosine = fakeBackend("cosine", [e("para", 0.6), e("exact", 0.5)]);
    const lexical = fakeBackend("lexical", [e("exact", 9)]);
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.search("q", {
      scope: "s",
      limit: 6,
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(out.map((x) => x.id)).toEqual(["exact", "para"]);
  });

  it("hard-negative: cosine floored empty + weak lexical overlap → empty (dual gate, floored-union default)", async () => {
    const cosine = fakeBackend("cosine", []); // floored upstream
    // Realistic hard-negative lexical: generic token overlap scores BELOW the
    // identifier bar (0.5), so the union does not admit it.
    const lexical = fakeBackend("lexical", [e("weak-overlap", 0.2)]);
    const hy = new HybridBackend(lexical, cosine);
    expect(
      await hy.search("kubernetes", {
        scope: "s",
        fusionMode: "floored-union",
        minScore: 0.45,
      }),
    ).toEqual([]);
  });

  it("floored-union: a STRONG lexical-only identifier hit IS admitted (captures exact-id the cosine leg missed)", async () => {
    // The cosine leg missed the exact-id target entirely (only a paraphrase);
    // the lexical leg matched it on the identifier token (score ≥ 0.5) → admitted.
    const cosine = fakeBackend("cosine", [e("para", 0.55)]);
    const lexical = fakeBackend("lexical", [e("CODEX_HOME", 1.0)]);
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.search("CODEX_HOME rotation", {
      scope: "s",
      limit: 6,
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(out.map((x) => x.id).sort()).toEqual(["CODEX_HOME", "para"]);
  });

  it("floored-union: a WEAK lexical-only hit (below minLexScore) is NOT admitted", async () => {
    const cosine = fakeBackend("cosine", [e("para", 0.55)]);
    const lexical = fakeBackend("lexical", [e("weak", 0.3)]);
    const hy = new HybridBackend(lexical, cosine);
    // Per-call minLexScore override (0.5) — 0.3 is below it → excluded.
    const out = await hy.search("q", {
      scope: "s",
      fusionMode: "floored-union",
      minScore: 0.45,
      minLexScore: 0.5,
    });
    expect(out.map((x) => x.id)).toEqual(["para"]); // weak lexical-only excluded
  });

  it("per-call minLexScore override beats the constructor/default bar", async () => {
    const cosine = fakeBackend("cosine", [e("para", 0.55)]);
    const lexical = fakeBackend("lexical", [e("mid", 0.35)]);
    // Default bar 0.30 would admit 0.35; a per-call 0.50 override rejects it.
    const hy = new HybridBackend(lexical, cosine);
    expect(
      (await hy.search("q", { scope: "s" })).map((x) => x.id).sort(),
    ).toEqual(["mid", "para"]);
    expect(
      (await hy.search("q", { scope: "s", minLexScore: 0.5 })).map((x) => x.id),
    ).toEqual(["para"]);
  });

  it("cosine-gated mode: a lexical-only hit is NEVER admitted, even when strong", async () => {
    const cosine = fakeBackend("cosine", []); // floored empty
    const lexical = fakeBackend("lexical", [e("CODEX_HOME", 1.0)]);
    const hy = new HybridBackend(lexical, cosine, {
      fusionMode: "cosine-gated",
    });
    // Mirrors the constructor above: this call inherited 'cosine-gated' from it
    // before the floor/fusion pairing made the mode explicit at every floored site.
    expect(
      await hy.search("CODEX_HOME", {
        scope: "s",
        fusionMode: "cosine-gated",
        minScore: 0.45,
      }),
    ).toEqual([]);
  });

  it("preserves a paraphrase (cosine finds it, lexical misses)", async () => {
    const cosine = fakeBackend("cosine", [e("para", 0.55)]);
    const lexical = fakeBackend("lexical", []);
    const hy = new HybridBackend(lexical, cosine);
    expect(
      (await hy.search("reworded", { scope: "s" })).map((x) => x.id),
    ).toEqual(["para"]);
  });

  it("degrades to cosine-only when the lexical leg throws", async () => {
    const cosine = fakeBackend("cosine", [e("a", 0.6), e("b", 0.5)]);
    const lexical = fakeBackend("lexical", [], {
      search: async () => {
        throw new Error("claude dir missing");
      },
    });
    const hy = new HybridBackend(lexical, cosine);
    expect((await hy.search("q", { scope: "s" })).map((x) => x.id)).toEqual([
      "a",
      "b",
    ]);
  });

  // EI-2777: the lexical-leg call must survive a SYNCHRONOUS throw, not just an async
  // rejection. The async-reject case (above) passed even before the fix because a
  // trailing `.catch()` handles a rejected promise — but a sync throw from the method
  // ACCESS (an undefined/misconfigured leg, or a leg missing `.search`) escapes `.catch()`
  // and surfaced as the deterministic, search-only "reading 'search'" tool error (while
  // remember()'s try/catch swallowed the identical fault). These two would FAIL on the
  // old `.catch()` code and pass on the try/catch.
  it("degrades to cosine-only when the lexical leg is undefined (EI-2777)", async () => {
    const cosine = fakeBackend("cosine", [e("a", 0.6), e("b", 0.5)]);
    const hy = new HybridBackend(undefined as unknown as MemoryBackend, cosine);
    expect((await hy.search("q", { scope: "s" })).map((x) => x.id)).toEqual([
      "a",
      "b",
    ]);
  });

  it("degrades to cosine-only when the lexical leg.search throws synchronously (EI-2777)", async () => {
    const cosine = fakeBackend("cosine", [e("a", 0.6), e("b", 0.5)]);
    const lexical = fakeBackend("lexical", [], {
      search: (() => {
        throw new Error("sync boom — not a rejected promise");
      }) as never,
    });
    const hy = new HybridBackend(lexical, cosine);
    expect((await hy.search("q", { scope: "s" })).map((x) => x.id)).toEqual([
      "a",
      "b",
    ]);
  });

  it("writes the canonical id and forget removes linked lexical projections before the canonical row", async () => {
    const remember = vi.fn(async () => ({ ids: ["1"] }));
    const order: string[] = [];
    const forget = vi.fn(async () => { order.push("canonical"); });
    const cosine = fakeBackend("cosine", [], { remember, forget, get: async () => e("1") });
    const lexicalForget = vi.fn(async () => { order.push("lexical"); });
    const lexical = fakeBackend("lexical", [], {
      list: async () => [
        { ...e("projection-1"), metadata: { link_id: "1" } },
        { ...e("unrelated"), metadata: { link_id: "other" } },
      ],
      forget: lexicalForget,
    });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.remember("fact", { scope: "s" });
    await hy.forget("1");
    expect(remember).toHaveBeenCalledOnce();
    expect(out.ids).toEqual(["1"]); // canonical (cosine) ids are returned
    expect(lexicalForget).toHaveBeenCalledWith("projection-1");
    expect(forget).toHaveBeenCalledWith("1");
    expect(order).toEqual(["lexical", "canonical"]);
  });

  it("does not delete the canonical row when lexical projection cleanup fails", async () => {
    const canonicalForget = vi.fn(async () => {});
    const cosine = fakeBackend("cosine", [], { get: async () => e("1"), forget: canonicalForget });
    const lexical = fakeBackend("lexical", [], { list: async () => { throw new Error("projection unavailable"); } });
    const hy = new HybridBackend(lexical, cosine);
    await expect(hy.forget("1")).rejects.toThrow("projection unavailable");
    expect(canonicalForget).not.toHaveBeenCalled();
  });

  it("write-throughs the projection into the lexical leg (best-effort), still returns cosine ids", async () => {
    const cosineRemember = vi.fn(async () => ({ ids: ["c1"] }));
    const lexicalRemember = vi.fn(async () => ({ ids: ["l1"] }));
    const cosine = fakeBackend("cosine", [], { remember: cosineRemember });
    const lexical = fakeBackend("lexical", [], { remember: lexicalRemember });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.remember("the marker token kestrel-7", { scope: "s" });
    expect(cosineRemember).toHaveBeenCalledOnce();
    expect(lexicalRemember).toHaveBeenCalledOnce(); // projected to the native surface
    expect(out.ids).toEqual(["c1"]); // canonical store owns ids
  });

  it("a throwing lexical projection is non-fatal — the canonical write still succeeds", async () => {
    const cosine = fakeBackend("cosine", [], {
      remember: async () => ({ ids: ["c1"] }),
    });
    const lexical = fakeBackend("lexical", [], {
      remember: async () => {
        throw new Error("no ~/.claude dir");
      },
    });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.remember("fact", { scope: "s" });
    expect(out.ids).toEqual(["c1"]);
  });

  it("write-through stamps link_id = canonical id onto the lexical projection (for cross-leg dedup)", async () => {
    const cosine = fakeBackend("cosine", [], {
      remember: async () => ({ ids: ["canon-9"] }),
    });
    const lexRemember = vi.fn(
      async (_text: string, _opts: RememberOptions) => ({ ids: ["l"] }),
    );
    const lexical = fakeBackend("lexical", [], { remember: lexRemember });
    const hy = new HybridBackend(lexical, cosine);
    await hy.remember("a fact", { scope: "s", metadata: { kind: "x" } });
    // lexical write carries the original metadata PLUS link_id = the canonical id.
    expect(lexRemember.mock.calls[0]?.[1]?.metadata).toMatchObject({
      kind: "x",
      link_id: "canon-9",
    });
  });

  it("passes the FP floor through to the cosine leg", async () => {
    const search = vi.fn(async (_q: string, _opts: SearchOptions) => [
      e("a", 0.6),
    ]);
    const cosine = fakeBackend("cosine", [], { search });
    const hy = new HybridBackend(fakeBackend("lexical", []), cosine);
    await hy.search("q", {
      scope: "s",
      limit: 6,
      fusionMode: "floored-union",
      minScore: 0.42,
    });
    expect(search.mock.calls[0]?.[1]).toMatchObject({ minScore: 0.42 });
  });

  it("cross-backend by construction: a remember via one client is recallable via another (P-022 / D-002)", async () => {
    // One shared canonical (cosine) store; two HybridBackends over it model two
    // different MCP clients (e.g. claude-su + codex-su) hitting the same operator.
    const store: MemoryEntry[] = [];
    const canonical: MemoryBackend = {
      name: "canonical",
      available: async () => ({ ok: true }),
      remember: async (text, opts) => {
        const id = `c${store.length}`;
        store.push({ id, text, scope: opts.scope as string, score: 0.7 });
        return { ids: [id] };
      },
      search: async (q) => store.filter((e) => e.text.includes(q)),
      list: async () => store,
      get: async (id) => store.find((e) => e.id === id) ?? null,
      forget: async () => {},
      update: async () => {},
    };
    const clientA = new HybridBackend(fakeBackend("lexical", []), canonical);
    const clientB = new HybridBackend(fakeBackend("lexical", []), canonical);

    await clientA.remember("the user prefers nuqs", { scope: "s" });
    // Client B (a different client instance) recalls A's write from the one store.
    const recalled = await clientB.search("nuqs", {
      scope: "s",
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(recalled.map((r) => r.text)).toEqual(["the user prefers nuqs"]);
  });

  // ── GAP 6: available() is a PROBE, not a writability guarantee ──────────────
  // available() forwards to cosine.available() (hybrid-backend.ts:69). A green
  // probe is NOT a promise that remember succeeds: the cosine leg can report
  // available={ok:true} (e.g. PG reachable, schema present) yet its embedder is
  // dead so remember THROWS. This pins the GAP-1 root at the backend layer — the
  // documented contract that available() does not gate writes. (No defect: the
  // probe is non-throwing by design; callers must still handle a write throw.)
  it("available() GREEN does NOT guarantee remember succeeds (false-green probe)", async () => {
    const cosine = fakeBackend("cosine", [], {
      available: async () => ({ ok: true }),
      remember: async () => {
        throw new MemoryUnavailableError("embedder_dead");
      },
    });
    const hy = new HybridBackend(fakeBackend("lexical", []), cosine);
    // The probe reports GREEN...
    expect(await hy.available()).toEqual({ ok: true });
    // ...yet the very next write THROWS — green probe ≠ writable backend.
    await expect(hy.remember("fact", { scope: "s" })).rejects.toBeInstanceOf(
      MemoryUnavailableError,
    );
  });

  it("available() forwards the cosine leg verbatim (ok:false + reason passes through)", async () => {
    const cosine = fakeBackend("cosine", [], {
      available: async () => ({ ok: false, reason: "pg_unreachable" }),
    });
    const hy = new HybridBackend(fakeBackend("lexical", []), cosine);
    // available() tracks the cosine leg (the write target / source of truth);
    // the lexical leg's health is irrelevant to the probe.
    expect(await hy.available()).toEqual({
      ok: false,
      reason: "pg_unreachable",
    });
  });

  // ── GAP 12: NoopBackend as the cosine leg ──────────────────────────────────
  // A hybrid whose cosine leg is the deliberate "no store": available is
  // {ok:false}, search returns [] (NOT a throw — the dedup/search path never
  // probes available, so the Noop's empty search degrades cleanly), but remember
  // THROWS MemoryUnavailableError (no caller is lied to about a fact storing).
  describe("NoopBackend as the cosine leg (memory OFF)", () => {
    it("available() reports {ok:false} with the disabled reason", async () => {
      const hy = new HybridBackend(
        fakeBackend("lexical", []),
        new NoopBackend(),
      );
      expect(await hy.available()).toEqual({
        ok: false,
        reason: NOOP_DISABLED_REASON,
      });
    });

    it("search returns [] (does NOT throw — bypasses the availability probe)", async () => {
      const hy = new HybridBackend(
        fakeBackend("lexical", []),
        new NoopBackend(),
      );
      // floored-union default: cosine is [] but NOT cosine-gated, so the lexical
      // leg still runs; with an empty lexical leg the fused result is [].
      await expect(
        hy.search("anything", {
          scope: "s",
          fusionMode: "floored-union",
          minScore: 0.45,
        }),
      ).resolves.toEqual([]);
    });

    it("remember THROWS MemoryUnavailableError — the write is NOT silently dropped", async () => {
      const hy = new HybridBackend(
        fakeBackend("lexical", []),
        new NoopBackend(),
      );
      await expect(hy.remember("fact", { scope: "s" })).rejects.toBeInstanceOf(
        MemoryUnavailableError,
      );
    });
  });

  // ── GAP 13: per-call fusionMode override + cosine-gated early-return ────────
  // A hybrid constructed floored-union, then CALLED with {fusionMode:'cosine-gated'}:
  // the per-call override beats the constructor default (hybrid-backend.ts:112),
  // and with an empty cosine set the cosine-gated branch early-returns []
  // (hybrid-backend.ts:118) BEFORE the lexical leg can admit anything — so a
  // strong lexical-only identifier hit is NOT admitted (override wins).
  it("per-call fusionMode:cosine-gated beats a floored-union constructor + floors a strong lexical-only hit to []", async () => {
    const cosine = fakeBackend("cosine", []); // floored empty
    const lexical = fakeBackend("lexical", [e("CODEX_HOME", 1.0)]); // strong identifier hit
    // Constructor says floored-union (which WOULD admit the strong lexical hit)...
    const hy = new HybridBackend(lexical, cosine, {
      fusionMode: "floored-union",
    });
    // ...but the per-call override flips to cosine-gated → empty cosine early-returns [].
    expect(
      await hy.search("CODEX_HOME", {
        scope: "s",
        minScore: 0.45,
        fusionMode: "cosine-gated",
      }),
    ).toEqual([]);
    // Control: with NO override the constructor's floored-union DOES admit it.
    expect(
      (
        await hy.search("CODEX_HOME", {
          scope: "s",
          fusionMode: "floored-union",
          minScore: 0.45,
        })
      ).map((x) => x.id),
    ).toEqual(["CODEX_HOME"]);
  });

  it("cosine-gated early-return short-circuits the lexical leg entirely (it is never searched)", async () => {
    const cosine = fakeBackend("cosine", []); // floored empty
    const lexSearch = vi.fn(async () => [e("CODEX_HOME", 1.0)]);
    const lexical = fakeBackend("lexical", [], { search: lexSearch });
    const hy = new HybridBackend(lexical, cosine, {
      fusionMode: "cosine-gated",
    });
    // Mirrors the constructor above — the early-return under test is cosine-gated's.
    expect(
      await hy.search("CODEX_HOME", {
        scope: "s",
        fusionMode: "cosine-gated",
        minScore: 0.45,
      }),
    ).toEqual([]);
    // The early-return fires BEFORE the lexical search — no wasted lexical probe.
    expect(lexSearch).not.toHaveBeenCalled();
  });

  describe("onLegStats — per-leg reporting (P-002)", () => {
    it("reports the short-circuited lexical leg as ran:false, NOT as zero candidates", async () => {
      // THE distinction this seam exists for. The lexical leg contributing
      // nothing has two causes with OPPOSITE fixes: it ran and found nothing
      // (a tokenization/identifier problem), or it never ran at all (contracted
      // cosine-gated behaviour, nothing wrong). Collapsing both to "0" sends a
      // reader to fix a leg that was never asked a question.
      const cosine = fakeBackend("cosine", []);
      const lexSearch = vi.fn(async () => [e("CODEX_HOME", 1.0)]);
      const lexical = fakeBackend("lexical", [], { search: lexSearch });
      const hy = new HybridBackend(lexical, cosine, {
        fusionMode: "cosine-gated",
      });
      const seen: unknown[] = [];
      await hy.search("CODEX_HOME", {
        scope: "s",
        fusionMode: "cosine-gated",
        minScore: 0.45,
        onLegStats: (s) => seen.push(s),
      });
      expect(lexSearch).not.toHaveBeenCalled();
      expect(seen).toEqual([
        {
          mode: "cosine-gated",
          cosine: {
            ran: true,
            candidates: 0,
            qualifying: 0,
            durationMs: expect.any(Number),
          },
          lexical: { ran: false },
          fused: 0,
          totalMs: expect.any(Number),
        },
      ]);
      expect(seen[0]).not.toHaveProperty("fusionMs");
    });

    it("records each leg its OWN depth — the lexical leg pulls 3x the cosine budget", async () => {
      // Sharing one budget across both legs would misjudge the other leg's
      // saturation by exactly that factor, and saturation is what says whether
      // a pool was under-sampled rather than empty.
      const cosine = fakeBackend("cosine", [e("a", 0.9)]);
      const lexical = fakeBackend("lexical", [e("a", 0.9)]);
      const hy = new HybridBackend(lexical, cosine, {
        fusionMode: "floored-union",
      });
      let seen: {
        cosine: { depth?: number };
        lexical: { depth?: number };
      } | null = null;
      await hy.search("q", {
        scope: "s",
        limit: 4,
        fusionMode: "floored-union",
        minScore: 0.45,
        onLegStats: (s) => {
          seen = s;
        },
      });
      expect(seen!.cosine.depth).toBe(4);
      expect(seen!.lexical.depth).toBe(12); // (limit ?? 6) * 3
    });

    it("reports per-leg, fusion, and end-to-end durations on a fused call", async () => {
      const cosine = fakeBackend("cosine", [e("a", 0.9)]);
      const lexical = fakeBackend("lexical", [e("a", 0.9)]);
      const hy = new HybridBackend(lexical, cosine, {
        fusionMode: "floored-union",
      });
      let seen: SearchLegStats | null = null;
      await hy.search("q", {
        scope: "s",
        limit: 4,
        fusionMode: "floored-union",
        minScore: 0.45,
        onLegStats: (stats) => {
          seen = stats;
        },
      });

      expect(seen).toMatchObject({
        cosine: { durationMs: expect.any(Number) },
        lexical: { durationMs: expect.any(Number) },
        fusionMs: expect.any(Number),
        totalMs: expect.any(Number),
      });
      expect(seen!.cosine.durationMs).toBeGreaterThanOrEqual(0);
      expect(seen!.lexical.durationMs).toBeGreaterThanOrEqual(0);
      expect(seen!.fusionMs).toBeGreaterThanOrEqual(0);
      expect(seen!.totalMs).toBeGreaterThanOrEqual(seen!.fusionMs!);
      expect(seen!.totalMs).toBeGreaterThanOrEqual(seen!.cosine.durationMs!);
      expect(seen!.totalMs).toBeGreaterThanOrEqual(seen!.lexical.durationMs!);
    });

    it("a throwing reporter never fails the search", async () => {
      const cosine = fakeBackend("cosine", [e("a", 0.9)]);
      const lexical = fakeBackend("lexical", []);
      const hy = new HybridBackend(lexical, cosine, {
        fusionMode: "floored-union",
      });
      const out = await hy.search("q", {
        scope: "s",
        fusionMode: "floored-union",
        minScore: 0.45,
        onLegStats: () => {
          throw new Error("reporter exploded");
        },
      });
      expect(out.map((x) => x.id)).toEqual(["a"]);
    });

    it("does not forward the reporter to the cosine leg — it would report half as the whole", async () => {
      // Params declared so the mock's call tuple is typed — `vi.fn(async () => …)`
      // infers `[]`, and indexing [1] on it is a compile error, not a runtime one.
      const cosineSearch = vi.fn(async (_q: string, _opts: SearchOptions) => [
        e("a", 0.9),
      ]);
      const cosine = fakeBackend("cosine", [], { search: cosineSearch });
      const lexical = fakeBackend("lexical", []);
      const hy = new HybridBackend(lexical, cosine, {
        fusionMode: "floored-union",
      });
      await hy.search("q", {
        scope: "s",
        fusionMode: "floored-union",
        minScore: 0.45,
        onLegStats: () => {},
      });
      expect(cosineSearch.mock.calls[0]![1]).not.toHaveProperty("onLegStats");
    });
  });

  /**
   * Recurrence guard: in floored-union mode the two legs MUST run CONCURRENTLY.
   *
   * They are independent (two rankings of the same rows — neither's input depends on the
   * other's output), so awaiting them in sequence puts the lexical leg's whole cost on the
   * critical path: total = cosine + lexical instead of max(cosine, lexical). That was free
   * to ignore while the lexical leg was a local file scan, and became the DOMINANT cost the
   * moment the leg became a PG query — most of why `hybrid-pg` benched at p50 1182ms against
   * `hybrid`'s 821ms (memory-pg-lexical-own-injection-2026-07-13 P-006 run 8).
   *
   * A future refactor that innocently re-writes this back to `await cosine; await lexical;`
   * would silently restore the regression with every test still green — so pin the overlap
   * itself: the lexical leg must be STARTED before the cosine leg has resolved.
   */
  it("floored-union runs BOTH legs concurrently (the lexical leg is not on the critical path)", async () => {
    let cosineResolved = false;
    let lexicalStartedBeforeCosineResolved = false;

    const cosine = fakeBackend("cosine", [], {
      search: vi.fn(async () => {
        // Yield to the microtask queue so a concurrently-started lexical leg can run.
        await new Promise((r) => setTimeout(r, 10));
        cosineResolved = true;
        return [e("cosine-hit", 0.9)];
      }),
    });
    const lexical = fakeBackend("lexical", [], {
      search: vi.fn(async () => {
        lexicalStartedBeforeCosineResolved = !cosineResolved;
        return [e("CODEX_HOME", 1.0)];
      }),
    });

    const hy = new HybridBackend(lexical, cosine, {
      fusionMode: "floored-union",
    });
    const hits = await hy.search("CODEX_HOME", { scope: "s" });

    // The overlap itself — the thing a re-serializing refactor would break.
    expect(lexicalStartedBeforeCosineResolved).toBe(true);
    // ...and both legs still actually contribute to the fused result.
    expect(hits.map((h) => h.text)).toEqual(
      expect.arrayContaining(["CODEX_HOME", "cosine-hit"]),
    );
  });
});

describe("HybridBackend.embedQuery (EI-12992 shared-embed delegation)", () => {
  it("exposes the COSINE leg capability and delegates to it — never the lexical leg", async () => {
    const cosineEmbed = vi.fn(async () => [0.1, 0.2]);
    const lexEmbed = vi.fn(async () => [9, 9]);
    const hy = new HybridBackend(
      fakeBackend("lexical", [], { embedQuery: lexEmbed }),
      fakeBackend("cosine", [], { embedQuery: cosineEmbed }),
    );
    expect(await hy.embedQuery?.("q")).toEqual([0.1, 0.2]);
    expect(cosineEmbed).toHaveBeenCalledWith("q");
    // The vector MUST come from the leg that CONSUMES it: search() forwards
    // opts.vector to the cosine leg, so a lexical-leg vector would be compared
    // across embedding spaces and return plausible garbage instead of failing.
    expect(lexEmbed).not.toHaveBeenCalled();
  });

  it("is UNDEFINED when the cosine leg cannot embed — the capability feature-test stays honest", () => {
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", []),
    );
    expect(hy.embedQuery).toBeUndefined();
  });

  it("preserves the cosine leg's `this` binding (delegation is bound, not a bare fn ref)", async () => {
    const cosine = fakeBackend("cosine", []) as MemoryBackend & {
      model?: string;
    };
    cosine.model = "bge-small";
    cosine.embedQuery = async function (this: { model?: string }) {
      return this.model === "bge-small" ? [1] : null;
    };
    const hy = new HybridBackend(fakeBackend("lexical", []), cosine);
    expect(await hy.embedQuery?.("q")).toEqual([1]);
  });

  it("forwards a precomputed vector to the cosine leg, and never leaks it to the lexical leg", async () => {
    const cosineSearch = vi.fn(async (_q: string, _o: SearchOptions) => [
      e("a", 0.6),
    ]);
    const lexSearch = vi.fn(async (_q: string, _o: SearchOptions) => []);
    const hy = new HybridBackend(
      fakeBackend("lexical", [], { search: lexSearch }),
      fakeBackend("cosine", [], { search: cosineSearch }),
    );
    await hy.search("q", {
      scope: "s",
      limit: 6,
      fusionMode: "floored-union",
      minScore: 0.45,
      vector: [0.3, 0.4],
    });
    expect(cosineSearch.mock.calls[0]?.[1]).toMatchObject({
      vector: [0.3, 0.4],
    });
    // The lexical leg is embed-free by construction — it gets scope+limit only.
    expect(lexSearch.mock.calls[0]?.[1]).not.toHaveProperty("vector");
  });

  it("PRODUCTION COMPOSITION (hybrid-pg) exposes the capability — the seam that actually broke", () => {
    // Mirrors configure.ts's `hybrid-pg` registration EXACTLY:
    //   const mem0 = new Mem0Backend();
    //   new HybridBackend(new LexicalLegBackend(mem0), mem0)
    // Both constructors are inert (field assignment + a capability check — no PG,
    // no network), so this stays hermetic.
    //
    // Why this test and not just the fake-leg ones above: the ORIGINAL defect was
    // invisible to every fake-based test AND to the bare-Mem0Backend tests in
    // mem0-backend.test.ts, because the capability existed on the leg and only
    // disappeared under COMPOSITION. A capability assertion against the real
    // production wiring is the only shape that catches "a wrapper silently drops
    // a leg's optional capability" — for this wrapper and any future one.
    const mem0 = new Mem0Backend();
    const hy = new HybridBackend(new LexicalLegBackend(mem0), mem0);
    expect(typeof hy.embedQuery).toBe("function");
  });

  it("REGRESSION (production shape): 3 pulls over one query embed ONCE, not 3×", async () => {
    // Mirrors buildMemoryContextBlock's fan-out (user/harness/hive pools over the
    // SAME query text) against a cosine leg that embeds per-search unless handed a
    // vector — exactly Mem0Backend's `opts.vector ?? await this.embedQuery(query)`.
    // Before this delegation the wrapper hid embedQuery, so `hybrid-pg` — the
    // PRODUCTION backend — silently paid all 3 embeds and the fix was a no-op.
    let embeds = 0;
    const cosine = fakeBackend("cosine", [], {
      embedQuery: async () => {
        embeds += 1;
        return [0.5];
      },
      search: async (_q: string, o: SearchOptions) => {
        if (!o.vector) embeds += 1; // the leg had to embed for itself
        return [e("a", 0.6)];
      },
    });
    const hy = new HybridBackend(fakeBackend("lexical", []), cosine);

    // Deliberately NO assertion on `shared` here: the embed COUNT below must be
    // what discriminates, so a regression fails with "expected 3 to be 1" (the
    // actual defect) rather than tripping an earlier assertion that never
    // exercises the fan-out. The `?.` + conditional spread mirror injection.ts's
    // real feature-test/fallback, so this stays a faithful simulation on both sides.
    const shared = await hy.embedQuery?.("the same query text");
    for (const scope of ["user", "harness:papercusp", "hive:pc"]) {
      await hy.search("the same query text", {
        scope,
        limit: 3,
        fusionMode: "floored-union",
        minScore: 0.45,
        ...(shared ? { vector: shared } : {}),
      });
    }
    expect(embeds).toBe(1);
  });
});

describe("HybridBackend searchLexical (WI-4214 embed-free fallback)", () => {
  it("prefers the cosine leg's lexical capability (canonical store covers ALL memories)", async () => {
    const cosineLex = vi.fn(async () => [e("canonical-hit", 0.5)]);
    const cosine = fakeBackend("cosine", [], {
      searchLexical: cosineLex,
    } as Partial<MemoryBackend>);
    const lexSearch = vi.fn(async () => [e("projected-hit", 1)]);
    const lexical = fakeBackend("lexical", [], { search: lexSearch });
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.searchLexical("q", { scope: "s", limit: 3 });
    expect(out.map((x) => x.id)).toEqual(["canonical-hit"]);
    expect(cosineLex).toHaveBeenCalledWith("q", { scope: "s", limit: 3 });
    expect(lexSearch).not.toHaveBeenCalled();
  });

  it("falls back to the native lexical leg when the cosine leg lacks the capability", async () => {
    const cosine = fakeBackend("cosine", []);
    const lexical = fakeBackend("lexical", [e("projected-hit", 1)]);
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.searchLexical("q", { scope: "s", limit: 3 });
    expect(out.map((x) => x.id)).toEqual(["projected-hit"]);
  });

  it("falls back to the native lexical leg when the cosine lexical THROWS (degraded path stays serving)", async () => {
    const cosine = fakeBackend("cosine", [], {
      searchLexical: vi.fn(async () => {
        throw new Error("pg down");
      }),
    } as Partial<MemoryBackend>);
    const lexical = fakeBackend("lexical", [e("projected-hit", 1)]);
    const hy = new HybridBackend(lexical, cosine);
    expect(
      (await hy.searchLexical("q", { scope: "s" })).map((x) => x.id),
    ).toEqual(["projected-hit"]);
  });
});

describe("HybridBackend.invalidateEntry (temporal-lite lifecycle delegation)", () => {
  it("delegates to the COSINE leg (the canonical store owns lifecycle) with the supersededBy option", async () => {
    const inv = vi.fn(async () => true);
    const cosine = fakeBackend("cosine", [], { invalidateEntry: inv });
    const hybrid = new HybridBackend(fakeBackend("lex", []), cosine);
    await expect(
      hybrid.invalidateEntry("old", { supersededBy: "new" }),
    ).resolves.toBe(true);
    expect(inv).toHaveBeenCalledWith("old", { supersededBy: "new" });
  });

  it('a cosine leg WITHOUT the capability throws — a false would read as "not found" upstream', () => {
    const hybrid = new HybridBackend(
      fakeBackend("lex", []),
      fakeBackend("cosine", []),
    );
    expect(() => hybrid.invalidateEntry("old")).toThrow(/validity-window/);
  });
});

// ---------------------------------------------------------------------------
// Diversity re-rank over the FUSED list (context-injection-audit-2026-07-28
// P-034 / F-D, D-014). Before this, `diversify` reached HybridBackend and was
// forwarded verbatim to the cosine leg — where it reordered the very ranking
// fuse() consumes as RANK, instead of diversifying the output.
// ---------------------------------------------------------------------------
describe("HybridBackend.search — diversity re-rank (F-D / D-014)", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  /** Entry with REAL text — the similarity proxy is trigram-Jaccard over text. */
  const t = (id: string, text: string, score: number): MemoryEntry => ({
    id,
    text,
    scope: "s",
    score,
  });

  // 'dup' is a near-verbatim paraphrase of 'best'; 'distinct' shares almost no
  // trigrams with either. Pure relevance ranks best > dup > distinct.
  const nearDup = [
    t(
      "best",
      "the deploy pipeline uses a two port model for staging and release",
      0.95,
    ),
    t(
      "dup",
      "the deploy pipeline uses a two-port model for staging vs release",
      0.94,
    ),
    t("distinct", "git-sync owns commit and push for the shared tree", 0.6),
  ];

  it("does NOT forward `diversify` to the cosine leg — that would re-rank the fusion INPUT", async () => {
    // The bug this pass exists to prevent: Mem0Backend honors `diversify` by
    // reordering its own results, and fuse() scores each hit by its rank within
    // that leg. Forwarding it silently rewrites the RRF input.
    let seen: SearchOptions | undefined;
    const cosine = fakeBackend("cosine", nearDup, {
      search: async (_q: string, o: SearchOptions) => {
        seen = o;
        return nearDup;
      },
    } as Partial<MemoryBackend>);
    const hy = new HybridBackend(fakeBackend("lexical", []), cosine);
    await hy.search("q", {
      scope: "s",
      limit: 2,
      fusionMode: "floored-union",
      minScore: 0.45,
      diversify: { lambda: 0.5 },
    });
    expect(seen?.diversify).toBeUndefined();
    // …while every OTHER option still passes through untouched.
    expect(seen?.minScore).toBe(0.45);
    expect(seen?.limit).toBe(2);
    expect(seen?.scope).toBe("s");
  });

  it("is an exact no-op when `diversify` is omitted", async () => {
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", nearDup),
    );
    const out = await hy.search("q", { scope: "s", limit: 3 });
    expect(out.map((x) => x.id)).toEqual(["best", "dup", "distinct"]);
  });

  it("re-ranks BEFORE the limit slice, so a near-duplicate loses its slot to a distinct fact", async () => {
    // The load-bearing assertion of F-D. Applied AFTER the slice this test would
    // fail: the top-2 would already be [best, dup] and reordering two
    // near-copies reclaims nothing.
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", nearDup),
    );
    const plain = await hy.search("q", { scope: "s", limit: 2 });
    expect(plain.map((x) => x.id)).toEqual(["best", "dup"]);

    const diverse = await hy.search("q", {
      scope: "s",
      limit: 2,
      diversify: { lambda: 0.5 },
    });
    expect(diverse.map((x) => x.id)).toEqual(["best", "distinct"]);
  });

  it("never grows the block — the limit is still the one ceiling", async () => {
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", nearDup),
    );
    const out = await hy.search("q", {
      scope: "s",
      limit: 2,
      diversify: { lambda: 0 },
    });
    expect(out).toHaveLength(2);
  });

  it("drops nothing when the caller sets no limit — the un-reranked tail is preserved", async () => {
    const many = [
      ...nearDup,
      t("tail", "an unrelated fact about pgbouncer pooling", 0.5),
    ];
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", many),
    );
    const out = await hy.search("q", {
      scope: "s",
      diversify: { lambda: 0.5 },
    });
    expect(out.map((x) => x.id).sort()).toEqual([
      "best",
      "distinct",
      "dup",
      "tail",
    ]);
  });

  it("PAPERCUSP_MEMORY_MMR=0 kill-switches it here too, not just in Mem0Backend", async () => {
    // The reason the switch moved into diversity-rerank.ts: a per-backend copy
    // would have covered only half the seams that apply the pass.
    vi.stubEnv("PAPERCUSP_MEMORY_MMR", "0");
    const hy = new HybridBackend(
      fakeBackend("lexical", []),
      fakeBackend("cosine", nearDup),
    );
    const out = await hy.search("q", {
      scope: "s",
      limit: 2,
      diversify: { lambda: 0.5 },
    });
    expect(out.map((x) => x.id)).toEqual(["best", "dup"]);
  });

  it("re-ranks only the admitted set — a cosine-gated empty result stays empty", async () => {
    const hy = new HybridBackend(
      fakeBackend("lexical", [e("lex-only", 9)]),
      fakeBackend("cosine", []),
    );
    const out = await hy.search("q", {
      scope: "s",
      limit: 4,
      fusionMode: "cosine-gated",
      diversify: { lambda: 0.5 },
    });
    expect(out).toEqual([]);
  });
});

describe("score-scale declaration (context-injection-audit-2026-07-28 P-036)", () => {
  // Not decoration: recall telemetry stamps these onto every stats row so a
  // reader can refuse to compare a fused RRF score (bounded by 2/61 = 0.0328)
  // against a cosine one (floored at ~0.45). A WRONG declaration is worse than
  // none — it would license exactly the cross-scale comparison the label exists
  // to prevent — so pin both, and pin that they DIFFER.
  it("declares rrf for search() and lexical for searchLexical()", () => {
    const hy = new HybridBackend(new NoopBackend(), new NoopBackend());
    expect(hy.scoreScale).toBe("rrf");
    expect(hy.lexicalScoreScale).toBe("lexical");
  });

  it("declares rrf regardless of fusion mode — search() has no non-fused exit", () => {
    // `fuse()` recomputes `score` for every surviving candidate, and the only
    // other way out of search() is cosine-gated's empty early-return. So the
    // scale is a constant here; if a future mode ever returned a leg's own
    // scores un-fused, this pin is what makes that a visible decision.
    const gated = new HybridBackend(new NoopBackend(), new NoopBackend(), {
      fusionMode: "cosine-gated",
    });
    expect(gated.scoreScale).toBe("rrf");
  });

  it("the cosine and lexical legs declare the scales the hybrid composes from", () => {
    // The hybrid's 'rrf' is not inherited from either leg — it is produced by
    // the fusion. Pinning the legs' own scales keeps that distinction honest.
    expect(
      new Mem0Backend({ getClient: async () => ({}) as never }).scoreScale,
    ).toBe("cosine");
    expect(new NoopBackend().scoreScale).toBe("unknown");
  });
});

describe("per-leg queries — SearchOptions.lexicalQuery (context-injection-audit-2026-07-28 P-044)", () => {
  /** Records the exact text each leg was asked for. */
  function spyLegs(): {
    lexical: MemoryBackend;
    cosine: MemoryBackend;
    asked: { lexical: string[]; cosine: string[] };
    cosineOpts: SearchOptions[];
  } {
    const asked = { lexical: [] as string[], cosine: [] as string[] };
    const cosineOpts: SearchOptions[] = [];
    const lexical = fakeBackend("lexical", [], {
      search: async (q: string) => {
        asked.lexical.push(q);
        return [e("lex-hit", 0.9)];
      },
    });
    const cosine = fakeBackend("cosine", [], {
      search: async (q: string, opts: SearchOptions) => {
        asked.cosine.push(q);
        cosineOpts.push(opts);
        return [e("cos-hit", 0.8)];
      },
    });
    return { lexical, cosine, asked, cosineOpts };
  }

  it("sends lexicalQuery to the LEXICAL leg and query to the COSINE leg", async () => {
    const { lexical, cosine, asked } = spyLegs();
    const hy = new HybridBackend(lexical, cosine);
    await hy.search("dropdown selection broken", {
      scope: "s",
      lexicalQuery: "dropdown selection broken apps/operator-vite/src WI-6512",
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(asked.cosine).toEqual(["dropdown selection broken"]);
    expect(asked.lexical).toEqual([
      "dropdown selection broken apps/operator-vite/src WI-6512",
    ]);
  });

  it("omitted ⇒ BOTH legs get `query` — the pre-P-044 behaviour, unchanged", async () => {
    const { lexical, cosine, asked } = spyLegs();
    const hy = new HybridBackend(lexical, cosine);
    await hy.search("deploy pipeline", {
      scope: "s",
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(asked.cosine).toEqual(["deploy pipeline"]);
    expect(asked.lexical).toEqual(["deploy pipeline"]);
  });

  it("a blank/whitespace override falls back to `query` rather than asking for nothing", async () => {
    // An empty lexical query returns zero tokens and therefore zero hits, which
    // would silently delete the whole leg — a caller that composed an empty
    // string meant "I have nothing to add", not "search for nothing".
    const { lexical, cosine, asked } = spyLegs();
    const hy = new HybridBackend(lexical, cosine);
    await hy.search("deploy pipeline", {
      scope: "s",
      lexicalQuery: "   ",
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(asked.lexical).toEqual(["deploy pipeline"]);
  });

  it("STRIPS lexicalQuery from what the cosine leg receives", async () => {
    // Same class as `diversify` (D-014), and the reason is the same: an option
    // naming a DIFFERENT query text has no honest meaning on a leg that already
    // took its query as an argument. A leg that later learned to read it would
    // embed the identifier-bearing string this option exists to keep OUT of the
    // vector — the exact dilution D-041 measured.
    const { lexical, cosine, cosineOpts } = spyLegs();
    const hy = new HybridBackend(lexical, cosine);
    await hy.search("q", {
      scope: "s",
      lexicalQuery: "q libs/generic/memory/src/hybrid-backend.ts",
      fusionMode: "floored-union",
      minScore: 0.45,
    });
    expect(cosineOpts[0]).not.toHaveProperty("lexicalQuery");
  });

  it("routes in cosine-gated mode too — where the leg RE-RANKS rather than admits", async () => {
    // The operator's push path runs cosine-gated (injectFusionMode), so this is
    // the mode that actually ships. The lexical leg still runs and still gets
    // its own query; what it cannot do here is admit a row the cosine leg
    // missed. Pinning the routing separately from the admission keeps that
    // distinction visible instead of implied.
    const { lexical, cosine, asked } = spyLegs();
    const hy = new HybridBackend(lexical, cosine);
    const out = await hy.search("symptom in plain words", {
      scope: "s",
      lexicalQuery: "symptom in plain words WI-6512",
      fusionMode: "cosine-gated",
      minScore: 0.45,
    });
    expect(asked.lexical).toEqual(["symptom in plain words WI-6512"]);
    expect(out.map((x) => x.id)).toEqual(["cos-hit"]);
  });

  it("does not start the lexical leg at all when cosine-gated returns nothing", async () => {
    // The gated short-circuit is a cost contract (see search()'s comment); a
    // per-leg query must not become a reason to spend that work.
    const { lexical, asked } = spyLegs();
    const cosine = fakeBackend("cosine", []);
    const hy = new HybridBackend(lexical, cosine);
    expect(
      await hy.search("q", {
        scope: "s",
        lexicalQuery: "q WI-6512",
        fusionMode: "cosine-gated",
        minScore: 0.45,
      }),
    ).toEqual([]);
    expect(asked.lexical).toEqual([]);
  });
});
