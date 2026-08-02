import { describe, it, expect } from 'vitest';
import { fuse, fuseCosineGated } from './hybrid-fusion';
import type { MemoryEntry } from './backend';

const e = (id: string, score?: number): MemoryEntry => ({
  id,
  text: id,
  scope: 's',
  ...(score !== undefined ? { score } : {}),
});

describe('fuseCosineGated (P-020)', () => {
  it('returns empty when the cosine gate is empty (hard-negative discipline)', () => {
    expect(fuseCosineGated([], [e('x'), e('y')])).toEqual([]);
  });

  it('keeps only cosine-gated entries — lexical-only hits are excluded', () => {
    const out = fuseCosineGated([e('a')], [e('a'), e('b')]); // b is lexical-only
    expect(out.map((x) => x.id)).toEqual(['a']);
  });

  it('lexical re-rank lifts an exact match above a cosine-only paraphrase', () => {
    // cosine order: [para(rank1), exact(rank2)]; lexical: exact is rank1.
    // exact = 1/62 + 1/61 > para = 1/61 → exact wins.
    const out = fuseCosineGated([e('para'), e('exact')], [e('exact')], 60);
    expect(out.map((x) => x.id)).toEqual(['exact', 'para']);
  });

  it('a paraphrase hit (cosine-only, no lexical match) is still returned', () => {
    const out = fuseCosineGated([e('para')], []);
    expect(out.map((x) => x.id)).toEqual(['para']);
    expect(out[0].score).toBeGreaterThan(0);
  });

  it('writes the fused RRF score and orders by it', () => {
    const out = fuseCosineGated([e('a'), e('b')], [e('b')]); // b boosted by lexical
    expect(out.every((x) => typeof x.score === 'number')).toBe(true);
    expect(out.map((x) => x.id)).toEqual(['b', 'a']);
  });
});

describe('fuse — floored-union (P-031)', () => {
  it('admits a strong lexical-ONLY hit (exact-id the cosine leg missed)', () => {
    // cosine has only a paraphrase; lexical has the exact-id target (score 1.0).
    const out = fuse([e('para', 0.55)], [e('exact', 1.0)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id).sort()).toEqual(['exact', 'para']);
  });

  it('does NOT admit a weak lexical-only hit (below minLexScore)', () => {
    const out = fuse([e('para', 0.55)], [e('weak', 0.3)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id)).toEqual(['para']);
  });

  it('hard-negative: empty cosine + only weak lexical overlap → empty', () => {
    const out = fuse([], [e('weak', 0.2)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out).toEqual([]);
  });

  it('a hit in BOTH legs outranks a lexical-only and a cosine-only hit', () => {
    // both: cosRank + lexRank; cosine-only: cosRank; lexical-only(strong): lexRank.
    const out = fuse([e('both', 0.6), e('cosOnly', 0.5)], [e('both', 0.9), e('lexOnly', 0.8)], {
      mode: 'floored-union',
      minLexScore: 0.5,
    });
    expect(out[0].id).toBe('both');
    expect(out.map((x) => x.id).sort()).toEqual(['both', 'cosOnly', 'lexOnly']);
  });

  it('floored-union is the default mode', () => {
    const out = fuse([e('para', 0.55)], [e('exact', 1.0)]); // no mode → floored-union
    expect(out.map((x) => x.id).sort()).toEqual(['exact', 'para']);
  });
});

describe('fuse — cross-leg dedup via metadata.link_id', () => {
  const linked = (id: string, linkId: string, score: number): MemoryEntry => ({
    id, text: id, scope: 's', score, metadata: { link_id: linkId },
  });

  it('a memory in BOTH legs (shared link_id) collapses to ONE result, keeping the canonical entry', () => {
    const cosineHit = e('canon-1', 0.6);                 // canonical id
    const lexicalHit = linked('file_slug', 'canon-1', 0.9); // projection, link_id → canonical
    const out = fuse([cosineHit], [lexicalHit], { mode: 'floored-union' });
    expect(out.length).toBe(1);            // not 2 — deduped across legs
    expect(out[0].id).toBe('canon-1');     // canonical (cosine) entry kept
    expect(out[0].score).toBeCloseTo(1 / 61 + 1 / 61, 6); // got BOTH legs' rank boost
  });

  it('distinct memories (no shared key) are NOT collapsed', () => {
    const out = fuse([e('a', 0.6)], [e('b', 0.9)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id).sort()).toEqual(['a', 'b']); // b admitted (0.9 ≥ 0.5), both kept
  });

  it('a lexical-only projection (link_id to a cosine MISS) is still admitted once', () => {
    // cosine missed it entirely; lexical has it with a link to an absent canonical id.
    const out = fuse([], [linked('file_x', 'canon-absent', 0.8)], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id)).toEqual(['file_x']);
  });

  it('MIGRATED case: same verbatim TEXT, different ids, NO link_id → still dedupes to one', () => {
    // The migration stores a fact verbatim in BOTH legs (cosine PG + ~/.claude file)
    // without mutating the file, so there is no link_id — identical text is the key.
    const cos: MemoryEntry = { id: 'pg-uuid-1', text: 'the user prefers nuqs over useState', scope: 's', score: 0.6 };
    const lex: MemoryEntry = { id: 'file_nuqs', text: '  the user prefers nuqs over useState ', scope: 's', score: 0.9 };
    const out = fuse([cos], [lex], { mode: 'floored-union' });
    expect(out.length).toBe(1);            // deduped on normalized text (whitespace-insensitive)
    expect(out[0].id).toBe('pg-uuid-1');   // canonical (cosine) entry kept
  });

  it('different text is NOT merged by the text key', () => {
    const cos: MemoryEntry = { id: 'x', text: 'fact about redis', scope: 's', score: 0.6 };
    const lex: MemoryEntry = { id: 'y', text: 'fact about postgres', scope: 's', score: 0.9 };
    const out = fuse([cos], [lex], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((e) => e.id).sort()).toEqual(['x', 'y']);
  });

  it('REWORDED cosine text + link_id projection still collapses (live double-surface regression)', () => {
    // mem0 extraction REWORDS the canonical text on write, so the cosine entry's
    // text no longer matches the verbatim projection — the link_id must resolve
    // against the cosine entry's OWN id, not its text. Observed live 2026-06-10:
    // one hybrid write surfaced twice on recall (uuid row + file projection).
    const cos: MemoryEntry = {
      id: '1f534811-uuid',
      text: 'The exporter authenticates using a rotating header, regenerated weekly', // reworded
      scope: 's',
      score: 0.7,
    };
    const lex: MemoryEntry = {
      id: 'reference_exporter_auth_file',
      text: 'The exporter authenticates with the rotating header; the value is regenerated weekly.', // verbatim
      scope: 's',
      score: 0.9,
      metadata: { link_id: '1f534811-uuid' },
    };
    const out = fuse([cos], [lex], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.length).toBe(1);             // ONE result, not the double-surface
    expect(out[0].id).toBe('1f534811-uuid'); // canonical entry kept
    expect(out[0].score).toBeCloseTo(1 / 61 + 1 / 61, 6); // both legs' rank boost
  });

  it('two lexical projections sharing a link_id collapse even when the cosine leg missed', () => {
    const a = linked('file_a', 'canon-z', 0.9);
    const b = linked('file_b', 'canon-z', 0.8);
    const out = fuse([], [a, b], { mode: 'floored-union', minLexScore: 0.5 });
    expect(out.map((x) => x.id)).toEqual(['file_a']); // first (best-ranked) projection wins
  });
});

describe('fuse — minLexScore gates conferring a RANK, not just admission (WI-7154)', () => {
  it('a weak lexical hit no longer confers an RRF bonus under cosine-gated', () => {
    // Under cosine-gated the bar used to be dead code, so ANY lexical hit — however
    // weak — handed its slot a full second RRF term. `weak` scores 0.05, far below the
    // 0.3 default bar, so it must NOT overtake the better cosine candidate.
    const out = fuse([e('better'), e('weak')], [e('weak', 0.05)], { mode: 'cosine-gated' });
    expect(out.map((x) => x.id)).toEqual(['better', 'weak']);
    expect(out[1].score).toBeCloseTo(1 / 62, 6); // cosine term only — no lexical bonus
  });

  it('a genuine identifier match still gets its boost', () => {
    // Real exact-id matches score 0.33–0.96 on this leg, so the bar must not touch them.
    const out = fuse([e('para'), e('exact')], [e('exact', 0.9)], { mode: 'cosine-gated' });
    expect(out.map((x) => x.id)).toEqual(['exact', 'para']);
    expect(out[0].score).toBeCloseTo(1 / 62 + 1 / 61, 6);
  });

  it('THE SCALE INVERSION: a small pool cannot win slots on weak lexical overlap alone', () => {
    // The live shape of WI-7154. The lexical leg pulls `depth` PER SCOPE, so a tiny pool
    // gets most of its corpus into the lexical list on generic token overlap while a large
    // pool gets ~2% — and every one of those weak hits used to outweigh the entire cosine
    // ordering. Measured live: the big pool held 5 of the cosine top-12 and 0 of 12 fused.
    const big = ['b1', 'b2', 'b3', 'b4', 'b5'].map((id) => e(id)); // large pool, no lexical
    const small = ['s1', 's2', 's3'].map((id) => e(id)); // small pool, weak lexical
    const cosine = [big[0], big[1], small[0], big[2], small[1], big[3], small[2], big[4]];
    const weakLexical = small.map((x) => e(x.id, 0.08)); // all below the bar

    const out = fuse(cosine, weakLexical, { mode: 'cosine-gated' });
    // Cosine order must survive: the big pool keeps its leading slots.
    expect(out.slice(0, 2).map((x) => x.id)).toEqual(['b1', 'b2']);
    // and the large pool is not shut out of the top half.
    expect(out.slice(0, 4).filter((x) => x.id.startsWith('b')).length).toBeGreaterThanOrEqual(3);
  });

  it('rank is DENSE over qualifying hits — filtered noise leaves no gap', () => {
    // `real` sits at index 2 of the raw lexical list behind two sub-bar hits. Its bonus
    // must be 1/(60+1), not 1/(60+3) — otherwise the bar's effect on a genuine match
    // would depend on how much noise happened to precede it.
    const out = fuse([e('real')], [e('n1', 0.01), e('n2', 0.02), e('real', 0.8)], {
      mode: 'cosine-gated',
    });
    expect(out[0].score).toBeCloseTo(1 / 61 + 1 / 61, 6);
  });
});
