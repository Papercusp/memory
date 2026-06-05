/**
 * Bench-engine tests (memory-backend-benchmark-2026-06-05 P-003).
 *
 * The engine must measure ANY backend through the seam alone, so the
 * suite drives it three ways:
 *   - a deterministic in-memory lexical double (engine mechanics,
 *     hand-computable rank metrics),
 *   - `NoopBackend` (the D-001 control — every metric must read zero;
 *     a check a no-op "passes" measures nothing),
 *   - `ClaudeFileMemoryBackend` on a temp dir (a real backend
 *     end-to-end; never the live ~/.claude store).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import type {
  ListOptions,
  MemoryAvailability,
  MemoryBackend,
  MemoryEntry,
  RememberOptions,
  SearchOptions,
  UpdatePatch,
} from '../backend';
import { ClaudeFileMemoryBackend } from '../claude-file-backend';
import { NoopBackend } from '../noop-backend';
import {
  aggregateOutcomes,
  latencyStats,
  precisionAtK,
  recallAtK,
  reciprocalRank,
} from './metrics';
import { rankedCorpusKeys, runGoldSet } from './retrieval';
import { distinctiveToken, runRoundtrips } from './roundtrip';
import { renderScorecardMarkdown, rememberP50 } from './scorecard';
import { seedCorpus, unseedCorpus } from './seed';
import { generateSyntheticCorpus } from './synthetic';
import type { BackendScorecard, CorpusEntry, GoldQuery, QueryOutcome } from './types';

/* ── deterministic in-memory lexical double ───────────────────────── */

class LexicalDouble implements MemoryBackend {
  readonly name = 'lexical-double';
  private store = new Map<string, MemoryEntry>();
  private seq = 0;

  async available(): Promise<MemoryAvailability> {
    return { ok: true };
  }

  async remember(text: string, opts: RememberOptions): Promise<{ ids: string[]; storedEvents?: number }> {
    const id = `dbl_${this.seq++}`;
    this.store.set(id, {
      id,
      text,
      kind: opts.kind,
      scope: opts.scope,
      metadata: { ...(opts.metadata ?? {}) },
    });
    return { ids: [id], storedEvents: 1 };
  }

  async search(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    const tokens = [...new Set(query.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length >= 2))];
    const scopes = new Set(Array.isArray(opts.scope) ? opts.scope : [opts.scope]);
    const hits = [...this.store.values()]
      .filter((e) => scopes.has(e.scope))
      .map((e) => ({
        e,
        score: tokens.filter((t) => e.text.toLowerCase().includes(t)).length / Math.max(1, tokens.length),
      }))
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score || a.e.id.localeCompare(b.e.id))
      .slice(0, opts.limit ?? 10);
    return hits.map(({ e, score }) => ({ ...e, score }));
  }

  async list(opts: ListOptions): Promise<MemoryEntry[]> {
    const scopes = new Set(Array.isArray(opts.scope) ? opts.scope : [opts.scope]);
    return [...this.store.values()].filter(
      (e) => scopes.has(e.scope) && (!opts.kind || e.kind === opts.kind),
    );
  }

  async get(id: string): Promise<MemoryEntry | null> {
    return this.store.get(id) ?? null;
  }

  async forget(id: string): Promise<void> {
    this.store.delete(id);
  }

  async update(id: string, patch: UpdatePatch): Promise<void> {
    const e = this.store.get(id);
    if (!e) throw new Error(`not found: ${id}`);
    if (patch.text !== undefined) e.text = patch.text;
    if (patch.metadata) e.metadata = { ...(e.metadata ?? {}), ...patch.metadata };
  }
}

const CORPUS: CorpusEntry[] = [
  { key: 'alpha', text: 'The deploy gate restarts the green host via deploy-cli with rollback.', kind: 'project' },
  { key: 'beta', text: 'Vitest jsdom suites break under the root flag; run from inside the package.', kind: 'feedback' },
  { key: 'gamma', text: 'The staging operator on port 3170 runs from the main checkout.', kind: 'reference' },
];

const GOLD: GoldQuery[] = [
  // exact-identifier: strong lexical overlap with gamma
  { id: 'q1', class: 'exact-identifier', query: 'port 3170 staging operator', expected: ['gamma'] },
  // session-start intent: overlaps alpha
  { id: 'q2', class: 'session-start-intent', query: 'deploy the green host', expected: ['alpha'] },
  // lexical-gap: no token overlap with beta on purpose
  { id: 'q3', class: 'lexical-gap', query: 'unit harness fails when launched with a scoping option', expected: ['beta'] },
  // hard negative: nothing in the corpus matches
  { id: 'q4', class: 'hard-negative', query: 'zebra quantum lighthouse', expected: [] },
];

/* ── metrics math ─────────────────────────────────────────────────── */

describe('rank metrics', () => {
  it('precision/recall/RR hand cases', () => {
    expect(precisionAtK(['a'], ['a', 'b', 'c', 'd', 'e'], 5)).toBeCloseTo(1 / 5);
    expect(precisionAtK(['a', 'b'], ['b', 'x', 'a', 'y', 'z'], 5)).toBeCloseTo(2 / 5);
    expect(recallAtK(['a', 'b'], ['b', 'x', 'a'], 10)).toBeCloseTo(1);
    expect(recallAtK(['a', 'b'], ['b'], 10)).toBeCloseTo(0.5);
    expect(reciprocalRank(['a'], ['x', 'a'])).toBeCloseTo(0.5);
    expect(reciprocalRank(['a'], ['a'])).toBe(1);
    expect(reciprocalRank(['a'], ['x', 'y'])).toBe(0);
    expect(precisionAtK([], ['x'], 5)).toBe(0); // negatives don't enter positive metrics
  });

  it('aggregateOutcomes separates negatives into fpAt5', () => {
    const outcomes: QueryOutcome[] = [
      { queryId: 'p1', class: 'exact-identifier', expected: ['a'], rankedKeys: ['a'], rawHits: 1, ms: 1 },
      { queryId: 'n1', class: 'hard-negative', expected: [], rankedKeys: [], rawHits: 0, ms: 1 },
      { queryId: 'n2', class: 'hard-negative', expected: [], rankedKeys: ['x'], rawHits: 2, ms: 1 },
    ];
    const m = aggregateOutcomes(outcomes);
    expect(m.n).toBe(3);
    expect(m.mrr).toBe(1); // positives only
    expect(m.fpAt5).toBeCloseTo(0.5); // 1 of 2 negatives returned hits
  });

  it('latencyStats percentiles', () => {
    const s = latencyStats([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    expect(s.p50).toBe(50);
    expect(s.p95).toBe(100);
    expect(s.max).toBe(100);
    expect(s.n).toBe(10);
    expect(latencyStats([]).n).toBe(0);
  });
});

/* ── seeding ──────────────────────────────────────────────────────── */

describe('seedCorpus', () => {
  it('stamps corpus_key, fills the manifest, counts chars', async () => {
    const be = new LexicalDouble();
    const manifest = await seedCorpus(be, CORPUS, { scope: 'bench', concurrency: 2 });
    expect(manifest.backend).toBe('lexical-double');
    expect(Object.keys(manifest.ids).sort()).toEqual(['alpha', 'beta', 'gamma']);
    expect(manifest.failed).toEqual([]);
    expect(manifest.totalChars).toBe(CORPUS.reduce((a, c) => a + c.text.length, 0));
    expect(manifest.rememberMs).toHaveLength(3);
    const all = await be.list({ scope: 'bench' });
    expect(all.map((e) => e.metadata?.corpus_key).sort()).toEqual(['alpha', 'beta', 'gamma']);
  });

  it('records honest failures (throwing or zero-event backends)', async () => {
    const noop = new NoopBackend();
    const manifest = await seedCorpus(noop, CORPUS, { scope: 'bench' });
    expect(manifest.failed.sort()).toEqual(['alpha', 'beta', 'gamma']);
    expect(Object.values(manifest.ids).flat()).toEqual([]);
  });

  it('unseedCorpus removes what was created', async () => {
    const be = new LexicalDouble();
    const manifest = await seedCorpus(be, CORPUS, { scope: 'bench' });
    const removed = await unseedCorpus(be, manifest);
    expect(removed).toBe(3);
    expect(await be.list({ scope: 'bench' })).toEqual([]);
  });
});

/* ── retrieval ────────────────────────────────────────────────────── */

describe('runGoldSet', () => {
  it('hand-computable rank metrics on the lexical double', async () => {
    const be = new LexicalDouble();
    await seedCorpus(be, CORPUS, { scope: 'bench' });
    const r = await runGoldSet(be, GOLD, { scope: 'bench', limit: 10 });

    // exact-identifier + session-start hit; lexical-gap misses by design.
    expect(r.byClass['exact-identifier']?.mrr).toBe(1);
    expect(r.byClass['session-start-intent']?.mrr).toBe(1);
    expect(r.byClass['lexical-gap']?.mrr).toBe(0);
    // hard negative returns nothing on a lexical store → clean FP row.
    expect(r.byClass['hard-negative']?.fpAt5).toBe(0);
    // overall = positives only (3 queries, 2 hits at rank 1).
    expect(r.overall.n).toBe(3);
    expect(r.overall.mrr).toBeCloseTo(2 / 3);
    expect(r.latency.n).toBe(4);
  });

  it('NoopBackend control scores zero everywhere', async () => {
    const r = await runGoldSet(new NoopBackend(), GOLD, { scope: 'bench' });
    expect(r.overall.p5).toBe(0);
    expect(r.overall.r10).toBe(0);
    expect(r.overall.mrr).toBe(0);
    expect(r.byClass['hard-negative']?.fpAt5).toBe(0);
    expect(r.perQuery.every((q) => q.rawHits === 0)).toBe(true);
  });

  it('rankedCorpusKeys dedupes and drops unstamped hits', () => {
    const hits: MemoryEntry[] = [
      { id: '1', text: 'x', scope: 's', metadata: { corpus_key: 'a' } },
      { id: '2', text: 'y', scope: 's', metadata: { corpus_key: 'a' } },
      { id: '3', text: 'z', scope: 's' },
      { id: '4', text: 'w', scope: 's', metadata: { corpus_key: 'b' } },
    ];
    expect(rankedCorpusKeys(hits)).toEqual(['a', 'b']);
  });
});

/* ── round-trips ──────────────────────────────────────────────────── */

const RT_SPECS = [
  {
    id: 'rt1',
    fact: 'The flux capacitor cache invalidates on the brontosaurus-velvet schedule.',
    paraphrase: 'when does that capacitor cache get cleared?',
    updatedText: 'The flux capacitor cache invalidates on the marzipan-thunder schedule.',
    nearDup: 'Flux capacitor cache invalidation follows the brontosaurus-velvet schedule.',
    marker: 'brontosaurus-velvet',
    updatedMarker: 'marzipan-thunder',
  },
];

describe('runRoundtrips', () => {
  it('observes the lexical double honestly (store/update/forget yes, dedup no)', async () => {
    const be = new LexicalDouble();
    const [o] = await runRoundtrips(be, RT_SPECS, { scope: 'rt', settleMs: 0 });
    expect(o.stored).toBe(true);
    expect(o.paraphraseFound).toBe(true); // 'capacitor'+'cache' overlap
    expect(o.updateHonored).toBe(true);
    expect(o.forgetHonored).toBe(true);
    expect(o.nearDupNewEntries).toBe(1); // no dedup in the double
    expect(o.error).toBeUndefined();
    // cleanup happened
    expect(await be.list({ scope: 'rt' })).toEqual([]);
  });

  it('NoopBackend control: nothing stores, nothing passes', async () => {
    const [o] = await runRoundtrips(new NoopBackend(), RT_SPECS, { scope: 'rt', settleMs: 0 });
    expect(o.stored).toBe(false);
    expect(o.paraphraseFound).toBe(false);
    expect(o.updateHonored).toBe(false);
    expect(o.forgetHonored).toBe(false);
  });

  it('distinctiveToken picks the longest token', () => {
    expect(distinctiveToken('short and brontosaurus-velvet word')).toBe('brontosaurus-velvet');
  });
});

/* ── claude-file end-to-end on a temp dir ─────────────────────────── */

describe('claude-file backend through the bench engine', () => {
  let dir: string;
  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mem-bench-'));
  });
  afterEach(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('seeds, retrieves exact-identifier, misses lexical-gap, round-trips writes', async () => {
    const be = new ClaudeFileMemoryBackend({ memoryDir: dir, createIfMissing: true });
    const manifest = await seedCorpus(be, CORPUS, { scope: 'bench' });
    expect(manifest.failed).toEqual([]);

    const r = await runGoldSet(be, GOLD, { scope: 'bench' });
    expect(r.byClass['exact-identifier']?.mrr).toBe(1);
    expect(r.byClass['lexical-gap']?.mrr).toBe(0); // grep loses lexical-gap by design (D-003)
    expect(r.byClass['hard-negative']?.fpAt5).toBe(0);

    const [o] = await runRoundtrips(be, RT_SPECS, { scope: 'bench', settleMs: 0 });
    expect(o.stored).toBe(true);
    expect(o.updateHonored).toBe(true);
    // forget archives the topic file → get() null afterward.
    expect(o.forgetHonored).toBe(true);
    expect(o.nearDupNewEntries).toBe(1);
  });
});

/* ── synthetic generator ──────────────────────────────────────────── */

describe('generateSyntheticCorpus', () => {
  it('is deterministic per seed and yields unique keys', () => {
    const a = generateSyntheticCorpus(50, 42);
    const b = generateSyntheticCorpus(50, 42);
    expect(a).toEqual(b);
    expect(new Set(a.map((e) => e.key)).size).toBe(50);
    const c = generateSyntheticCorpus(50, 43);
    expect(c).not.toEqual(a);
  });
});

/* ── scorecard rendering ──────────────────────────────────────────── */

describe('renderScorecardMarkdown', () => {
  it('renders one comparison table with no verdict', async () => {
    const be = new LexicalDouble();
    const manifest = await seedCorpus(be, CORPUS, { scope: 'bench' });
    const retrieval = await runGoldSet(be, GOLD, { scope: 'bench' });
    const roundtrips = await runRoundtrips(be, RT_SPECS, { scope: 'bench', settleMs: 0 });
    const card: BackendScorecard = {
      backend: be.name,
      seeded: CORPUS.length,
      seedFailed: manifest.failed.length,
      retrieval,
      roundtrips,
      rememberP50Ms: rememberP50(manifest.rememberMs),
      costPer1kRemembers: 0,
      costPer1kSearches: 0,
      scale: [{ size: 111, p5: 0.4, mrr: 0.7, searchP50Ms: 3 }],
      reach: { 'multi-process': 'no — in-process only (test double)' },
      ops: { remembers: 3, searches: 4, totalCharsWritten: manifest.totalChars, totalCharsQueried: 0 },
    };
    const md = renderScorecardMarkdown([card, card]);
    expect(md).toContain('| metric | **lexical-double** | **lexical-double** |');
    expect(md).toContain('P@5 overall');
    expect(md).toContain('hard-negative — FP@5');
    expect(md).toContain('near-dup behavior');
    expect(md).toContain('@111 — P@5 / search p50');
    expect(md).toContain('reach: multi-process');
    // D-006: the renderer never editorializes.
    expect(md.toLowerCase()).not.toContain('winner');
    expect(md.toLowerCase()).not.toContain('recommend');
  });
});
