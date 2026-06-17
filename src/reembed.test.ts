/**
 * GAP 5 — `reembedMemories` (reembed.ts) was UNTESTED, and it is the ONLY
 * embed path that degrades PER-FACT: each row is embedded+upserted inside a
 * `try { … } catch { errors++ }` so ONE bad fact (embedder throw, wrong dims)
 * must NOT abort the whole pass. That per-fact resilience is the documented
 * data-loss-prevention — without it, a single poison fact mid-walk would
 * silently leave the rest of the user's memories unembedded in the target
 * vec table (i.e. invisible to recall after a mode switch). We pin:
 *   - the `fromMode === toMode` guard throw (reembed_noop_same_mode),
 *   - per-fact embed → upsert into the TARGET vec table,
 *   - dim-mismatch → errors++ (no bad row written),
 *   - per-row catch{errors++} CONTINUE (one bad fact doesn't abort the pass),
 *   - empty/non-string payload → skipped,
 *   - progress callbacks fire per fact,
 *   - the returned {reembedded, skipped, errors, from/toCollection} counts.
 *
 * Mirrors canonical-store.test.ts's fake-pg pattern: `pg` is swapped for a
 * query-capturing fake Client (we pin the SQL/params the worker emits, not PG
 * behavior). `pgFields()` resolves naturally from the configured host's
 * getAdminUrl (no separate mock needed). The target-mode embedder comes from
 * the host seam `buildEmbedderForMode`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { configureMemory, type EmbedFn } from './config';

const EMBEDDER_DIM = 384;
const goodVec = () => Array.from({ length: EMBEDDER_DIM }, () => 0.01);

type CapturedQuery = { sql: string; params: unknown[] };

// --- query-capturing fake pg.Client (the canonical-store.test.ts pattern) ---
// Scripted by the active test: SELECT returns the source rows, INSERT records
// (and may be told to throw to exercise the per-row catch).
interface FakeScript {
  selectRows: Array<{ id: string; payload: Record<string, unknown> }>;
  // memory_ids for which the INSERT should throw (simulates a PG write failure)
  failInsertForIds?: Set<string>;
}

let script: FakeScript = { selectRows: [] };
let captured: CapturedQuery[] = [];
let connectCalls = 0;
let endCalls = 0;

class FakeClient {
  constructor(public cfg: Record<string, unknown>) {}
  async connect() {
    connectCalls += 1;
  }
  async end() {
    endCalls += 1;
  }
  async query(sql: string, params: unknown[] = []) {
    captured.push({ sql, params });
    if (/^\s*SELECT/i.test(sql)) {
      return { rows: script.selectRows, rowCount: script.selectRows.length };
    }
    // INSERT ... ON CONFLICT — params[0] is the memory_id
    const memId = params[0] as string;
    if (script.failInsertForIds?.has(memId)) {
      throw new Error(`pg_insert_failed_for_${memId}`);
    }
    return { rows: [], rowCount: 1 };
  }
}

vi.mock('pg', () => ({ Client: FakeClient, default: { Client: FakeClient } }));

// --- host seam: getAdminUrl feeds pgFields(); buildEmbedderForMode feeds the
//     re-embed pass with the TARGET-mode embedder. ---
let embedderForMode: (mode: 'openai' | 'local') => Promise<EmbedFn>;

function configure(embed: (mode: 'openai' | 'local') => Promise<EmbedFn>): void {
  embedderForMode = embed;
  configureMemory({
    getAdminUrl: () => 'postgres://u:p@localhost:5432/db',
    getCredentials: async () => ({}),
    resolveEmbedder: async () => ({ mode: 'disabled' as const }),
    buildEmbedderForMode: (mode) => embedderForMode(mode),
    schema: 'harness_shared',
  });
}

// Import the SUT after the pg mock is registered.
let reembedMemories: typeof import('./reembed').reembedMemories;
beforeEach(async () => {
  ({ reembedMemories } = await import('./reembed'));
  script = { selectRows: [] };
  captured = [];
  connectCalls = 0;
  endCalls = 0;
});
afterEach(() => vi.restoreAllMocks());

const fact = (id: string, data: string) => ({ id, payload: { data } });

describe('reembedMemories — same-mode guard (GAP 5)', () => {
  it('throws reembed_noop_same_mode when from === to (never opens a connection)', async () => {
    configure(async () => async () => goodVec());
    await expect(reembedMemories('openai', 'openai')).rejects.toThrow('reembed_noop_same_mode');
    await expect(reembedMemories('local', 'local')).rejects.toThrow('reembed_noop_same_mode');
    expect(connectCalls).toBe(0); // guard short-circuits before any PG work
  });
});

describe('reembedMemories — happy path (GAP 5)', () => {
  it('embeds each source fact under the TARGET embedder and upserts into the target vec table', async () => {
    script.selectRows = [fact('m1', 'alpha'), fact('m2', 'beta'), fact('m3', 'gamma')];
    const embed = vi.fn(async () => goodVec());
    const buildForMode = vi.fn(async (_mode: 'openai' | 'local') => embed as EmbedFn);
    configure(buildForMode);

    const res = await reembedMemories('openai', 'local');

    // Target embedder built for the TO mode, not the FROM mode.
    expect(buildForMode).toHaveBeenCalledWith('local');
    expect(embed).toHaveBeenCalledTimes(3);
    expect(res).toMatchObject({ totalSource: 3, reembedded: 3, skipped: 0, errors: 0 });
    expect(res.fromCollection).toBe('harness_shared.memory_vec_openai');
    expect(res.toCollection).toBe('harness_shared.memory_vec_local');
    expect(typeof res.durationMs).toBe('number');

    // SELECT joins the FROM vec table; INSERTs hit the TO vec table.
    const select = captured.find((q) => /SELECT/i.test(q.sql));
    expect(select?.sql).toContain('harness_shared.memory_canonical');
    expect(select?.sql).toContain('harness_shared.memory_vec_openai');
    const inserts = captured.filter((q) => /INSERT/i.test(q.sql));
    expect(inserts).toHaveLength(3);
    expect(inserts[0].sql).toContain('harness_shared.memory_vec_local');
    expect(inserts[0].sql).toContain('ON CONFLICT (memory_id) DO UPDATE');
    expect(inserts[0].params[0]).toBe('m1'); // memory_id
    expect(inserts[0].params[1]).toBe(`[${goodVec().join(',')}]`); // vector literal

    // The connection is always closed.
    expect(endCalls).toBe(1);
  });

  it('reads the body from payload.memory when payload.data is absent', async () => {
    script.selectRows = [{ id: 'm1', payload: { memory: 'from-memory-field' } }];
    const embed = vi.fn(async () => goodVec());
    configure(async () => embed as EmbedFn);

    const res = await reembedMemories('local', 'openai');
    expect(res.reembedded).toBe(1);
    expect(embed).toHaveBeenCalledWith('from-memory-field');
  });

  it('fires the progress callback once per fact with running counts', async () => {
    script.selectRows = [fact('m1', 'a'), fact('m2', 'b')];
    configure(async () => async () => goodVec());
    const seen: number[] = [];
    const res = await reembedMemories('openai', 'local', {
      progress: (p) => seen.push(p.reembedded),
    });
    expect(res.reembedded).toBe(2);
    // progress observed the running tally climb 1 → 2
    expect(seen).toEqual([1, 2]);
  });
});

describe('reembedMemories — per-fact degradation does NOT abort the pass (GAP 5, the data-loss guard)', () => {
  it('a fact whose embed THROWS counts as an error but the pass continues', async () => {
    script.selectRows = [fact('m1', 'ok-1'), fact('m2', 'BOOM'), fact('m3', 'ok-2')];
    const embed = vi.fn(async (text: string) => {
      if (text === 'BOOM') throw new Error('embedder exploded');
      return goodVec();
    });
    configure(async () => embed as EmbedFn);

    const res = await reembedMemories('openai', 'local');

    // m1 + m3 still re-embedded; only m2 errored — the bad fact did NOT abort.
    expect(res).toMatchObject({ totalSource: 3, reembedded: 2, errors: 1, skipped: 0 });
    const inserts = captured.filter((q) => /INSERT/i.test(q.sql));
    expect(inserts.map((q) => q.params[0])).toEqual(['m1', 'm3']); // NOT m2
  });

  it('a fact whose embed returns the WRONG dims counts as an error, writes no row, continues', async () => {
    script.selectRows = [fact('m1', 'ok'), fact('m2', 'short'), fact('m3', 'ok2')];
    const embed = vi.fn(async (text: string) => (text === 'short' ? [1, 2, 3] : goodVec()));
    configure(async () => embed as EmbedFn);

    const res = await reembedMemories('openai', 'local');

    expect(res).toMatchObject({ reembedded: 2, errors: 1, skipped: 0 });
    const inserts = captured.filter((q) => /INSERT/i.test(q.sql));
    // the dim-mismatched fact (m2) never produced an INSERT
    expect(inserts.map((q) => q.params[0])).toEqual(['m1', 'm3']);
  });

  it('a fact whose UPSERT throws counts as an error and the pass continues', async () => {
    script.selectRows = [fact('m1', 'a'), fact('m2', 'b'), fact('m3', 'c')];
    script.failInsertForIds = new Set(['m2']);
    configure(async () => async () => goodVec());

    const res = await reembedMemories('openai', 'local');
    expect(res).toMatchObject({ reembedded: 2, errors: 1, skipped: 0 });
  });

  it('an empty / non-string payload body is SKIPPED (not errored), pass continues', async () => {
    script.selectRows = [
      fact('m1', 'good'),
      { id: 'm2', payload: { data: '' } }, // empty
      { id: 'm3', payload: {} }, // missing
      { id: 'm4', payload: { data: 123 as unknown as string } }, // non-string
      fact('m5', 'also-good'),
    ];
    const embed = vi.fn(async () => goodVec());
    configure(async () => embed as EmbedFn);

    const res = await reembedMemories('openai', 'local');
    expect(res).toMatchObject({ totalSource: 5, reembedded: 2, skipped: 3, errors: 0 });
    expect(embed).toHaveBeenCalledTimes(2); // only the two good facts
  });

  it('mixed batch: skipped + errored + ok all tallied independently, none abort the rest', async () => {
    script.selectRows = [
      fact('m1', 'ok-a'),
      { id: 'm2', payload: { data: '' } }, // skipped
      fact('m3', 'BOOM'), // error (throw)
      fact('m4', 'short-dims'), // error (wrong dims)
      fact('m5', 'ok-b'),
    ];
    const embed = vi.fn(async (text: string) => {
      if (text === 'BOOM') throw new Error('boom');
      if (text === 'short-dims') return [0.1];
      return goodVec();
    });
    configure(async () => embed as EmbedFn);

    const res = await reembedMemories('openai', 'local');
    expect(res).toMatchObject({ totalSource: 5, reembedded: 2, skipped: 1, errors: 2 });
    const inserts = captured.filter((q) => /INSERT/i.test(q.sql));
    expect(inserts.map((q) => q.params[0])).toEqual(['m1', 'm5']);
    // connection still closed despite the mid-walk failures
    expect(endCalls).toBe(1);
  });
});
