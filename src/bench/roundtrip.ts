/**
 * Write-round-trip checks — the write tier of the memory-backend
 * benchmark (D-004): remember → paraphrase-search finds it; update and
 * forget honored; near-dup behavior observed. Everything goes through
 * the seam; outcomes are OBSERVED facts, not pass/fail judgments —
 * e.g. `nearDupNewEntries` reports how the backend treated a duplicate
 * (mem0's extractor should merge; a file store appends), and the
 * scorecard renders the difference rather than declaring a winner.
 *
 * Probes use plain (non-verbatim) remember on purpose: this tier
 * measures the backend's REAL write path, extraction and all.
 */

import type { MemoryBackend } from '../backend';
import type { RoundtripOutcome, RoundtripSpec } from './types';

export interface RoundtripOptions {
  /** Pool for the probe writes (cleaned up per spec). */
  scope: string;
  /** Search limit for the paraphrase probe (rank window; default 5). */
  limit?: number;
  /**
   * Settle delay (ms) between a write and the search that must see it —
   * stores with async indexing need a beat (default 250).
   */
  settleMs?: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Run every spec against one backend; entries are cleaned up per spec. */
export async function runRoundtrips(
  backend: MemoryBackend,
  specs: readonly RoundtripSpec[],
  opts: RoundtripOptions,
): Promise<RoundtripOutcome[]> {
  const out: RoundtripOutcome[] = [];
  for (const spec of specs) {
    out.push(await runOneRoundtrip(backend, spec, opts));
  }
  return out;
}

async function runOneRoundtrip(
  backend: MemoryBackend,
  spec: RoundtripSpec,
  opts: RoundtripOptions,
): Promise<RoundtripOutcome> {
  const limit = opts.limit ?? 5;
  const settleMs = opts.settleMs ?? 250;
  const outcome: RoundtripOutcome = {
    specId: spec.id,
    stored: false,
    paraphraseFound: false,
    paraphraseRank: 0,
    updateHonored: false,
    forgetHonored: false,
    nearDupNewEntries: 0,
    rememberMs: 0,
  };
  const created: string[] = [];
  try {
    // 1. Write the fact through the real path (extraction included).
    const t0 = performance.now();
    const r = await backend.remember(spec.fact, { scope: opts.scope, kind: 'project' });
    outcome.rememberMs = performance.now() - t0;
    created.push(...r.ids);
    outcome.stored = (r.storedEvents ?? r.ids.length) > 0;
    if (!outcome.stored) return outcome;
    await sleep(settleMs);

    // 2. Paraphrase search — does low-lexical-overlap recall find it?
    const hits = await backend.search(spec.paraphrase, { scope: opts.scope, limit });
    const marker = (spec.marker ?? distinctiveToken(spec.fact)).toLowerCase();
    const rank = hits.findIndex((h) => h.text.toLowerCase().includes(marker));
    outcome.paraphraseRank = rank + 1;
    outcome.paraphraseFound = rank !== -1 && rank < 3;

    // 3. Update honored (text-only patch — the lowest common denominator).
    const targetId = r.ids[0] ?? hits[rank]?.id;
    if (targetId) {
      try {
        await backend.update(targetId, { text: spec.updatedText });
        const got = await backend.get(targetId);
        const updatedMarker = (spec.updatedMarker ?? distinctiveToken(spec.updatedText)).toLowerCase();
        outcome.updateHonored = !!got && got.text.toLowerCase().includes(updatedMarker);
      } catch {
        outcome.updateHonored = false;
      }
    }

    // 4. Near-dup write — observe whether the backend merges or appends.
    try {
      const dup = await backend.remember(spec.nearDup, { scope: opts.scope, kind: 'project' });
      created.push(...dup.ids);
      outcome.nearDupNewEntries = dup.ids.length;
    } catch {
      outcome.nearDupNewEntries = -1; // threw — visible in the report
    }

    // 5. Forget honored.
    if (targetId) {
      try {
        await backend.forget(targetId);
        const got = await backend.get(targetId);
        outcome.forgetHonored = got === null;
      } catch {
        outcome.forgetHonored = false;
      }
    }
  } catch (e) {
    outcome.error = (e as Error).message;
  } finally {
    // Cleanup everything this spec created (incl. extraction side-facts).
    for (const id of created) {
      try {
        await backend.forget(id);
      } catch {
        /* best-effort */
      }
    }
  }
  return outcome;
}

/**
 * The longest token of a probe text (≥6 chars) — a cheap distinctive
 * marker for "did the search find THIS fact". Probe authors embed a
 * unique cipher word per spec so this never collides with the corpus.
 */
export function distinctiveToken(text: string): string {
  const tokens = text.toLowerCase().split(/[^a-z0-9-]+/).filter((t) => t.length >= 6);
  return tokens.sort((a, b) => b.length - a.length)[0] ?? text.toLowerCase().slice(0, 12);
}
