/**
 * Scorecard rendering — the decision artifact (memory-backend-benchmark
 * D-004/D-006): ONE comparison table over every benched backend. The
 * scorecard INFORMS; it never recommends — the revive-vs-retire call is
 * the owner's, so the renderer emits measurements and observed behavior
 * only, no verdict line.
 */

import type { GoldQueryClass, BackendScorecard, RoundtripOutcome } from './types';
import { GOLD_QUERY_CLASSES } from './types';

function pct(x: number | undefined): string {
  return x === undefined ? '—' : `${(x * 100).toFixed(0)}%`;
}

function num(x: number | undefined, unit = '', digits = 0): string {
  return x === undefined ? '—' : `${x.toFixed(digits)}${unit}`;
}

function money(x: number | undefined): string {
  if (x === undefined) return '—';
  if (x === 0) return '$0';
  return x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`;
}

function roundtripSummary(rts: readonly RoundtripOutcome[]): {
  stored: string; paraphrase: string; update: string; forget: string; nearDup: string;
} {
  const frac = (f: (r: RoundtripOutcome) => boolean) => `${rts.filter(f).length}/${rts.length}`;
  // Near-dup behavior is only meaningful for specs whose FIRST write stored
  // (a backend that stores nothing "merges" nothing — the control must read
  // '—', not a perfect merge score).
  const stored = rts.filter((r) => r.stored);
  const dupCounts = stored.map((r) => r.nearDupNewEntries);
  const merged = dupCounts.filter((c) => c === 0).length;
  return {
    stored: frac((r) => r.stored),
    paraphrase: frac((r) => r.paraphraseFound),
    update: frac((r) => r.updateHonored),
    forget: frac((r) => r.forgetHonored),
    nearDup: stored.length === 0
      ? '—'
      : `${merged}/${stored.length} merged` + (dupCounts.some((c) => c < 0) ? ' (some threw)' : ''),
  };
}

/** Render the one comparison table (GitHub-flavored markdown). */
export function renderScorecardMarkdown(cards: readonly BackendScorecard[]): string {
  const lines: string[] = [];
  const header = ['metric', ...cards.map((c) => `**${c.backend}**`)];
  const sep = header.map(() => '---');
  const row = (label: string, cells: (string | undefined)[]) =>
    lines.push(`| ${[label, ...cells.map((c) => c ?? '—')].join(' | ')} |`);

  lines.push(`| ${header.join(' | ')} |`);
  lines.push(`| ${sep.join(' | ')} |`);

  row('seeded (of corpus)', cards.map((c) => `${c.seeded - c.seedFailed}/${c.seeded}`));

  // Retrieval — overall + per class.
  row('P@5 overall', cards.map((c) => pct(c.retrieval.overall.p5)));
  row('R@10 overall', cards.map((c) => pct(c.retrieval.overall.r10)));
  row('MRR overall', cards.map((c) => num(c.retrieval.overall.mrr, '', 2)));
  for (const cls of GOLD_QUERY_CLASSES) {
    const present = cards.some((c) => c.retrieval.byClass[cls]);
    if (!present) continue;
    if (cls === 'hard-negative') {
      row(`${cls} — FP@5 (lower better)`, cards.map((c) => pct(c.retrieval.byClass[cls]?.fpAt5)));
      row(`${cls} — median top score`, cards.map((c) => num(c.retrieval.byClass[cls]?.medianTopScore, '', 3)));
    } else {
      row(`${cls} — P@5`, cards.map((c) => pct(c.retrieval.byClass[cls]?.p5)));
      row(`${cls} — MRR`, cards.map((c) => num(c.retrieval.byClass[cls]?.mrr, '', 2)));
    }
  }

  // Write round-trips.
  const rt = cards.map((c) => roundtripSummary(c.roundtrips));
  row('write stored', rt.map((r) => r.stored));
  row('paraphrase recall (top-3)', rt.map((r) => r.paraphrase));
  row('update honored', rt.map((r) => r.update));
  row('forget honored', rt.map((r) => r.forget));
  row('near-dup behavior', rt.map((r) => r.nearDup));

  // Latency.
  row('search p50', cards.map((c) => num(c.retrieval.latency.p50, 'ms')));
  row('search p95', cards.map((c) => num(c.retrieval.latency.p95, 'ms')));
  row('remember p50', cards.map((c) => num(c.rememberP50Ms, 'ms')));

  // Cost.
  row('cost / 1k remembers', cards.map((c) => money(c.costPer1kRemembers)));
  row('cost / 1k searches', cards.map((c) => money(c.costPer1kSearches)));

  // Scale curve.
  const scaleSizes = [...new Set(cards.flatMap((c) => (c.scale ?? []).map((s) => s.size)))].sort((a, b) => a - b);
  for (const size of scaleSizes) {
    row(`@${size} — P@5 / search p50`, cards.map((c) => {
      const pt = c.scale?.find((s) => s.size === size);
      return pt ? `${pct(pt.p5)} / ${num(pt.searchP50Ms, 'ms')}` : undefined;
    }));
  }

  // Reach checklist.
  const reachKeys = [...new Set(cards.flatMap((c) => Object.keys(c.reach ?? {})))];
  for (const key of reachKeys) {
    row(`reach: ${key}`, cards.map((c) => c.reach?.[key]));
  }

  return lines.join('\n');
}

/** remember-latency p50 over a manifest's samples. */
export function rememberP50(samplesMs: readonly number[]): number {
  if (samplesMs.length === 0) return 0;
  const sorted = [...samplesMs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
