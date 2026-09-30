/**
 * Corpus seeding through the neutral seam (memory-backend-benchmark
 * D-002): every backend under test receives the SAME corpus through
 * `backend.remember()` — the only API the bench touches. Each created
 * entry is stamped with `metadata.corpus_key` so ranked hits resolve
 * back to corpus keys regardless of the backend's own id scheme.
 *
 * Writes default to `verbatim: true` (D-008) so the corpus lands
 * byte-identical (no extract/transform lossiness) and seeding is
 * LLM-free; the extraction write path is measured separately by the
 * round-trip checks.
 */

import type { MemoryBackend } from '../backend';
import type { CorpusEntry, SeedManifest } from './types';

export interface SeedOptions {
  /** The pool to seed into. */
  scope: string;
  /** Store as-is (default true — fair, byte-identical seeding). */
  verbatim?: boolean;
  /** Parallel remember() calls (default 8; 1 = serial). */
  concurrency?: number;
  /** Total attempts for rows whose remember() throws or persists zero events. */
  maxAttempts?: number;
  /** Delay between failed-row-only attempts (default 0). */
  retryDelayMs?: number;
  /** Progress callback (done, total). */
  onProgress?: (done: number, total: number) => void;
}

/** Seed one corpus into one backend; returns the key→ids manifest. */
/**
 * `null` when every one of `expected` corpus entries persisted; otherwise a
 * one-line reason: how many seeded, how many failed, a sample of failed keys, and
 * the FIRST error remember() threw. A bench must refuse to report over a partially
 * seeded store, because an empty pool scores zero (or "admits nothing") for
 * infrastructure reasons, not retrieval reasons. The error text is the point: a
 * schema fault that rejected every write read only as "114 failed" for four weeks
 * (WI-10004107).
 *
 * A failed key is not counted as seeded even though seedCorpus records `ids[key] = []`
 * for it, so a total failure reads "0/N seeded", never "N/N seeded, N failed".
 */
export function seedFailureReason(manifest: SeedManifest, expected: number): string | null {
  const failed = new Set(manifest.failed);
  const seeded = Object.keys(manifest.ids).filter((k) => !failed.has(k)).length;
  if (seeded === expected && failed.size === 0) return null;
  const sample = manifest.failed.slice(0, 3).join(', ');
  const firstKey = manifest.failed.find((k) => manifest.errors?.[k]);
  return (
    `corpus seed incomplete: ${seeded}/${expected} seeded, ${manifest.failed.length} failed` +
    (sample ? ` (e.g. ${sample})` : '') +
    (firstKey ? ` — first error (${firstKey}): ${manifest.errors[firstKey]}` : '')
  );
}

export async function seedCorpus(
  backend: MemoryBackend,
  corpus: readonly CorpusEntry[],
  opts: SeedOptions,
): Promise<SeedManifest> {
  const verbatim = opts.verbatim ?? true;
  const concurrency = Math.max(1, opts.concurrency ?? 8);
  const maxAttempts = Math.max(1, Math.floor(opts.maxAttempts ?? 1));
  const retryDelayMs = Math.max(0, Math.floor(opts.retryDelayMs ?? 0));
  const manifest: SeedManifest = {
    backend: backend.name,
    scope: opts.scope,
    ids: {},
    failed: [],
    errors: {},
    rememberMs: new Array(corpus.length).fill(0),
    totalChars: corpus.reduce((sum, entry) => sum + entry.text.length, 0),
  };

  let pending = corpus.map((_, index) => index);
  for (let attempt = 1; attempt <= maxAttempts && pending.length > 0; attempt++) {
    let next = 0;
    let done = 0;
    const failed: number[] = [];
    async function worker(): Promise<void> {
      for (;;) {
        const pendingIndex = next++;
        if (pendingIndex >= pending.length) return;
        const i = pending[pendingIndex];
        const entry = corpus[i];
        const t0 = performance.now();
        let persisted = false;
        try {
          const r = await backend.remember(entry.text, {
            scope: opts.scope,
            kind: entry.kind,
            verbatim,
            metadata: {
              ...(entry.description ? { description: entry.description } : {}),
              ...(entry.metadata ?? {}),
              corpus_key: entry.key,
            },
          });
          manifest.ids[entry.key] = r.ids;
          persisted = (r.storedEvents ?? r.ids.length) > 0;
          if (persisted) delete manifest.errors[entry.key];
          else manifest.errors[entry.key] = 'remember() returned without persisting anything';
        } catch (err) {
          manifest.ids[entry.key] = [];
          manifest.errors[entry.key] = err instanceof Error ? err.message : String(err);
        }
        manifest.rememberMs[i] += performance.now() - t0;
        if (!persisted) failed.push(i);
        if (attempt === 1) opts.onProgress?.(++done, corpus.length);
      }
    }
    await Promise.all(Array.from({ length: Math.min(concurrency, pending.length || 1) }, worker));
    pending = failed;
    if (pending.length > 0 && attempt < maxAttempts && retryDelayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
    }
  }
  manifest.failed = pending.map((index) => corpus[index].key);
  return manifest;
}

/** Remove every entry the manifest created (best-effort, id-by-id). */
export async function unseedCorpus(backend: MemoryBackend, manifest: SeedManifest): Promise<number> {
  let removed = 0;
  for (const ids of Object.values(manifest.ids)) {
    for (const id of ids) {
      try {
        await backend.forget(id);
        removed += 1;
      } catch {
        /* best-effort */
      }
    }
  }
  return removed;
}
