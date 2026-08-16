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
  /** Progress callback (done, total). */
  onProgress?: (done: number, total: number) => void;
}

/** Seed one corpus into one backend; returns the key→ids manifest. */
export async function seedCorpus(
  backend: MemoryBackend,
  corpus: readonly CorpusEntry[],
  opts: SeedOptions,
): Promise<SeedManifest> {
  const verbatim = opts.verbatim ?? true;
  const concurrency = Math.max(1, opts.concurrency ?? 8);
  const manifest: SeedManifest = {
    backend: backend.name,
    scope: opts.scope,
    ids: {},
    failed: [],
    rememberMs: new Array(corpus.length).fill(0),
    totalChars: 0,
  };

  let next = 0;
  let done = 0;
  async function worker(): Promise<void> {
    for (;;) {
      const i = next++;
      if (i >= corpus.length) return;
      const entry = corpus[i];
      const t0 = performance.now();
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
        const persisted = r.storedEvents ?? r.ids.length;
        if (persisted === 0) manifest.failed.push(entry.key);
      } catch {
        manifest.ids[entry.key] = [];
        manifest.failed.push(entry.key);
      }
      manifest.rememberMs[i] = performance.now() - t0;
      manifest.totalChars += entry.text.length;
      done += 1;
      opts.onProgress?.(done, corpus.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, corpus.length || 1) }, worker));
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
