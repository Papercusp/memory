/**
 * LexicalLegBackend — present a backend's EMBED-FREE lexical capability
 * (`searchLexical`) as a standalone `MemoryBackend`, for use as the LEXICAL
 * leg of a `HybridBackend` when BOTH legs are served by the SAME canonical
 * store (the `hybrid-pg` wiring — memory-pg-lexical-own-injection-2026-07-13
 * P-003).
 *
 * Why this exists: the original hybrid pairs two SEPARATE stores (claude
 * topic files + the canonical PG store) and write-through-projects every
 * remember into the lexical one. `hybrid-pg` instead runs BOTH legs over the
 * one `memory_canonical` table — the lexical leg is just a different RANKING
 * (canonical-store `lexicalSearch`: field-weighted token match, no embedder)
 * of the same rows the cosine leg ranks by pgvector distance. Hence:
 *
 * - `search()` = the wrapped backend's `searchLexical` — so inside a
 *   HybridBackend the fusion's lexical leg IS the PG lexical ranking.
 * - `remember()` is a DELIBERATE NO-OP (`{ ids: [], storedEvents: 0 }`):
 *   HybridBackend.remember write-through-projects into its lexical leg,
 *   which is correct for separate stores but would DOUBLE-WRITE here — the
 *   cosine leg's remember already landed the row in the shared table.
 * - Everything else delegates to the wrapped backend (same store), so the
 *   adapter stays an honest `MemoryBackend` wherever it's held.
 *
 * Fusion dedupe is safe by construction: both legs return the SAME canonical
 * ids, so RRF's id-keyed slots merge them (no link_id stamping needed).
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

export class LexicalLegBackend implements MemoryBackend {
  readonly name = 'lexical-leg';

  constructor(private readonly inner: MemoryBackend) {
    if (typeof inner.searchLexical !== 'function') {
      throw new Error('LexicalLegBackend: wrapped backend has no searchLexical capability');
    }
  }

  available(): Promise<MemoryAvailability> {
    return this.inner.available();
  }

  /**
   * NO-OP by contract (see header): the shared-store write already happened
   * via the cosine leg; a real write here would double-store the fact.
   */
  async remember(_text: string, _opts: RememberOptions): Promise<{ ids: string[]; storedEvents?: number }> {
    return { ids: [], storedEvents: 0 };
  }

  search(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    // Constructor-verified present; the non-null assert is safe.
    return this.inner.searchLexical!(query, opts);
  }

  searchLexical(query: string, opts: SearchOptions): Promise<MemoryEntry[]> {
    return this.inner.searchLexical!(query, opts);
  }

  list(opts: ListOptions): Promise<MemoryEntry[]> {
    return this.inner.list(opts);
  }

  get(id: string): Promise<MemoryEntry | null> {
    return this.inner.get(id);
  }

  forget(id: string): Promise<void> {
    return this.inner.forget(id);
  }

  update(id: string, patch: UpdatePatch): Promise<void> {
    return this.inner.update(id, patch);
  }
}
