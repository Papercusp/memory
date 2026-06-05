/**
 * `NoopBackend` — the deliberate "no store" implementation of the
 * neutral `MemoryBackend` seam (generalize-memory-backend-swappable
 * D-004).
 *
 * Selecting it makes "memory is off" a CLEAN, testable state rather
 * than a broken-mem0 state: reads come back empty (surfaces render
 * their empty state, not an error), and writes throw
 * `MemoryUnavailableError('memory_backend_disabled')` so no caller is
 * ever lied to about a fact having been stored.
 *
 * `available()` reports `{ ok: false }` because the store is OFF by
 * choice — a backend that is merely empty reports `{ ok: true }`.
 */

import {
  MemoryUnavailableError,
  type ListOptions,
  type MemoryAvailability,
  type MemoryBackend,
  type MemoryEntry,
  type RememberOptions,
  type SearchOptions,
  type UpdatePatch,
} from './backend';

export const NOOP_DISABLED_REASON = 'memory_backend_disabled';

export class NoopBackend implements MemoryBackend {
  readonly name: string = 'noop';

  async available(): Promise<MemoryAvailability> {
    return { ok: false, reason: NOOP_DISABLED_REASON };
  }

  async remember(_text: string, _opts: RememberOptions): Promise<{ ids: string[] }> {
    throw new MemoryUnavailableError(NOOP_DISABLED_REASON);
  }

  async search(_query: string, _opts: SearchOptions): Promise<MemoryEntry[]> {
    return [];
  }

  async list(_opts: ListOptions): Promise<MemoryEntry[]> {
    return [];
  }

  async get(_id: string): Promise<MemoryEntry | null> {
    return null;
  }

  async forget(_id: string): Promise<void> {
    throw new MemoryUnavailableError(NOOP_DISABLED_REASON);
  }

  async update(_id: string, _patch: UpdatePatch): Promise<void> {
    throw new MemoryUnavailableError(NOOP_DISABLED_REASON);
  }
}
