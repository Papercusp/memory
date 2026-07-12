/**
 * noop-backend.test.ts — the deliberate "no store" backend (task #36
 * hardening). NoopBackend is what makes "memory is OFF" a CLEAN, testable
 * state instead of a broken-mem0 state, so its two-sided contract is
 * load-bearing for a public release:
 *
 *  1. READS DEGRADE TO EMPTY, never throw — search/list return [], get
 *     returns null. A surface renders its empty state, not an error banner.
 *  2. WRITES THROW MemoryUnavailableError('memory_backend_disabled') —
 *     remember/forget/update must NEVER resolve silently, or a caller would
 *     be lied to about a fact having been stored.
 *  3. available() reports { ok:false, reason } — OFF BY CHOICE, distinct from
 *     a merely-empty store (which reports { ok:true }). This is the signal the
 *     registry/degrade paths branch on.
 *
 * Run: cd libs/generic/memory && npx vitest run src/noop-backend.test.ts
 */
import { describe, it, expect } from 'vitest';
import { NoopBackend, NOOP_DISABLED_REASON } from './noop-backend';
import { MemoryUnavailableError } from './backend';

const b = () => new NoopBackend();

describe('NoopBackend — identity + availability', () => {
  it("names itself 'noop'", () => {
    expect(b().name).toBe('noop');
  });

  it('reports OFF-by-choice: available() = { ok:false, reason:memory_backend_disabled }', async () => {
    expect(await b().available()).toEqual({ ok: false, reason: NOOP_DISABLED_REASON });
  });

  it('the disabled reason is the stable machine token surfaces render verbatim', () => {
    expect(NOOP_DISABLED_REASON).toBe('memory_backend_disabled');
  });
});

describe('NoopBackend — reads degrade to EMPTY (never throw)', () => {
  it('search returns [] for any query/scope', async () => {
    await expect(b().search('anything', { scope: 'user:1' })).resolves.toEqual([]);
  });

  it('list returns []', async () => {
    await expect(b().list({ scope: 'user:1' })).resolves.toEqual([]);
  });

  it('get returns null (not found, not an error)', async () => {
    await expect(b().get('some-id')).resolves.toBeNull();
  });
});

describe('NoopBackend — writes THROW (a caller is never lied to about a store)', () => {
  it('remember throws MemoryUnavailableError(disabled) — not a resolved empty-ids', async () => {
    await expect(b().remember('a fact', { scope: 'user:1' })).rejects.toBeInstanceOf(MemoryUnavailableError);
    await expect(b().remember('a fact', { scope: 'user:1' })).rejects.toMatchObject({
      reason: NOOP_DISABLED_REASON,
    });
  });

  it('forget throws MemoryUnavailableError(disabled)', async () => {
    await expect(b().forget('some-id')).rejects.toBeInstanceOf(MemoryUnavailableError);
    await expect(b().forget('some-id')).rejects.toMatchObject({ reason: NOOP_DISABLED_REASON });
  });

  it('update throws MemoryUnavailableError(disabled)', async () => {
    await expect(b().update('some-id', { text: 'x' })).rejects.toBeInstanceOf(MemoryUnavailableError);
    await expect(b().update('some-id', { text: 'x' })).rejects.toMatchObject({ reason: NOOP_DISABLED_REASON });
  });
});
