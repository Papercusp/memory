/**
 * LexicalLegBackend (memory-pg-lexical-own-injection-2026-07-13 P-003) —
 * the shared-store lexical leg of `hybrid-pg`. The load-bearing contracts:
 * search() IS the wrapped searchLexical, and remember() is a NO-OP (the
 * double-write guard for legs sharing one canonical table).
 */
import { describe, expect, it, vi } from 'vitest';

import type { MemoryBackend, MemoryEntry } from './backend';
import { HybridBackend } from './hybrid-backend';
import { LexicalLegBackend } from './lexical-leg';

function makeInner(overrides: Partial<MemoryBackend> = {}): MemoryBackend {
  return {
    name: 'fake-inner',
    available: vi.fn(async () => ({ ok: true }) as const),
    remember: vi.fn(async () => ({ ids: ['real-id'] })),
    search: vi.fn(async () => [] as MemoryEntry[]),
    searchLexical: vi.fn(async () => [
      { id: 'lex-1', text: 'hit', scope: 's', score: 0.9 },
    ]),
    list: vi.fn(async () => []),
    get: vi.fn(async () => null),
    forget: vi.fn(async () => {}),
    update: vi.fn(async () => {}),
    ...overrides,
  };
}

describe('LexicalLegBackend', () => {
  it('REFUSES to wrap a backend without the searchLexical capability', () => {
    const inner = makeInner();
    delete (inner as { searchLexical?: unknown }).searchLexical;
    expect(() => new LexicalLegBackend(inner)).toThrow(/no searchLexical capability/);
  });

  it('search() delegates to the wrapped searchLexical (the lexical ranking IS the leg)', async () => {
    const inner = makeInner();
    const leg = new LexicalLegBackend(inner);
    const out = await leg.search('WI-4214 embed', { scope: 's', limit: 5 });
    expect(inner.searchLexical).toHaveBeenCalledWith('WI-4214 embed', { scope: 's', limit: 5 });
    expect(inner.search).not.toHaveBeenCalled();
    expect(out.map((e) => e.id)).toEqual(['lex-1']);
  });

  it('remember() is a NO-OP: never touches the wrapped backend, reports nothing stored', async () => {
    const inner = makeInner();
    const leg = new LexicalLegBackend(inner);
    const res = await leg.remember('a fact', { scope: 's' });
    // { ids: [], storedEvents: 0 } is the honest "nothing persisted" shape
    // (EI-25) — the cosine leg's remember already landed the row.
    expect(res).toEqual({ ids: [], storedEvents: 0 });
    expect(inner.remember).not.toHaveBeenCalled();
  });

  it('inside a HybridBackend, the write-through projection does NOT double-write the shared store', async () => {
    const inner = makeInner();
    const hybrid = new HybridBackend(new LexicalLegBackend(inner), inner);
    const res = await hybrid.remember('a fact', { scope: 's' });
    // The canonical write happened EXACTLY once (the cosine leg); the
    // lexical write-through hit the no-op.
    expect(res.ids).toEqual(['real-id']);
    expect(inner.remember).toHaveBeenCalledTimes(1);
  });

  it('inside a HybridBackend, search() fuses the wrapped searchLexical as the lexical leg', async () => {
    const inner = makeInner({
      search: vi.fn(async () => [
        { id: 'cos-1', text: 'cosine hit', scope: 's', score: 0.8 },
      ]),
    });
    const hybrid = new HybridBackend(new LexicalLegBackend(inner), inner);
    const out = await hybrid.search('embed', { scope: 's', limit: 5 });
    expect(inner.searchLexical).toHaveBeenCalled();
    expect(out.map((e) => e.id)).toEqual(expect.arrayContaining(['cos-1', 'lex-1']));
  });

  it('fusion DEDUPES a row surfaced by both legs (same canonical id)', async () => {
    const shared: MemoryEntry = { id: 'same-row', text: 'fact', scope: 's', score: 0.9 };
    const inner = makeInner({
      search: vi.fn(async () => [{ ...shared, score: 0.8 }]),
      searchLexical: vi.fn(async () => [{ ...shared, score: 0.9 }]),
    });
    const hybrid = new HybridBackend(new LexicalLegBackend(inner), inner);
    const out = await hybrid.search('fact', { scope: 's', limit: 5 });
    expect(out.filter((e) => e.id === 'same-row')).toHaveLength(1);
  });

  it('lifecycle calls delegate to the wrapped backend (same store)', async () => {
    const inner = makeInner();
    const leg = new LexicalLegBackend(inner);
    await leg.available();
    await leg.list({ scope: 's' });
    await leg.get('id-1');
    await leg.forget('id-1');
    await leg.update('id-1', { text: 'new' });
    expect(inner.available).toHaveBeenCalled();
    expect(inner.list).toHaveBeenCalled();
    expect(inner.get).toHaveBeenCalledWith('id-1');
    expect(inner.forget).toHaveBeenCalledWith('id-1');
    expect(inner.update).toHaveBeenCalledWith('id-1', { text: 'new' });
  });
});
