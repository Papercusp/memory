/**
 * getMemoryBackend() thunk resolution — the live backend-switch mechanism
 * (mem0-revive-or-retire / Brief 30). The host's `backend` may be a thunk
 * re-evaluated on every call, so the operator can flip the active store
 * from the UI without reconfiguring/restarting.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { configureMemory } from './config';
import {
  getMemoryBackend,
  registerMemoryBackend,
  _resetMemoryBackendsForTest,
} from './backend-registry';
import type { MemoryBackend } from './backend';

function fakeBackend(name: string): MemoryBackend {
  return {
    name,
    available: async () => ({ ok: true }),
    remember: async () => ({ ids: [] }),
    search: async () => [],
    list: async () => [],
    get: async () => null,
    forget: async () => {},
    update: async () => {},
  };
}

function configureWithBackend(backend: (() => string | MemoryBackend | undefined)): void {
  configureMemory({
    getAdminUrl: () => 'postgres://stub',
    getCredentials: async () => ({}),
    resolveEmbedder: async () => ({ mode: 'disabled' as const }),
    buildEmbedderForMode: async () => async () => [],
    backend,
  });
}

describe('getMemoryBackend thunk resolution', () => {
  afterEach(() => { _resetMemoryBackendsForTest(); });

  it('re-evaluates the thunk on every call so a flip takes effect with no reconfigure', () => {
    registerMemoryBackend('alpha', () => fakeBackend('alpha'));
    registerMemoryBackend('beta', () => fakeBackend('beta'));
    let choice = 'alpha';
    configureWithBackend(() => choice);
    expect(getMemoryBackend().name).toBe('alpha');
    choice = 'beta';
    expect(getMemoryBackend().name).toBe('beta');
    choice = 'alpha';
    expect(getMemoryBackend().name).toBe('alpha');
  });

  it('falls back to mem0 when the thunk returns undefined', () => {
    configureWithBackend(() => undefined);
    expect(getMemoryBackend().name).toBe('mem0');
  });

  it('accepts a thunk returning a direct instance', () => {
    const inst = fakeBackend('direct');
    configureWithBackend(() => inst);
    expect(getMemoryBackend()).toBe(inst);
  });

  it('throws loud on a thunk returning an unregistered name', () => {
    configureWithBackend(() => 'nope-not-registered');
    expect(() => getMemoryBackend()).toThrow(/unknown memory backend 'nope-not-registered'/);
  });
});
