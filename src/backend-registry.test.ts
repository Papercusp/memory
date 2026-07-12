/**
 * backend-registry.test.ts — the memory backend registry + selector (task
 * #36 hardening). backend-registry.ts had NO test, yet it is the choke point
 * EVERY memory consumer routes through (`getMemoryBackend()`), so its
 * invariants are load-bearing for a public release:
 *
 *  1. DEFAULT 'mem0' — an unconfigured host (or a host with no `backend`)
 *     resolves the mem0 store, never throws just for being un-wired.
 *  2. LAZY + PER-NAME CACHE — a factory is built on first selection and the
 *     instance is reused; the factory runs exactly once per name.
 *  3. RE-REGISTRATION INVALIDATES the cached instance (a swapped factory must
 *     actually take effect), and can REPLACE a built-in.
 *  4. THUNK re-read every call — a function `backend` is re-evaluated per
 *     call so a live UI flip switches the store with no restart.
 *  5. DIRECT-INSTANCE passthrough — a non-string choice is returned as-is
 *     (and is not smuggled into the per-name cache).
 *  6. UNKNOWN name throws LOUD, listing the registered names — a typo'd
 *     PAPERCUSP_MEMORY_BACKEND must fail visibly, never degrade to a silent
 *     "memory is just empty".
 *
 * ./config is mocked so isMemoryConfigured/memoryHost are steerable; the REAL
 * Mem0Backend/NoopBackend back the built-ins so the default + 'noop' identity
 * assertions verify the actual lazy seeding. _resetMemoryBackendsForTest()
 * clears the Symbol.for process-globals between cases.
 *
 * Run: cd libs/generic/memory && npx vitest run src/backend-registry.test.ts
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { MemoryBackend } from './backend';

const mocks = vi.hoisted(() => ({
  isMemoryConfigured: vi.fn<() => boolean>(),
  memoryHost: vi.fn<() => { backend?: unknown }>(),
}));

vi.mock('./config', () => ({
  isMemoryConfigured: mocks.isMemoryConfigured,
  memoryHost: mocks.memoryHost,
}));

import {
  registerMemoryBackend,
  registeredMemoryBackends,
  getMemoryBackend,
  _resetMemoryBackendsForTest,
} from './backend-registry';

/** A minimal typed stub backend — identity is all these tests care about. */
function fake(name: string): MemoryBackend {
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

/** Wire the host as configured with a given `backend` choice. */
function hostBackend(backend: unknown): void {
  mocks.isMemoryConfigured.mockReturnValue(true);
  mocks.memoryHost.mockReturnValue({ backend });
}

beforeEach(() => {
  vi.clearAllMocks();
  _resetMemoryBackendsForTest();
  // Default: host NOT configured (the unconfigured-default path).
  mocks.isMemoryConfigured.mockReturnValue(false);
  mocks.memoryHost.mockImplementation(() => {
    throw new Error('memoryHost() called while unconfigured');
  });
});

describe('registeredMemoryBackends — built-ins seed lazily', () => {
  it('a fresh registry exposes exactly the two built-ins', () => {
    expect(registeredMemoryBackends().sort()).toEqual(['mem0', 'noop']);
  });

  it('a registered name is added to the selectable set', () => {
    registerMemoryBackend('claude-file', () => fake('claude-file'));
    expect(registeredMemoryBackends().sort()).toEqual(['claude-file', 'mem0', 'noop']);
  });
});

describe('getMemoryBackend — default is mem0', () => {
  it('an UNCONFIGURED host resolves the mem0 backend (never throws for being un-wired)', () => {
    const b = getMemoryBackend();
    expect(b.name).toBe('mem0');
    expect(mocks.memoryHost).not.toHaveBeenCalled(); // short-circuits on isMemoryConfigured=false
  });

  it('a configured host with NO backend field also defaults to mem0', () => {
    hostBackend(undefined);
    expect(getMemoryBackend().name).toBe('mem0');
  });

  it('the default mem0 instance is cached — same reference across calls', () => {
    const a = getMemoryBackend();
    const b = getMemoryBackend();
    expect(a).toBe(b);
    expect(a.name).toBe('mem0');
  });
});

describe('getMemoryBackend — a static string names a registered backend', () => {
  it("backend:'noop' resolves the real NoopBackend", () => {
    hostBackend('noop');
    expect(getMemoryBackend().name).toBe('noop');
  });

  it('a registered custom name resolves its factory output', () => {
    registerMemoryBackend('claude-file', () => fake('claude-file'));
    hostBackend('claude-file');
    expect(getMemoryBackend().name).toBe('claude-file');
  });
});

describe('getMemoryBackend — lazy build + per-name instance cache (factory runs once)', () => {
  it('builds on first selection, caches, and never re-invokes the factory for the same name', () => {
    const factory = vi.fn(() => fake('claude-file'));
    registerMemoryBackend('claude-file', factory);
    hostBackend('claude-file');

    const first = getMemoryBackend();
    const second = getMemoryBackend();

    expect(factory).toHaveBeenCalledTimes(1); // lazy + cached, not rebuilt per call
    expect(first).toBe(second);
    expect(first.name).toBe('claude-file');
  });

  it('does not build a factory that is registered but never selected', () => {
    const factory = vi.fn(() => fake('claude-file'));
    registerMemoryBackend('claude-file', factory);
    hostBackend('noop'); // select something else
    getMemoryBackend();
    expect(factory).not.toHaveBeenCalled();
  });
});

describe('registerMemoryBackend — re-registration invalidates the cached instance', () => {
  it('a swapped factory actually takes effect (the old cached instance is dropped)', () => {
    const v1 = fake('claude-file');
    const v2 = fake('claude-file');
    const f1 = vi.fn(() => v1);
    const f2 = vi.fn(() => v2);

    registerMemoryBackend('claude-file', f1);
    hostBackend('claude-file');
    expect(getMemoryBackend()).toBe(v1);
    expect(f1).toHaveBeenCalledTimes(1);

    registerMemoryBackend('claude-file', f2); // re-registration must drop the v1 cache
    const after = getMemoryBackend();
    expect(after).toBe(v2);
    expect(f2).toHaveBeenCalledTimes(1);
    expect(f1).toHaveBeenCalledTimes(1); // v1 factory NOT called again
  });

  it('can REPLACE a built-in name (host may override mem0)', () => {
    const custom = fake('mem0-override');
    registerMemoryBackend('mem0', () => custom);
    // default path selects 'mem0' → now resolves the override
    expect(getMemoryBackend()).toBe(custom);
  });
});

describe('getMemoryBackend — a THUNK backend is re-read on every call (live flip, no restart)', () => {
  it('re-evaluates the thunk each call and switches the resolved backend', () => {
    let choice = 'noop';
    const thunk = vi.fn(() => choice);
    hostBackend(thunk);

    expect(getMemoryBackend().name).toBe('noop');
    choice = 'mem0';
    expect(getMemoryBackend().name).toBe('mem0'); // the flip took effect with no re-config
    expect(thunk).toHaveBeenCalledTimes(2); // read once per getMemoryBackend, not cached
  });

  it('a thunk returning undefined falls back to the mem0 default', () => {
    hostBackend(() => undefined);
    expect(getMemoryBackend().name).toBe('mem0');
  });
});

describe('getMemoryBackend — direct-instance passthrough (non-string choice returned as-is)', () => {
  it('a host backend that IS an instance is returned by reference', () => {
    const instance = fake('injected');
    hostBackend(instance);
    expect(getMemoryBackend()).toBe(instance);
  });

  it('a thunk returning an instance is returned as-is, and is NOT smuggled into the name cache', () => {
    const instance = fake('injected');
    hostBackend(() => instance);
    expect(getMemoryBackend()).toBe(instance);
    // the instance was never registered, so it did not join the selectable set
    expect(registeredMemoryBackends()).not.toContain('injected');
  });
});

describe('getMemoryBackend — an UNKNOWN name throws LOUD (no silent empty-memory fallback)', () => {
  it('throws, naming the bad choice AND the registered set', () => {
    hostBackend('typo-backend');
    expect(() => getMemoryBackend()).toThrow(/unknown memory backend 'typo-backend'/);
    expect(() => getMemoryBackend()).toThrow(/registered:.*mem0/);
    expect(() => getMemoryBackend()).toThrow(/registerMemoryBackend/);
  });

  it('the registered list in the error reflects custom registrations', () => {
    registerMemoryBackend('claude-file', () => fake('claude-file'));
    hostBackend('nope');
    expect(() => getMemoryBackend()).toThrow(/registered:.*claude-file/);
  });
});
