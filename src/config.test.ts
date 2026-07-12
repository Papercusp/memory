/**
 * config.test.ts — the host seam for @papercusp/memory (task #36 hardening).
 * config.ts had NO test, yet EVERY store surface reads through memoryHost():
 * mem0-connection, the backend registry, pre-turn injection, the user routes.
 * Its invariants are small but load-bearing for a public release:
 *
 *  1. UNCONFIGURED THROWS LOUD — memoryHost() before configureMemory() throws
 *     the "not configured — call configureMemory({…})" error, NOT undefined
 *     that surfaces later as an opaque property access on nothing.
 *  2. isMemoryConfigured() is the non-throwing probe the registry/degrade
 *     paths branch on (true iff a host is wired).
 *  3. LAST-CALL-WINS — configureMemory is idempotent; the newest host is what
 *     memoryHost() returns, by reference (no clone).
 *  4. FORK-SAFETY — the host lives on the Symbol.for('@papercusp/memory:host')
 *     process-global, NOT a module-level singleton (the tsx symlink double-load
 *     that config.ts's own docstring warns about). This test PINS the slot
 *     location: a value written by configureMemory is readable at that exact
 *     global key, and a value planted there is seen by memoryHost().
 *  5. memorySchema() = host.schema ?? 'public' — nullish default, a set schema
 *     passes through.
 *
 * No vi.mock: config.ts has no runtime deps (type-only imports). We drive the
 * REAL global slot and reset it around every test so no state leaks.
 *
 * Run: cd libs/generic/memory && npx vitest run src/config.test.ts
 */
import { describe, it, expect, beforeEach, afterAll } from 'vitest';
import {
  configureMemory,
  memoryHost,
  isMemoryConfigured,
  memorySchema,
  type MemoryHost,
} from './config';

const HOST_KEY = Symbol.for('@papercusp/memory:host');
type HostGlobal = typeof globalThis & { [HOST_KEY]?: MemoryHost | null };

/** A minimal valid host; override just the field under test. */
function host(over: Partial<MemoryHost> = {}): MemoryHost {
  return {
    getAdminUrl: () => 'postgres://localhost/test',
    getCredentials: async () => ({}),
    resolveEmbedder: async () => ({ mode: 'disabled' }),
    buildEmbedderForMode: async () => async (_t: string) => [],
    ...over,
  };
}

/** Snapshot whatever the worker's global slot held, so we can restore it. */
const savedSlot = (globalThis as HostGlobal)[HOST_KEY];

beforeEach(() => {
  // Start every case UNCONFIGURED (isMemoryConfigured checks `!= null`).
  (globalThis as HostGlobal)[HOST_KEY] = undefined;
});

afterAll(() => {
  // Be a good worker citizen — leave the slot as we found it.
  (globalThis as HostGlobal)[HOST_KEY] = savedSlot;
});

describe('isMemoryConfigured / memoryHost — unconfigured state', () => {
  it('reports not-configured before any configureMemory call', () => {
    expect(isMemoryConfigured()).toBe(false);
  });

  it('memoryHost() throws a LOUD, actionable error when unconfigured (not undefined)', () => {
    expect(() => memoryHost()).toThrow(/not configured/);
    expect(() => memoryHost()).toThrow(/configureMemory/);
  });

  it('a null slot (explicitly cleared) also reads as not-configured', () => {
    (globalThis as HostGlobal)[HOST_KEY] = null;
    expect(isMemoryConfigured()).toBe(false);
    expect(() => memoryHost()).toThrow(/not configured/);
  });
});

describe('configureMemory — wires the host', () => {
  it('flips isMemoryConfigured to true and returns the SAME host by reference', () => {
    const h = host();
    configureMemory(h);
    expect(isMemoryConfigured()).toBe(true);
    expect(memoryHost()).toBe(h); // no clone — the exact object
  });

  it('is idempotent — the LAST call wins', () => {
    const a = host({ defaultDbName: 'a' });
    const b = host({ defaultDbName: 'b' });
    configureMemory(a);
    configureMemory(b);
    expect(memoryHost()).toBe(b);
    expect(memoryHost().defaultDbName).toBe('b');
  });
});

describe('fork-safety — the host lives on the Symbol.for process-global, not a module singleton', () => {
  it('configureMemory writes to the exact Symbol.for global slot', () => {
    const h = host();
    configureMemory(h);
    expect((globalThis as HostGlobal)[HOST_KEY]).toBe(h);
  });

  it('a host planted directly on that global slot is seen by memoryHost() (the shared-slot read)', () => {
    const h = host();
    (globalThis as HostGlobal)[HOST_KEY] = h; // simulate a second forked module instance
    expect(isMemoryConfigured()).toBe(true);
    expect(memoryHost()).toBe(h);
  });
});

describe('memorySchema — host.schema ?? "public"', () => {
  it('defaults to public when the host sets no schema', () => {
    configureMemory(host());
    expect(memorySchema()).toBe('public');
  });

  it('passes a configured schema through (the operator uses harness_shared)', () => {
    configureMemory(host({ schema: 'harness_shared' }));
    expect(memorySchema()).toBe('harness_shared');
  });

  it('throws (via memoryHost) when unconfigured — schema is not resolvable with no host', () => {
    expect(() => memorySchema()).toThrow(/not configured/);
  });
});
