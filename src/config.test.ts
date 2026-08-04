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
 *  4. FORK-SAFETY — the host lives in realm-pinned state keyed
 *     '@papercusp/memory:host', NOT a module-level singleton (the tsx symlink
 *     double-load that config.ts's own docstring warns about). This test PINS
 *     the property that matters: a value written by configureMemory is seen by
 *     a SECOND module record, and a value written by that second record is seen
 *     here.
 *  5. memorySchema() = host.schema ?? 'public' — nullish default, a set schema
 *     passes through.
 *
 * No vi.mock: config.ts has no runtime deps beyond the pin primitive. We drive
 * the REAL pinned state and reset it around every test so no state leaks.
 *
 * ⚠ This file used to poke `globalThis[Symbol.for('@papercusp/memory:host')]`
 * directly — the storage LOCATION rather than the module's own seam. When the
 * storage moved into @papercusp/module-singleton (EI-19469900474673886) those
 * writes would have kept compiling and kept "passing" while resetting nothing.
 * Reset and plant through the module's exported hooks, never past them.
 *
 * Run: cd libs/generic/memory && npx vitest run src/config.test.ts
 */
import { describe, it, expect, beforeEach, afterAll } from 'vitest';
import { listPinnedModuleKeys, pinModuleState } from '@papercusp/module-singleton';
import {
  __setMemoryHostForTests,
  configureMemory,
  memoryHost,
  isMemoryConfigured,
  memorySchema,
  type MemoryHost,
} from './config';

/** The pinned key config.ts reports under — asserted below, not assumed. */
const STATE_KEY = '@papercusp/memory:host';

/** What a duplicate module record gets on evaluation: the SAME state object. */
function asSecondModuleRecord(): { host: MemoryHost | null | undefined } {
  return pinModuleState<{ host: MemoryHost | null | undefined }>(STATE_KEY, () => ({
    host: undefined,
  }));
}

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

/** Snapshot whatever the worker's pinned slot held, so we can restore it. */
const savedHost = isMemoryConfigured() ? memoryHost() : undefined;

beforeEach(() => {
  // Start every case UNCONFIGURED (isMemoryConfigured checks `!= null`).
  __setMemoryHostForTests(undefined);
});

afterAll(() => {
  // Be a good worker citizen — leave the slot as we found it.
  __setMemoryHostForTests(savedHost);
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
    __setMemoryHostForTests(null);
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

describe('fork-safety — the host is realm-pinned, not a module singleton', () => {
  it('registers its key with @papercusp/module-singleton (so a split here is REPORTED)', () => {
    // Pinning through the primitive rather than hand-rolling a Symbol.for slot
    // is what puts this module in `listModuleDuplications()`. A hand-rolled pin
    // is equally correct and completely invisible there, so the realm-wide
    // report would answer a clean `[]` while this module is duplicated — the
    // "detector cannot see its subject" shape (EI-19469900474673886).
    expect(listPinnedModuleKeys()).toContain(STATE_KEY);
  });

  it('configureMemory writes state a SECOND module record can read', () => {
    const h = host();
    configureMemory(h);
    expect(asSecondModuleRecord().host).toBe(h);
  });

  it('a host planted by a second module record is seen by memoryHost() (the shared-slot read)', () => {
    const h = host();
    asSecondModuleRecord().host = h; // what a forked module instance's configureMemory does
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
