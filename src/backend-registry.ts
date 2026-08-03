/**
 * Backend registry + selector — the choke point that makes the memory
 * store a config flip (generalize-memory-backend-swappable D-004).
 *
 * `getMemoryBackend()` is what every memory consumer calls (the
 * operator's `memory:*` tool handlers, the user routes, pre-turn
 * injection, …). Which backend it returns is decided by the host:
 *
 *   configureMemory({ …, backend: 'noop' })          // built-in by name
 *   configureMemory({ …, backend: new MyBackend() }) // any instance
 *
 * The operator feeds `backend` from `PAPERCUSP_MEMORY_BACKEND` (see
 * apps/operator's lib/memory/configure.ts); the default is `'mem0'`.
 *
 * Out-of-lib backends (e.g. a Claude-topic-file bridge) register a
 * factory under a name BEFORE first use:
 *
 *   registerMemoryBackend('claude-file', () => new ClaudeFileMemoryBackend(…));
 *
 * and become selectable by that name with zero handler changes.
 *
 * Both the registry and the instance cache live in ONE state object pinned
 * to the realm by `@papercusp/module-singleton` — the same fork-safety trick
 * as ./config's host slot (under tsx the package can load twice via the
 * node_modules symlink; module-level singletons would split).
 */

import { pinModuleState } from '@papercusp/module-singleton';
import type { MemoryBackend } from './backend';
import { isMemoryConfigured, memoryHost } from './config';
import { Mem0Backend } from './mem0-backend';
import { NoopBackend } from './noop-backend';

type BackendFactory = () => MemoryBackend;

/**
 * The pin key — also the id this module reports under in
 * `listModuleDuplications()`, so a split here is visible in the REALM-WIDE
 * report rather than only through this module's own accessors.
 *
 * This module previously hand-rolled TWO `Symbol.for` slots
 * (`:backend-registry` + `:backend-instances`). They are now ONE pinned state
 * object under the first of those two strings, because `evaluations` is only a
 * module-RECORD count when `pinModuleState` is called exactly once per module
 * evaluation — two keys from one module body would double-count the same
 * record. Nothing outside this file ever referenced either symbol
 * (EI-19469900474673886 verified this repo-wide before collapsing them).
 */
const STATE_KEY = '@papercusp/memory:backend-registry';

interface BackendRegistryState {
  registry: Map<string, BackendFactory>;
  instances: Map<string, MemoryBackend>;
}

/** The built-ins, re-seeded on reset. Factories, not instances — nothing is constructed here. */
function builtinFactories(): Map<string, BackendFactory> {
  return new Map<string, BackendFactory>([
    ['mem0', () => new Mem0Backend()],
    ['noop', () => new NoopBackend()],
  ]);
}

/**
 * Pinned + counted rather than hand-rolled. Must stay at module scope: moving
 * this inside a function would make `evaluations` count CALLS instead of module
 * records, and the duplication report would become meaningless for this key.
 */
const state = pinModuleState<BackendRegistryState>(STATE_KEY, () => ({
  registry: builtinFactories(),
  instances: new Map<string, MemoryBackend>(),
}));

function registry(): Map<string, BackendFactory> {
  return state.registry;
}

function instances(): Map<string, MemoryBackend> {
  return state.instances;
}

/**
 * Register (or replace) a named backend factory. Instances are built
 * lazily on first selection and cached per name.
 */
export function registerMemoryBackend(name: string, factory: BackendFactory): void {
  registry().set(name, factory);
  instances().delete(name); // a re-registration invalidates the cached instance
}

/** Names currently selectable (diagnostics / error messages). */
export function registeredMemoryBackends(): string[] {
  return [...registry().keys()];
}

/**
 * Resolve the active backend per the host's `backend` choice.
 * Default `'mem0'` (also when no host is configured yet — the mem0
 * backend then reports unavailable rather than this accessor throwing).
 * An unknown NAME throws loud — a silent fallback would mask a typo'd
 * `PAPERCUSP_MEMORY_BACKEND` as "memory is just empty".
 */
export function getMemoryBackend(): MemoryBackend {
  // The host's `backend` may be a static name/instance OR a thunk re-read
  // on every call (the operator passes a thunk over the live operator
  // setting so a UI flip takes effect without a restart — Brief 30).
  const raw = isMemoryConfigured() ? memoryHost().backend : undefined;
  const resolved = typeof raw === 'function'
    ? (raw as () => string | MemoryBackend | undefined)()
    : raw;
  const choice = resolved ?? 'mem0';
  if (typeof choice !== 'string') return choice; // a direct instance
  const cached = instances().get(choice);
  if (cached) return cached;
  const factory = registry().get(choice);
  if (!factory) {
    throw new Error(
      `unknown memory backend '${choice}' — registered: ${registeredMemoryBackends().join(', ')}. ` +
      'Register it via registerMemoryBackend() before selecting it.',
    );
  }
  const built = factory();
  instances().set(choice, built);
  return built;
}

/**
 * Test hook: drop cached instances + custom registrations (built-ins re-seed).
 *
 * Clears IN PLACE rather than replacing the pinned state object: every module
 * record holds a reference to that one object, so swapping it out would detach
 * the other records — reintroducing the very split the pin exists to close.
 */
export function _resetMemoryBackendsForTest(): void {
  state.instances.clear();
  state.registry.clear();
  for (const [name, factory] of builtinFactories()) state.registry.set(name, factory);
}
