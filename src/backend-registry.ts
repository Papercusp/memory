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
 * Both the registry and the instance cache live on `Symbol.for`-keyed
 * process-global slots — the same fork-safety trick as ./config's host
 * slot (under tsx the package can load twice via the node_modules
 * symlink; module-level singletons would split).
 */

import type { MemoryBackend } from './backend';
import { isMemoryConfigured, memoryHost } from './config';
import { Mem0Backend } from './mem0-backend';
import { NoopBackend } from './noop-backend';

type BackendFactory = () => MemoryBackend;

const REGISTRY_KEY = Symbol.for('@papercusp/memory:backend-registry');
const INSTANCES_KEY = Symbol.for('@papercusp/memory:backend-instances');

type RegistryGlobal = typeof globalThis & {
  [REGISTRY_KEY]?: Map<string, BackendFactory>;
  [INSTANCES_KEY]?: Map<string, MemoryBackend>;
};

function registry(): Map<string, BackendFactory> {
  const g = globalThis as RegistryGlobal;
  if (!g[REGISTRY_KEY]) {
    g[REGISTRY_KEY] = new Map<string, BackendFactory>([
      ['mem0', () => new Mem0Backend()],
      ['noop', () => new NoopBackend()],
    ]);
  }
  return g[REGISTRY_KEY];
}

function instances(): Map<string, MemoryBackend> {
  const g = globalThis as RegistryGlobal;
  if (!g[INSTANCES_KEY]) g[INSTANCES_KEY] = new Map();
  return g[INSTANCES_KEY];
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
  const choice = isMemoryConfigured() ? memoryHost().backend ?? 'mem0' : 'mem0';
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

/** Test hook: drop cached instances + custom registrations (built-ins re-seed lazily). */
export function _resetMemoryBackendsForTest(): void {
  const g = globalThis as RegistryGlobal;
  delete g[INSTANCES_KEY];
  delete g[REGISTRY_KEY];
}
