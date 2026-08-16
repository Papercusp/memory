# @papercusp/memory

A host-agnostic **persistent-memory store behind a swappable
`MemoryBackend` seam**. Consumers call `getMemoryBackend()` and five
neutral verbs (`remember` / `search` / `list` / `forget` / `update`,
plus `get` and an `available()` probe) over a neutral entry shape
`{ id, text, kind?, scope, score?, metadata? }` — which store actually
holds the facts is a config flip, not a caller concern.

Shipped backends:

- **`mem0`** (default) — wraps [`mem0ai/oss`](https://github.com/mem0ai/mem0)
  with a canonical pgvector store — one `memory_canonical` row per fact,
  plus a per-embedder-mode vector table (`memory_vec_openai` /
  `memory_vec_local`) joined by `memory_id` — so switching embedder modes
  re-embeds without duplicating the fact text. Includes a
  worker-thread-isolated local BGE embedder and a cross-model re-embed
  pass. Implements the optional capabilities (`rememberConversation` —
  LLM fact-extraction over a chat window — and `invalidate`).
- **`noop`** — the deliberate "no store": reads come back empty, writes
  throw `MemoryUnavailableError('memory_backend_disabled')`, `available()`
  reports `{ ok: false }`. Makes "memory is off" a clean, testable state.
- **`claude-file`** — reads/writes Claude Code's topic-file memory
  (`~/.claude/projects/<project>/memory/*.md`) through the same seam
  (see `claude-file-backend.ts`).

Out-of-lib backends register a factory and become selectable by name:

```ts
import { registerMemoryBackend, getMemoryBackend } from '@papercusp/memory';

registerMemoryBackend('my-store', () => new MyBackend());
configureMemory({ …, backend: 'my-store' }); // or a direct instance
const entries = await getMemoryBackend().search('q', { scope: 'user-1' });
```

The store core carries **no operator coupling**. Everything host-specific
is injected once via `configureMemory()` — including the `backend` choice
(the Papercusp operator feeds it from `PAPERCUSP_MEMORY_BACKEND`).

## The host seam — `configureMemory()`

```ts
import { configureMemory } from '@papercusp/memory';

configureMemory({
  // Where the embedded/admin Postgres lives (replaces a discovery file).
  getAdminUrl: () => resolvePgAdminUrl(),
  // LLM creds for mem0's fact-extraction (Anthropic Haiku → OpenAI mini).
  getCredentials: async () => ({ anthropic_api_key, openai_api_key }),
  // Resolve the embedder for the current user preference
  // (the openai → local → disabled cascade). Returns a pre-built embed
  // fn + its `mode` (drives the per-model vec table) + `dims`.
  resolveEmbedder: async () => ({ mode: 'openai', dims: 384, embed }),
  // Build an embedder for an EXPLICIT mode (the re-embed pass needs this).
  buildEmbedderForMode: async (mode) => embedFnForMode(mode),

  // ── Storage knobs (all optional; defaults make it borrowable as-is) ──
  schema: 'public',          // PG schema holding the memory tables
  defaultDbName: 'postgres', // admin-URL path fallback
  localStoreDir: undefined,  // dir for mem0's SQLite history; default OS
                             // tmpdir, `null` → in-memory (`:memory:`)

  // Optional: adaptive extraction instructions (black box to the package).
  getLearningInstructions: async () => undefined,
});
```

mem0 owns its own `pg.Client`, so — unlike `@papercusp/backup` /
`@papercusp/search` — there is **no** injected postgres-js `sql` handle.
The PG seam is just an admin-URL resolver; `mem0-connection` parses it
into the discrete fields mem0's PGVector provider wants. The table names
(`memory_canonical`, `memory_vec_*`) are the package's own; the **schema**
that holds them is yours to choose. Your migration must create the tables
in that schema (the package's `initialize()` is a no-op — it ships no DDL).

## What stays in the host

The discovery-file lookup, the embedder cascade, the credential store, the
session→user resolution, and the pre-turn memory-context injection are all
host concerns. The Papercusp operator wires them in
`apps/operator/lib/memory/configure.ts` — passing `schema: 'harness_shared'`,
`defaultDbName: 'papercusp'`, `localStoreDir: ~/.papercusp`.

## Extraction status

Extracted per `papercusp-systems-abstraction-2026-05-29`, item P-021. The
storage knobs (`schema` / `defaultDbName` / `localStoreDir`) were added in
the 2026-05-31 generalization follow-on (D-011) so the package names no
project.
