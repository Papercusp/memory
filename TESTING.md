# Testing — @papercusp/memory

**Run:** `npx vitest run` (from this dir) — Vitest, node env. The unit suite
needs **no Postgres, Docker, or network**: the Mem0Backend tests run against a
stubbed client, the claude-file tests against temp dirs, and the bench-engine
tests against temp-dir claude-file + noop backends. (Live PG/embedder coverage
lives with the host — see "What's NOT covered".)

## What's covered

| File | Covers |
|---|---|
| `src/backend.test.ts` | The neutral seam: `scopesOf`, `extractAddedIds` / `extractStoredEventCount`, `NoopBackend`, `Mem0Backend` over a stubbed mem0 client (incl. `verbatim` → `infer:false` mapping + the nested `metadata.event` wire shape, D-008), backend registry/selector. |
| `src/backend-thunk.test.ts` | `getMemoryBackend` thunk resolution + caching. |
| `src/claude-file-backend.test.ts` | The topic-file bridge: serialize/parse round-trip, writes (id slugging, uniqueness), reads (scope model, archive/skills exclusion), forget-archives-never-deletes, selector flip. |
| `src/mem0-client.test.ts` | `patchEmbedderFactory` (mem0ai 3.x custom-embedder compat) + `resolveExtractionLlmConfig` — the stale-Anthropic-key cascade to the OpenAI extractor (memory-backend-benchmark D-007). |
| `src/extraction-llm.test.ts` | The injectable extraction-LLM seam (mem0-extraction-via-claude-session D-003/D-004/D-005): the mem0ai 3.x custom-LLM conformance pin (`patchLlmFactory` + a REAL `Memory.add()` driven end-to-end through a custom LLM), `resolveExtractionLlmConfig`'s session rung #1 (key probe skipped when the host supplies an adapter), and `FallbackExtractionLlm`'s loud-demotion cascade — `ExtractionAuthError` → sticky demote, non-auth failure → per-call fallback, no-fallback → rethrow WITH the warning fired (the loudness contract: never a silent no-op). |
| `src/local-embedder-worker.test.ts` | The worker-protocol contract for the local embedder. |
| `src/bench/bench.test.ts` | The backend-parameterized bench engine (memory-backend-benchmark D-001/D-010): rank metrics (P@5/R@10/MRR, FP@5), `seedCorpus` manifests + failure honesty, `runGoldSet` key resolution, `runRoundtrips` (store/paraphrase/update/forget/near-dup), the claude-file backend driven through the whole engine, `generateSyntheticCorpus` determinism, scorecard rendering. The NoopBackend rows double as the engine's control (a no-op must score zero). |

## What's NOT covered here

- **Live mem0 against real PG + a real embedder** — the host's concern:
  `packages/operator-core/lib/memory/` (suite/checks.ts, `/admin/testing`
  Memory domain) and the live bench runner
  (`packages/operator-core/lib/memory/bench/`, memory-backend-benchmark
  P-006/P-007).
- **The frozen real-corpus/gold-set fixtures and index-cap probe** — also
  operator-core (`lib/memory/bench/*.test.ts`).
- The `reembed` / `canonical-store` PG paths (exercised via the host's
  integration suite).

## After editing

Run `npx vitest run` here. If you touch the seam types (`backend.ts`) or the
bench engine's exports, also run the operator-core memory tests
(`cd packages/operator-core && npx vitest run lib/memory`) — the fixtures,
suite, and live bench runner consume them.
