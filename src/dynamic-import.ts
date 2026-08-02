/**
 * dynamic-import.ts — ONE runtime-opaque dynamic import for the whole memory
 * stack (context-injection-audit-2026-07-28 P-002).
 *
 * ## What this replaces, and why it is not a style preference
 *
 * Six modules here each declared their own copy of:
 *
 *     const dynamicImport = new Function('specifier', 'return import(specifier)');
 *
 * The intent is sound and is preserved below: `@huggingface/transformers` and
 * `mem0ai` are OPTIONAL, lazily-resolved dependencies, so the import must be
 * invisible to a bundler's static analysis or the bundle hard-requires a package
 * that is legitimately absent.
 *
 * The defect is the mechanism. A function built by `new Function` is compiled in
 * a realm that, in several embedders (Vitest's module runner among them), has NO
 * host dynamic-import callback — so `import()` inside it throws
 *
 *     TypeError: A dynamic import callback was not specified.
 *
 * **regardless of whether the package is installed.** Every call site wrapped
 * that in a bare `catch` and reported a boolean or a degraded mode, so the
 * failure was indistinguishable from "the optional dependency is absent":
 *
 *   - `configure.ts`'s `localAvailable()` returned `false`, and the embedder
 *     cascade reported `harrier_forced_but_transformers_not_installed` on a box
 *     where the package imports fine from plain node;
 *   - `mem0-client.ts` surfaced `[mem0] A dynamic import callback was not specified.`
 *     and seeded 0/114 corpus entries.
 *
 * The consequence is bigger than a confusing message: it made the ENTIRE local
 * embedder + mem0 client path unreachable from any test that runs under Vitest.
 * `mem0-client.test.ts` says so in its own header — it cannot drive
 * `getMemoryClient()` end-to-end and asserts at a seam instead. So the one
 * environment that reproduces the bug is the one environment that could never
 * exercise the code, and a P-002-style recurrence guard (a real integration test
 * over the real corpus) could not be written at all.
 *
 * ## Why the fallback is still bundler-opaque
 *
 * `specifier` is a RUNTIME VALUE, so a static analyzer has no literal to follow
 * and cannot pull the optional package into a bundle — the same property the
 * `new Function` trick was bought for. `@vite-ignore` additionally silences
 * Vite's dynamic-import warning. The `new Function` form is kept as the primary
 * path so behavior in production (tsx/node, where it works) is byte-identical;
 * the fallback fires ONLY on the specific no-callback TypeError, so a genuine
 * module-resolution failure still propagates unchanged rather than being
 * retried into a confusing second error.
 */

const newFunctionImport = new Function('specifier', 'return import(specifier)') as <T>(
  specifier: string,
) => Promise<T>;

/** The message Node/V8 uses when a realm has no host import callback. */
const NO_IMPORT_CALLBACK = /dynamic import callback was not specified/i;

/**
 * Import an optional dependency by a runtime-valued specifier, without exposing
 * it to bundler static analysis and without falsely reporting "not installed"
 * inside a VM realm.
 *
 * Throws the module's own resolution error when the package is genuinely
 * missing — callers that want a boolean must catch it themselves (and should
 * consider whether "absent" and "unreachable" deserve the same answer).
 */
export async function dynamicImport<T>(specifier: string): Promise<T> {
  try {
    return await newFunctionImport<T>(specifier);
  } catch (err) {
    if (!NO_IMPORT_CALLBACK.test((err as Error)?.message ?? '')) throw err;
    return (await import(/* @vite-ignore */ specifier)) as T;
  }
}
