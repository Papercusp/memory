/**
 * Injectable extraction-LLM seam (mem0-extraction-via-claude-session
 * D-003): the host may supply its own LLM implementation for mem0's
 * fact-extraction step — e.g. the operator's `SessionExtractionLlm`,
 * which rides the Claude-session `anthropic-direct` transport with no
 * API key. This module is domain-free: it only knows the mem0ai LLM
 * interface shape and a generic primary→fallback cascade; everything
 * credential/transport-specific lives host-side behind
 * `MemoryHost.getExtractionLlm`.
 *
 * The interface mirrors mem0ai 3.x's `LLM` (pinned by the conformance
 * test in extraction-llm.test.ts): `Memory.add()` calls
 * `generateResponse(messages, { type: 'json_object' })` and expects a
 * STRING back, which it feeds through its own `extractJson` + Zod parse.
 */

export interface ExtractionLlmMessage {
  role: string;
  content: string;
}

export interface ExtractionLlmResponse {
  content: string;
  role: string;
  toolCalls?: Array<{ name: string; arguments: string }>;
}

/**
 * mem0ai 3.x's custom-LLM contract. `generateResponse` returns the raw
 * completion text (mem0 parses it); `generateChat` is the structured
 * variant (unused by the OSS additive-extraction path, kept for
 * interface completeness).
 */
export interface ExtractionLlm {
  generateResponse(
    messages: ExtractionLlmMessage[],
    responseFormat?: { type: string },
    tools?: unknown[],
  ): Promise<unknown>;
  generateChat(messages: ExtractionLlmMessage[]): Promise<ExtractionLlmResponse>;
}

/**
 * Thrown by a host extraction-LLM when its credential is rejected
 * (401/403) and its own refresh+retry didn't recover. Distinguished from
 * generic failures because the cascade treats it as a STICKY demotion
 * (D-004): the session rung is dead for this process lifetime, not just
 * this call.
 */
export class ExtractionAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ExtractionAuthError';
  }
}

/**
 * Primary→fallback cascade over two extraction LLMs (D-002/D-004/D-005).
 *
 * Reliability semantics, not best-effort: mem0 swallows an LLM throw
 * into `[]` (a silently dropped memory — the exact "stillborn" class),
 * so a failed primary call must fall THROUGH to the fallback within the
 * same call, never bubble to mem0 while a working rung exists.
 *
 *   - primary throws `ExtractionAuthError` → warn loud + STICKY demote
 *     (process lifetime; the host re-probes on next boot) + serve this
 *     and every later call from the fallback.
 *   - primary throws anything else → warn (once per error class) + serve
 *     THIS call from the fallback; the primary is retried next call.
 *   - no fallback available → rethrow (mem0 logs the failed extraction;
 *     the warning has already fired, so the failure is never silent).
 */
export class FallbackExtractionLlm implements ExtractionLlm {
  private demoted = false;
  private fallbackPromise: Promise<ExtractionLlm | null> | undefined;

  constructor(
    private readonly primary: ExtractionLlm,
    /** Lazily build the key-rung LLM. Only invoked on first demotion. */
    private readonly buildFallback: () => Promise<ExtractionLlm | null>,
    private readonly opts: {
      warn: (reason: string) => void;
      primaryLabel?: string;
    },
  ) {}

  /** Test/diagnostic hook: has the primary been sticky-demoted? */
  get isDemoted(): boolean {
    return this.demoted;
  }

  private label(): string {
    return this.opts.primaryLabel ?? 'session extraction LLM';
  }

  private async fallback(): Promise<ExtractionLlm | null> {
    // Memoized: the key-rung resolution probes the Anthropic key, so
    // don't redo it per extraction call.
    this.fallbackPromise ??= this.buildFallback().catch((e) => {
      this.opts.warn(
        `extraction fallback rung failed to build: ${(e as Error).message}`,
      );
      return null;
    });
    return this.fallbackPromise;
  }

  private async route<T>(
    viaPrimary: () => Promise<T>,
    viaFallback: (fb: ExtractionLlm) => Promise<T>,
  ): Promise<T> {
    if (!this.demoted) {
      try {
        return await viaPrimary();
      } catch (e) {
        const err = e as Error;
        if (err instanceof ExtractionAuthError) {
          this.demoted = true;
          this.opts.warn(
            `${this.label()} auth-rejected after refresh+retry — demoting to API-key extraction for this process lifetime (re-probed next boot): ${err.message}`,
          );
        } else {
          this.opts.warn(
            `${this.label()} failed (${err.message.slice(0, 140)}) — serving this extraction from the API-key rung`,
          );
        }
        const fb = await this.fallback();
        if (!fb) {
          this.opts.warn(
            `no API-key extraction rung available — this extraction is LOST (add a key at /settings/api-keys or restore the Claude session)`,
          );
          throw err;
        }
        return viaFallback(fb);
      }
    }
    const fb = await this.fallback();
    if (!fb) {
      throw new ExtractionAuthError(
        `${this.label()} demoted and no API-key extraction rung available`,
      );
    }
    return viaFallback(fb);
  }

  generateResponse(
    messages: ExtractionLlmMessage[],
    responseFormat?: { type: string },
    tools?: unknown[],
  ): Promise<unknown> {
    return this.route(
      () => this.primary.generateResponse(messages, responseFormat, tools),
      (fb) => fb.generateResponse(messages, responseFormat, tools),
    );
  }

  generateChat(messages: ExtractionLlmMessage[]): Promise<ExtractionLlmResponse> {
    return this.route(
      () => this.primary.generateChat(messages),
      (fb) => fb.generateChat(messages),
    );
  }
}
