/**
 * Claude Code topic-file format — the on-disk shape of the native Claude
 * Code file memory (`~/.claude/projects/<project-slug>/memory/*.md`).
 *
 * One file per fact: a small YAML-ish frontmatter (name / description /
 * metadata.type) followed by the markdown body. The format is CONSUMED by
 * the user-env index projector (`~/.claude/scripts/memory-compact.mjs`,
 * plan fix-claude-code-memory-2026-06-05), so this module deliberately
 * mirrors that script's parsing rules — a file we serialize must project
 * into the regenerated MEMORY.md index unchanged:
 *
 *   - frontmatter = the FIRST `---\n…\n---` block
 *   - `name:` / `description:` are top-level single-line scalars
 *     (description may be double- or single-quoted; quotes are stripped)
 *   - `type:` is either top-level or nested under `metadata:`
 *   - everything after the block is the body
 *
 * On top of the native keys, the neutral-backend fields (`scope`, `kind`)
 * and passthrough metadata (`extra:` as one-line JSON) are persisted under
 * `metadata:` — unknown keys there are ignored by the index projector, so
 * the file stays fully hook-compatible.
 *
 * Pure: no I/O. (claude-memory-projection-integration-2026-06-05 P-004.)
 */

/** Claude memory types. Durable types are never auto-evicted by the index
 *  projector; `project` entries are subject to derivable-state eviction. */
export const CLAUDE_MEMORY_TYPES = ['user', 'feedback', 'reference', 'project'] as const;
export type ClaudeMemoryType = (typeof CLAUDE_MEMORY_TYPES)[number];

export interface TopicFile {
  /** The memory's stable slug — also the filename (minus `.md`) and the id. */
  name: string;
  /** The one-line index hook (projected into MEMORY.md, truncated there). */
  description: string;
  /** Claude memory type — drives index grouping + eviction policy. */
  type: ClaudeMemoryType;
  /** Neutral-backend pool this entry belongs to (absent on hand-curated files). */
  scope?: string;
  /** Neutral-backend kind tag (absent on hand-curated files). */
  kind?: string;
  /** Passthrough metadata (JSON-serializable), persisted as one-line JSON. */
  extra?: Record<string, unknown>;
  /** The fact body (markdown). */
  body: string;
}

const FRONTMATTER_RE = /^---\n([\s\S]*?)\n---/;

function scalar(block: string, re: RegExp): string {
  const m = re.exec(block);
  return m ? m[1].trim() : '';
}

/** Strip one layer of matching surrounding quotes + YAML escapes (mirrors
 *  memory-compact.mjs `clean()`). */
function clean(s: string): string {
  if (!s) return '';
  s = s.trim();
  if (s.length >= 2 && s.startsWith('"') && s.endsWith('"')) {
    s = s.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  } else if (s.length >= 2 && s.startsWith("'") && s.endsWith("'")) {
    s = s.slice(1, -1).replace(/''/g, "'");
  }
  return s.replace(/\s+/g, ' ').trim();
}

function coerceType(raw: string): ClaudeMemoryType {
  const t = raw.toLowerCase();
  return (CLAUDE_MEMORY_TYPES as readonly string[]).includes(t) ? (t as ClaudeMemoryType) : 'project';
}

/** Parse one topic file. Lenient: a file with no/partial frontmatter still
 *  yields a usable record (the index projector is equally lenient). */
export function parseTopicFile(text: string): TopicFile {
  const normalized = text.replace(/\r\n/g, '\n');
  const m = FRONTMATTER_RE.exec(normalized);
  const fm = m ? m[1] : '';
  const body = (m ? normalized.slice(m[0].length) : normalized).replace(/^\n+/, '');

  const name = clean(scalar(fm, /^name:[ \t]*(.+)$/m));
  const description = clean(scalar(fm, /^description:[ \t]*(.+)$/m));
  // type: top-level wins, then metadata-nested — same precedence as the hook.
  const rawType = scalar(fm, /^type:[ \t]*(.+)$/m) || scalar(fm, /^[ \t]+type:[ \t]*(.+)$/m);
  const scope = clean(scalar(fm, /^[ \t]+scope:[ \t]*(.+)$/m)) || undefined;
  const kind = clean(scalar(fm, /^[ \t]+kind:[ \t]*(.+)$/m)) || undefined;

  let extra: Record<string, unknown> | undefined;
  const rawExtra = scalar(fm, /^[ \t]+extra:[ \t]*(.+)$/m);
  if (rawExtra) {
    try {
      const parsed = JSON.parse(rawExtra);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        extra = parsed as Record<string, unknown>;
      }
    } catch {
      /* malformed extra is dropped, never fatal */
    }
  }

  return { name, description, type: coerceType(clean(rawType)), scope, kind, extra, body };
}

/** One-line, double-quoted YAML scalar (what the hook's `clean()` unquotes). */
function quoted(s: string): string {
  return '"' + s.replace(/\s+/g, ' ').trim().replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

/** Serialize a topic file in the canonical hook-compatible shape. */
export function serializeTopicFile(tf: TopicFile): string {
  const lines = ['---', `name: ${tf.name}`, `description: ${quoted(tf.description)}`, 'metadata:', '  node_type: memory', `  type: ${tf.type}`];
  if (tf.scope) lines.push(`  scope: ${tf.scope}`);
  if (tf.kind) lines.push(`  kind: ${tf.kind}`);
  if (tf.extra && Object.keys(tf.extra).length > 0) lines.push(`  extra: ${JSON.stringify(tf.extra)}`);
  lines.push('---', '', tf.body.trim(), '');
  return lines.join('\n');
}

/**
 * Default neutral-kind → Claude-type mapping. Only kinds the operator
 * treats as durable land in durable (never-auto-evicted) types; everything
 * unknown lands in `project`, where the index projector's derivable-state
 * eviction can bound it — the bloat-safe default per the D-002 boundary
 * (derivable → plans/projections; only learned facts stay durable).
 */
export function typeForKind(kind: string | undefined): ClaudeMemoryType {
  switch ((kind ?? '').toLowerCase()) {
    case 'user': // unified taxonomy (memory-taxonomy-and-debt-followups D-001)
    case 'identity': // legacy mem0 kind
      return 'user';
    case 'preference':
    case 'correction':
    case 'feedback':
      return 'feedback';
    case 'reference':
      return 'reference';
    default:
      return 'project';
  }
}

/** Filename-safe slug for a new topic file (lowercase snake_case). */
export function slugify(text: string, maxWords = 7): string {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]+/g, ' ')
    .split(/[\s_-]+/)
    .filter(Boolean)
    .slice(0, maxWords);
  return words.join('_') || 'memory';
}

/** Derive the one-line index hook from a fact body: first sentence/line,
 *  bounded. (The index projector truncates again at render; this keeps the
 *  SOURCE description tight per the memory guidance.) */
export function deriveDescription(text: string, maxChars = 140): string {
  const firstLine = text
    .split('\n')
    .map((l) => l.trim())
    .find((l) => l && !l.startsWith('#') && !l.startsWith('---')) ?? '';
  const sentence = /^(.{10,}?[.!?])\s/.exec(firstLine + ' ')?.[1] ?? firstLine;
  const flat = sentence.replace(/\s+/g, ' ').trim();
  return flat.length <= maxChars ? flat : flat.slice(0, maxChars - 1).trimEnd() + '…';
}

/**
 * The Claude Code project-slug rule: a project's directory maps to
 * `~/.claude/projects/<dir with / and . replaced by ->/memory`. Mirrors
 * the derivation in memory-compact.mjs `resolveMemoryDir()`.
 */
export function claudeProjectMemoryDir(projectDir: string, claudeHome: string): string {
  const slug = projectDir.replace(/[/.]/g, '-');
  return `${claudeHome.replace(/\/+$/, '')}/projects/${slug}/memory`;
}
