/**
 * topic-file.test.ts — the on-disk Claude-Code memory format (task #36 memory
 * public-release hardening). topic-file.ts had NO test, yet it is the SERIALIZER
 * for every fact written to `~/.claude/projects/<slug>/memory/*.md` and the PARSER
 * that reads them back. Its whole contract is a data-integrity + hook-compatibility
 * one: a file we serialize must project through the MEMORY.md index projector
 * (memory-compact.mjs) UNCHANGED, and parse must be lenient enough to survive a
 * hand-curated / partial file without throwing or losing the body.
 *
 * The load-bearing invariants pinned here:
 *  1. ROUND-TRIP: parse(serialize(tf)) === tf EXACTLY for a trimmed body — parse
 *     strips the file's trailing newline(s) that serialize appends, so consumers
 *     never need a defensive .trim(). Special chars in the description (quote,
 *     backslash) and a nested `extra` object survive intact.
 *  2. HOOK-SHAPE: serialize emits frontmatter as the FIRST `---`-block with
 *     top-level `name:`/`description:` and a metadata-nested `type:` — exactly the
 *     precedence parse (and the projector) reads.
 *  3. LENIENT PARSE: no/partial frontmatter still yields a usable record; the FIRST
 *     `---` block is the frontmatter (a body horizontal-rule stays in the body);
 *     malformed `extra` JSON is dropped, never fatal; scope/kind are metadata-nested
 *     only; `type:` top-level wins over nested; an unknown type coerces to `project`.
 *  4. TAXONOMY: typeForKind maps the neutral kinds (incl. legacy `identity`) into the
 *     four Claude types, defaulting unknown → `project` (the bloat-safe D-002 default).
 *  5. DERIVATIONS: slugify (filename-safe, bounded, never-empty), deriveDescription
 *     (first real sentence, char-bounded + ellipsis), claudeProjectMemoryDir (the
 *     project-slug rule) all match memory-compact.mjs.
 *
 * Run: cd libs/generic/memory && npx vitest run src/topic-file.test.ts
 */
import { describe, it, expect } from 'vitest';
import {
  parseTopicFile,
  serializeTopicFile,
  typeForKind,
  slugify,
  deriveDescription,
  claudeProjectMemoryDir,
  CLAUDE_MEMORY_TYPES,
  type TopicFile,
} from './topic-file';

describe('parseTopicFile — well-formed file yields every field', () => {
  it('reads top-level name/description + metadata-nested type/scope/kind/extra', () => {
    const text = [
      '---',
      'name: build_server_window',
      'description: "the build server only deploys 2–4am UTC"',
      'metadata:',
      '  node_type: memory',
      '  type: reference',
      '  scope: owner-42',
      '  kind: reference',
      '  extra: {"orig_file":"notes.md","n":3}',
      '---',
      '',
      'The experimental build server accepts deploys 2–4am UTC only.',
      '',
    ].join('\n');
    const tf = parseTopicFile(text);
    expect(tf.name).toBe('build_server_window');
    expect(tf.description).toBe('the build server only deploys 2–4am UTC');
    expect(tf.type).toBe('reference');
    expect(tf.scope).toBe('owner-42');
    expect(tf.kind).toBe('reference');
    expect(tf.extra).toEqual({ orig_file: 'notes.md', n: 3 });
    // parse strips the separator blank line AND the file's trailing newline(s).
    expect(tf.body).toBe('The experimental build server accepts deploys 2–4am UTC only.');
  });

  it('normalizes CRLF line endings before parsing', () => {
    const text = '---\r\nname: crlf_fact\r\nmetadata:\r\n  type: user\r\n---\r\n\r\nBody line.\r\n';
    const tf = parseTopicFile(text);
    expect(tf.name).toBe('crlf_fact');
    expect(tf.type).toBe('user');
    expect(tf.body).toBe('Body line.');
  });

  it("strips a single layer of surrounding quotes on description (double AND single)", () => {
    expect(parseTopicFile('---\nname: n\ndescription: "double quoted"\n---\nb').description).toBe('double quoted');
    expect(parseTopicFile("---\nname: n\ndescription: 'single quoted'\n---\nb").description).toBe('single quoted');
    // Unquoted is kept verbatim (whitespace-collapsed).
    expect(parseTopicFile('---\nname: n\ndescription: bare words\n---\nb').description).toBe('bare words');
  });
});

describe('parseTopicFile — lenient (a hand-curated / partial file never throws or loses the body)', () => {
  it('a file with NO frontmatter is all-body, type defaults to project', () => {
    const tf = parseTopicFile('just a raw note with no frontmatter at all');
    expect(tf.name).toBe('');
    expect(tf.description).toBe('');
    expect(tf.type).toBe('project');
    expect(tf.scope).toBeUndefined();
    expect(tf.kind).toBeUndefined();
    expect(tf.body).toBe('just a raw note with no frontmatter at all');
  });

  it('the empty string yields an empty record (never throws)', () => {
    const tf = parseTopicFile('');
    expect(tf).toEqual({ name: '', description: '', type: 'project', scope: undefined, kind: undefined, extra: undefined, body: '' });
  });

  it('partial frontmatter (name only) still parses', () => {
    const tf = parseTopicFile('---\nname: partial\n---\n\nsome body');
    expect(tf.name).toBe('partial');
    expect(tf.description).toBe('');
    expect(tf.type).toBe('project'); // no type → coerced default
    expect(tf.body).toBe('some body');
  });

  it('frontmatter is the FIRST ---block; a horizontal rule in the body stays in the body', () => {
    const text = ['---', 'name: doc', 'metadata:', '  type: reference', '---', '', 'Intro paragraph.', '', '---', '', 'After a rule.'].join('\n');
    const tf = parseTopicFile(text);
    expect(tf.name).toBe('doc');
    expect(tf.type).toBe('reference');
    expect(tf.body.startsWith('Intro paragraph.')).toBe(true);
    expect(tf.body).toContain('---'); // the body's own rule survived, not swallowed as frontmatter
    expect(tf.body).toContain('After a rule.');
  });

  it('leading blank lines between the closing --- and the body are trimmed', () => {
    const tf = parseTopicFile('---\nname: n\nmetadata:\n  type: user\n---\n\n\n\nBody after blanks.');
    expect(tf.body).toBe('Body after blanks.');
  });
});

describe('parseTopicFile — type precedence + coercion', () => {
  it('a top-level type: WINS over a metadata-nested type:', () => {
    const text = '---\ntype: user\nname: n\nmetadata:\n  type: reference\n---\nb';
    expect(parseTopicFile(text).type).toBe('user');
  });

  it('a metadata-nested type: is used when there is no top-level one', () => {
    expect(parseTopicFile('---\nname: n\nmetadata:\n  type: feedback\n---\nb').type).toBe('feedback');
  });

  it('an unrecognised type coerces to project (case-insensitively recognised otherwise)', () => {
    expect(parseTopicFile('---\nname: n\ntype: PROJECT\n---\nb').type).toBe('project');
    expect(parseTopicFile('---\nname: n\ntype: Reference\n---\nb').type).toBe('reference');
    expect(parseTopicFile('---\nname: n\ntype: banana\n---\nb').type).toBe('project');
  });
});

describe('parseTopicFile — scope/kind/extra edge handling', () => {
  it('scope/kind are read ONLY from metadata-nested (indented) lines, never top-level', () => {
    // top-level scope:/kind: lines must NOT be picked up (the projector reads them nested)
    const topLevel = '---\nname: n\nscope: SHOULD_NOT_MATCH\nkind: SHOULD_NOT_MATCH\nmetadata:\n  type: user\n---\nb';
    const tf = parseTopicFile(topLevel);
    expect(tf.scope).toBeUndefined();
    expect(tf.kind).toBeUndefined();
  });

  it('malformed extra JSON is dropped, never fatal', () => {
    const tf = parseTopicFile('---\nname: n\nmetadata:\n  type: user\n  extra: {not valid json\n---\nb');
    expect(tf.extra).toBeUndefined(); // dropped, and no throw
    expect(tf.name).toBe('n'); // the rest of the record parsed fine
  });

  it('a JSON array or scalar for extra is ignored (only a plain object is kept)', () => {
    expect(parseTopicFile('---\nname: n\nmetadata:\n  extra: [1,2,3]\n---\nb').extra).toBeUndefined();
    expect(parseTopicFile('---\nname: n\nmetadata:\n  extra: 42\n---\nb').extra).toBeUndefined();
    expect(parseTopicFile('---\nname: n\nmetadata:\n  extra: {"ok":true}\n---\nb').extra).toEqual({ ok: true });
  });
});

describe('serializeTopicFile — canonical hook-compatible shape', () => {
  const base: TopicFile = { name: 'a_fact', description: 'a short hook', type: 'reference', body: 'The body.' };

  it('emits the frontmatter block with top-level name/description + metadata-nested type', () => {
    const out = serializeTopicFile(base);
    expect(out).toContain('---\nname: a_fact\n');
    expect(out).toContain('description: "a short hook"');
    expect(out).toMatch(/metadata:\n {2}node_type: memory\n {2}type: reference/);
    expect(out.trimEnd().endsWith('The body.')).toBe(true);
  });

  it('omits scope/kind/extra entirely when absent', () => {
    const out = serializeTopicFile(base);
    expect(out).not.toContain('scope:');
    expect(out).not.toContain('kind:');
    expect(out).not.toContain('extra:');
  });

  it('includes scope/kind, and extra only when it is a non-empty object', () => {
    expect(serializeTopicFile({ ...base, scope: 'owner-1', kind: 'reference' })).toContain('  scope: owner-1');
    expect(serializeTopicFile({ ...base, scope: 'owner-1', kind: 'reference' })).toContain('  kind: reference');
    expect(serializeTopicFile({ ...base, extra: {} })).not.toContain('extra:'); // empty object → omitted
    expect(serializeTopicFile({ ...base, extra: { a: 1 } })).toContain('  extra: {"a":1}');
  });

  it('trims the body (no leading/trailing whitespace bleed into the file)', () => {
    const out = serializeTopicFile({ ...base, body: '\n\n  padded body  \n\n' });
    // between the closing frontmatter --- and EOF, the only body text is the trimmed form
    expect(out).toContain('---\n\npadded body\n');
    expect(out).not.toContain('  padded body  ');
  });
});

describe('serialize ∘ parse round-trip — the write→re-read data-integrity invariant', () => {
  it('a full record round-trips identically (body exact — no trailing-newline growth)', () => {
    const tf: TopicFile = {
      name: 'deploy_window',
      description: 'build server deploys 2 to 4am utc only',
      type: 'reference',
      scope: 'owner-99',
      kind: 'reference',
      extra: { orig_file: 'ops.md', nested: { retries: 2 }, tags: ['a', 'b'] },
      body: 'The experimental build server accepts deploys between 2am and 4am UTC.',
    };
    const rt = parseTopicFile(serializeTopicFile(tf));
    expect(rt).toEqual(tf);
  });

  it('special characters in the description survive (quote + backslash escaping is symmetric)', () => {
    const tf: TopicFile = {
      name: 'quoting',
      description: 'she said "go" and the path was a\\b\\c',
      type: 'feedback',
      body: 'note',
    };
    const rt = parseTopicFile(serializeTopicFile(tf));
    expect(rt.description).toBe('she said "go" and the path was a\\b\\c');
    expect(rt.type).toBe('feedback');
  });

  it('a minimal record (no scope/kind/extra) round-trips with those fields still undefined', () => {
    const tf: TopicFile = { name: 'minimal', description: 'tiny', type: 'user', body: 'x' };
    const rt = parseTopicFile(serializeTopicFile(tf));
    expect(rt).toEqual(tf);
  });

  it('the parsed body never carries the file trailing newline(s); internal blank lines survive', () => {
    const body = 'Para one.\n\nPara two.';
    const rt = parseTopicFile(serializeTopicFile({ name: 'n', description: 'd', type: 'project', body }));
    expect(rt.body).toBe(body); // internal \n\n preserved, EOF newline stripped
    // Even a sloppy hand-edited file with extra blank lines at EOF parses clean.
    expect(parseTopicFile('---\nname: n\n---\n\nBody.\n\n\n').body).toBe('Body.');
  });

  it('parse→serialize→parse is stable (idempotent on an already-canonical file)', () => {
    const original = serializeTopicFile({ name: 'stable', description: 'hook text', type: 'project', kind: 'note', body: 'Body content.' });
    const once = parseTopicFile(original);
    const twice = parseTopicFile(serializeTopicFile(once));
    expect(twice).toEqual(once);
  });
});

describe('typeForKind — neutral kind → Claude type taxonomy (default project)', () => {
  it.each([
    ['user', 'user'],
    ['identity', 'user'], // legacy mem0 kind folds into user
    ['preference', 'feedback'],
    ['correction', 'feedback'],
    ['feedback', 'feedback'],
    ['reference', 'reference'],
  ] as const)('%s → %s', (kind, expected) => {
    expect(typeForKind(kind)).toBe(expected);
  });

  it('is case-insensitive', () => {
    expect(typeForKind('USER')).toBe('user');
    expect(typeForKind('Preference')).toBe('feedback');
  });

  it('undefined and any unknown kind default to project (the bloat-safe default)', () => {
    expect(typeForKind(undefined)).toBe('project');
    expect(typeForKind('project')).toBe('project');
    expect(typeForKind('banana')).toBe('project');
    expect(typeForKind('')).toBe('project');
  });

  it('every returned type is a valid Claude memory type', () => {
    for (const k of ['user', 'identity', 'preference', 'correction', 'feedback', 'reference', 'whatever', undefined]) {
      expect(CLAUDE_MEMORY_TYPES).toContain(typeForKind(k));
    }
  });
});

describe('slugify — filename-safe, bounded, never empty', () => {
  it('lowercases and joins words with underscores', () => {
    expect(slugify('Hello World')).toBe('hello_world');
  });

  it('strips punctuation and collapses separators', () => {
    expect(slugify('Foo: Bar! (baz)')).toBe('foo_bar_baz');
    expect(slugify('a---b___c   d')).toBe('a_b_c_d');
  });

  it('bounds to maxWords (default 7, overridable)', () => {
    expect(slugify('one two three four five six seven eight nine')).toBe('one_two_three_four_five_six_seven');
    expect(slugify('one two three four', 2)).toBe('one_two');
  });

  it('keeps digits, underscores and hyphens-as-separators', () => {
    expect(slugify('Test 123 v2')).toBe('test_123_v2');
    expect(slugify('already_snake_case')).toBe('already_snake_case');
  });

  it('falls back to "memory" when nothing usable remains', () => {
    expect(slugify('!!!')).toBe('memory');
    expect(slugify('   ')).toBe('memory');
    expect(slugify('')).toBe('memory');
  });
});

describe('deriveDescription — first real sentence, bounded', () => {
  it('takes the first sentence up to its terminator', () => {
    expect(deriveDescription('This is the first. This is the second.')).toBe('This is the first.');
  });

  it('skips heading (#) and rule (---) lines to the first real line', () => {
    expect(deriveDescription('# A Heading\n---\nThe real first line here.')).toBe('The real first line here.');
  });

  it('returns the whole (short) first line when there is no sentence terminator', () => {
    expect(deriveDescription('no terminator here just words')).toBe('no terminator here just words');
  });

  it('truncates an over-long first line with an ellipsis, bounded by maxChars', () => {
    const long = 'x'.repeat(300);
    const out = deriveDescription(long, 140);
    expect(out.length).toBeLessThanOrEqual(140);
    expect(out.endsWith('…')).toBe(true);
  });

  it('collapses internal whitespace', () => {
    expect(deriveDescription('a\t\t  spaced    out   line')).toBe('a spaced out line');
  });
});

describe('claudeProjectMemoryDir — the project-slug rule', () => {
  it('replaces / and . with - in the project dir and roots under <claudeHome>/projects/<slug>/memory', () => {
    expect(claudeProjectMemoryDir('/home/u/proj.dir', '/home/u/.claude')).toBe('/home/u/.claude/projects/-home-u-proj-dir/memory');
  });

  it('strips trailing slashes from claudeHome (but never slugifies claudeHome itself)', () => {
    expect(claudeProjectMemoryDir('/p', '/home/u/.claude///')).toBe('/home/u/.claude/projects/-p/memory');
  });
});
