/**
 * Tests for the Claude topic-file bridge (claude-memory-projection-
 * integration-2026-06-05 P-004 / generalize-memory-backend-swappable
 * D-005): format round-trip + compatibility with the user-env index
 * projector's parsing rules, the never-touch-MEMORY.md contract,
 * archive-not-delete forget, scope/kind semantics, search/list, and the
 * registry/selector flip.
 *
 * All tests run on temp dirs — NEVER the live ~/.claude memory.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  parseTopicFile,
  serializeTopicFile,
  typeForKind,
  slugify,
  deriveDescription,
  claudeProjectMemoryDir,
} from './topic-file';
import {
  ClaudeFileMemoryBackend,
  CLAUDE_FILE_BACKEND_NAME,
  MEMORY_DIR_MISSING_REASON,
} from './claude-file-backend';
import { MemoryUnavailableError } from './backend';
import {
  getMemoryBackend,
  registerMemoryBackend,
  _resetMemoryBackendsForTest,
} from './backend-registry';
import { configureMemory } from './config';

let dir: string;
const dirs: string[] = [];

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'claude-mem-'));
  dirs.push(dir);
});

afterEach(() => {
  for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
});

function backend(opts: Partial<ConstructorParameters<typeof ClaudeFileMemoryBackend>[0]> = {}) {
  return new ClaudeFileMemoryBackend({ memoryDir: dir, ...opts });
}

/** A hand-curated legacy file in the exact shape of the owner's real store. */
function writeLegacy(name: string, type: string, description: string, body: string) {
  fs.writeFileSync(
    path.join(dir, `${name}.md`),
    `---\nname: ${name}\ndescription: "${description}"\nmetadata: \n  node_type: memory\n  type: ${type}\n---\n\n${body}\n`,
  );
}

// ---- topic-file format ------------------------------------------------------

describe('topic-file format', () => {
  it('round-trips serialize → parse (incl. scope/kind/extra)', () => {
    const tf = {
      name: 'feedback_x',
      description: 'a "quoted" hook — with dash',
      type: 'feedback' as const,
      scope: 'harness:papercup',
      kind: 'correction',
      extra: { source: 'test', n: 3 },
      body: 'The full fact body.\n\nWith two paragraphs.',
    };
    const parsed = parseTopicFile(serializeTopicFile(tf));
    expect(parsed.name).toBe('feedback_x');
    expect(parsed.description).toBe('a "quoted" hook — with dash');
    expect(parsed.type).toBe('feedback');
    expect(parsed.scope).toBe('harness:papercup');
    expect(parsed.kind).toBe('correction');
    expect(parsed.extra).toEqual({ source: 'test', n: 3 });
    expect(parsed.body.trim()).toBe(tf.body);
  });

  it('parses the hand-curated legacy shape (metadata-nested type, quoted description)', () => {
    const parsed = parseTopicFile(
      '---\nname: project_x\ndescription: "The LIVE /adv UI is operator-vite"\nmetadata: \n  node_type: memory\n  type: reference\n  originSessionId: abc\n---\n\nBody here.\n',
    );
    expect(parsed.name).toBe('project_x');
    expect(parsed.type).toBe('reference');
    expect(parsed.description).toBe('The LIVE /adv UI is operator-vite');
    expect(parsed.body.trim()).toBe('Body here.');
  });

  it('serialized output satisfies the index projector frontmatter regexes', () => {
    // The EXACT patterns memory-compact.mjs parses with — a file we write
    // must project into the regenerated MEMORY.md unchanged.
    const text = serializeTopicFile({
      name: 'reference_y',
      description: 'multi\nline gets flattened',
      type: 'reference',
      body: 'b',
    });
    const fm = /^---\n([\s\S]*?)\n---/.exec(text)?.[1] ?? '';
    expect(/^name:[ \t]*(.+)$/m.exec(fm)?.[1]).toBe('reference_y');
    const desc = /^description:[ \t]*(.+)$/m.exec(fm)?.[1] ?? '';
    expect(desc.startsWith('"') && desc.endsWith('"')).toBe(true);
    expect(desc).not.toContain('\n');
    expect(/^[ \t]+type:[ \t]*(.+)$/m.exec(fm)?.[1]).toBe('reference');
  });

  it('typeForKind maps durable kinds to durable types and unknown to project', () => {
    expect(typeForKind('identity')).toBe('user');
    expect(typeForKind('preference')).toBe('feedback');
    expect(typeForKind('correction')).toBe('feedback');
    expect(typeForKind('reference')).toBe('reference');
    expect(typeForKind('project')).toBe('project');
    expect(typeForKind(undefined)).toBe('project');
    expect(typeForKind('whatever')).toBe('project');
  });

  it('slugify + deriveDescription produce filename-safe, bounded hooks', () => {
    expect(slugify('The QUICK brown fox: jumps! over (everything) and more words here')).toBe(
      'the_quick_brown_fox_jumps_over_everything',
    );
    const d = deriveDescription('x'.repeat(400));
    expect(d.length).toBeLessThanOrEqual(140);
    expect(deriveDescription('# heading\n\nFirst real sentence. Second one.')).toBe('First real sentence.');
  });

  it('claudeProjectMemoryDir applies the / and . → - slug rule', () => {
    expect(claudeProjectMemoryDir('/home/marsh-office', '/home/u/.claude')).toBe(
      '/home/u/.claude/projects/-home-marsh-office/memory',
    );
  });
});

// ---- backend: write side ----------------------------------------------------

describe('ClaudeFileMemoryBackend writes', () => {
  it('remember lands a valid, hook-parsable topic file and returns its id', async () => {
    const b = backend();
    const { ids } = await b.remember('WebKitGTK caps per-host connections at 6 — pool exhaustion looks like CORS.', {
      scope: 'user-1',
      kind: 'correction',
      metadata: { source: 'session-x' },
    });
    expect(ids).toHaveLength(1);
    const file = path.join(dir, `${ids[0]}.md`);
    expect(fs.existsSync(file)).toBe(true);
    expect(ids[0].startsWith('feedback_')).toBe(true); // correction → feedback type prefix

    const parsed = parseTopicFile(fs.readFileSync(file, 'utf8'));
    expect(parsed.type).toBe('feedback');
    expect(parsed.scope).toBe('user-1');
    expect(parsed.kind).toBe('correction');
    expect(parsed.extra).toEqual({ source: 'session-x' });
    expect(parsed.description).toMatch(/WebKitGTK caps per-host/);

    const got = await b.get(ids[0]);
    expect(got?.text).toMatch(/pool exhaustion looks like CORS/);
    expect(got?.kind).toBe('correction');
    expect(got?.scope).toBe('user-1');
  });

  it('uniquifies colliding ids instead of overwriting', async () => {
    const b = backend();
    const a = await b.remember('Same hook text.', { scope: 's' });
    const c = await b.remember('Same hook text.', { scope: 's' });
    expect(a.ids[0]).not.toBe(c.ids[0]);
    expect((await b.get(a.ids[0]))).not.toBeNull();
    expect((await b.get(c.ids[0]))).not.toBeNull();
  });

  it('NEVER touches MEMORY.md (the auto-generated index)', async () => {
    const sentinel = '# Memory Index\n\nsentinel-do-not-touch\n';
    fs.writeFileSync(path.join(dir, 'MEMORY.md'), sentinel);
    const b = backend();
    const { ids } = await b.remember('a fact', { scope: 's' });
    await b.update(ids[0], { text: 'changed' });
    await b.forget(ids[0]);
    expect(fs.readFileSync(path.join(dir, 'MEMORY.md'), 'utf8')).toBe(sentinel);
    const listed = await b.list({ scope: 's' });
    expect(listed.find((e) => e.id === 'MEMORY')).toBeUndefined();
  });

  it('throws MemoryUnavailableError on a missing dir unless createIfMissing', async () => {
    const missing = path.join(dir, 'nope');
    const b = new ClaudeFileMemoryBackend({ memoryDir: missing });
    await expect(b.available()).resolves.toEqual({ ok: false, reason: MEMORY_DIR_MISSING_REASON });
    await expect(b.remember('x', { scope: 's' })).rejects.toBeInstanceOf(MemoryUnavailableError);

    const creating = new ClaudeFileMemoryBackend({ memoryDir: missing, createIfMissing: true });
    await expect(creating.available()).resolves.toEqual({ ok: true });
    await expect(creating.remember('x', { scope: 's' })).resolves.toBeTruthy();
  });

  it('forget archives (recoverable), never deletes; idempotent on a gone id', async () => {
    const b = backend();
    const { ids } = await b.remember('to be archived', { scope: 's' });
    await b.forget(ids[0]);
    expect(fs.existsSync(path.join(dir, `${ids[0]}.md`))).toBe(false);
    expect(fs.existsSync(path.join(dir, 'archive', `${ids[0]}.md`))).toBe(true);
    expect(await b.get(ids[0])).toBeNull();
    await expect(b.forget(ids[0])).resolves.toBeUndefined(); // already gone → resolves
  });

  it('update patches text and merges metadata (kind/scope/description recognized)', async () => {
    const b = backend();
    const { ids } = await b.remember('original body', { scope: 's', kind: 'reference' });
    await b.update(ids[0], {
      text: 'replacement body',
      metadata: { description: 'new hook', kind: 'correction', custom: true },
    });
    const got = await b.get(ids[0]);
    expect(got?.text).toBe('replacement body');
    expect(got?.kind).toBe('correction');
    expect(got?.metadata?.description).toBe('new hook');
    expect(got?.metadata?.custom).toBe(true);
    await expect(b.update('missing_id', { text: 'x' })).rejects.toThrow(/not found/);
  });

  it('rejects path-traversal ids', async () => {
    const b = backend();
    expect(await b.get('../../../etc/passwd')).toBeNull();
    await expect(b.forget('../escape')).resolves.toBeUndefined();
    await expect(b.update('a/b', { text: 'x' })).rejects.toThrow();
  });
});

// ---- backend: read side -----------------------------------------------------

describe('ClaudeFileMemoryBackend reads', () => {
  it('surfaces hand-curated legacy files (no scope) under any queried scope by default', async () => {
    writeLegacy('feedback_legacy', 'feedback', 'never git stash in the shared tree', 'Full story.');
    const b = backend();
    const inUser = await b.list({ scope: 'user-1' });
    const inHarness = await b.list({ scope: 'harness:papercup' });
    expect(inUser.map((e) => e.id)).toContain('feedback_legacy');
    expect(inHarness.map((e) => e.id)).toContain('feedback_legacy');
    // …but a fan-out query returns it ONCE
    const fanout = await b.list({ scope: ['user-1', 'harness:papercup'] });
    expect(fanout.filter((e) => e.id === 'feedback_legacy')).toHaveLength(1);
  });

  it('confines unscoped files to unscopedScope when matchUnscopedToAnyScope=false', async () => {
    writeLegacy('reference_r', 'reference', 'a pointer', 'body');
    const b = backend({ unscopedScope: 'local', matchUnscopedToAnyScope: false });
    expect((await b.list({ scope: 'user-1' })).map((e) => e.id)).not.toContain('reference_r');
    expect((await b.list({ scope: 'local' })).map((e) => e.id)).toContain('reference_r');
  });

  it('scoped entries only match their own scope', async () => {
    const b = backend();
    await b.remember('harness fact', { scope: 'harness:a' });
    const other = await b.list({ scope: 'harness:b' });
    expect(other).toHaveLength(0);
    const own = await b.list({ scope: 'harness:a' });
    expect(own).toHaveLength(1);
  });

  it('search ranks name > description > body hits and respects per-scope limit', async () => {
    writeLegacy('feedback_webkit_pool', 'feedback', 'WebKitGTK connection pool exhaustion gotcha', 'libsoup max-conns-per-host is 6.');
    writeLegacy('reference_other', 'reference', 'unrelated pointer', 'mentions webkit once in the body.');
    writeLegacy('reference_noise', 'reference', 'totally unrelated', 'nothing relevant here.');
    const b = backend();

    const hits = await b.search('webkit pool', { scope: 'user-1' });
    expect(hits.length).toBe(2);
    expect(hits[0].id).toBe('feedback_webkit_pool'); // name+desc hits outrank body hit
    expect((hits[0].score ?? 0) > (hits[1].score ?? 0)).toBe(true);

    const limited = await b.search('webkit', { scope: 'user-1', limit: 1 });
    expect(limited).toHaveLength(1);
  });

  it('list filters by kind (falling back to the Claude type for legacy files)', async () => {
    writeLegacy('feedback_f', 'feedback', 'f', 'b');
    writeLegacy('reference_r', 'reference', 'r', 'b');
    const b = backend();
    const onlyRef = await b.list({ scope: 'x', kind: 'reference' });
    expect(onlyRef.map((e) => e.id)).toEqual(['reference_r']);
  });

  it('ignores archive/ and skills/ subdirectories', async () => {
    fs.mkdirSync(path.join(dir, 'archive'));
    fs.mkdirSync(path.join(dir, 'skills'));
    fs.writeFileSync(path.join(dir, 'archive', 'project_old.md'), '---\nname: old\ndescription: archived\nmetadata:\n  type: project\n---\nold');
    writeLegacy('user_me', 'user', 'who I am', 'body');
    const b = backend();
    const all = await b.list({ scope: 's' });
    expect(all.map((e) => e.id)).toEqual(['user_me']);
  });
});

// ---- registry/selector integration -------------------------------------------

describe('selector flip', () => {
  beforeEach(() => _resetMemoryBackendsForTest());
  afterEach(() => _resetMemoryBackendsForTest());

  it("registerMemoryBackend('claude-file') makes it selectable via configureMemory", async () => {
    registerMemoryBackend(CLAUDE_FILE_BACKEND_NAME, () => new ClaudeFileMemoryBackend({ memoryDir: dir }));
    configureMemory({
      getAdminUrl: () => 'postgres://stub',
      getCredentials: async () => ({}),
      resolveEmbedder: async () => ({ mode: 'disabled' as const }),
      buildEmbedderForMode: async () => async () => [],
      backend: CLAUDE_FILE_BACKEND_NAME,
    });
    const b = getMemoryBackend();
    expect(b.name).toBe(CLAUDE_FILE_BACKEND_NAME);
    await expect(b.available()).resolves.toEqual({ ok: true });
    const { ids } = await b.remember('selector-flip fact', { scope: 'user-1' });
    expect((await b.get(ids[0]))?.text).toBe('selector-flip fact');
  });
});
