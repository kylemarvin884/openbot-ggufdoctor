import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildGguf,
  baseMetadata,
  CHATML_TOOLS_TEMPLATE,
  CHATML_NO_TOOLS_TEMPLATE,
} from '../demo/gguf-writer.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const cli = path.join(here, '..', 'src', 'cli.mjs');
const dir = mkdtempSync(path.join(tmpdir(), 'ggufdoctor-'));

function write(name, metadata) {
  const file = path.join(dir, name);
  writeFileSync(file, buildGguf({ metadata }));
  return file;
}

function run(args) {
  const res = spawnSync(process.execPath, [cli, ...args], { encoding: 'utf8' });
  return { code: res.status, stdout: res.stdout, stderr: res.stderr };
}

const good = write('good.gguf', baseMetadata());
const noTools = write('no-tools.gguf', baseMetadata({ template: CHATML_NO_TOOLS_TEMPLATE }));
const warnOnly = write('warn.gguf', baseMetadata({
  template: CHATML_TOOLS_TEMPLATE.replaceAll('<|im_end|>', '<|end_of_turn|>'),
}));
const notGguf = path.join(dir, 'notes.txt');
writeFileSync(notGguf, 'this is not a model');

test('passing file exits 0 and prints a verdict', () => {
  const { code, stdout } = run(['check', good, '--ctx', '8192']);
  assert.equal(code, 0);
  assert.match(stdout, /verdict PASS/);
  assert.match(stdout, /arch=qwen3/);
});

test('failing file exits 1', () => {
  const { code, stdout } = run(['check', noTools]);
  assert.equal(code, 1);
  assert.match(stdout, /verdict FAIL/);
  assert.match(stdout, /tools-unsupported/);
});

test('--json emits parseable output with a stable shape', () => {
  const { code, stdout } = run(['check', noTools, '--json']);
  assert.equal(code, 1);
  const payload = JSON.parse(stdout);
  assert.equal(payload.tool, 'ggufdoctor');
  assert.equal(payload.verdict, 'fail');
  assert.ok(Array.isArray(payload.findings));
  assert.ok(payload.findings.some((f) => f.id === 'tools-unsupported'));
  assert.equal(payload.model.arch, 'qwen3');
});

test('--strict turns warnings into exit 1', () => {
  assert.equal(run(['check', warnOnly]).code, 0);
  assert.equal(run(['check', warnOnly, '--strict']).code, 1);
});

test('a file that is not GGUF exits 2, not 1', () => {
  const { code, stderr } = run(['check', notGguf]);
  assert.equal(code, 2);
  assert.match(stderr, /not a GGUF file/);
});

test('missing arguments exit 2 with usage', () => {
  const { code, stdout } = run([]);
  assert.equal(code, 2);
  assert.match(stdout, /Usage:/);
});

test('--help exits 0', () => {
  const { code, stdout } = run(['--help']);
  assert.equal(code, 0);
  assert.match(stdout, /--vram/);
});

test('--report writes a markdown report', () => {
  const reportPath = path.join(dir, 'report.md');
  const { code } = run(['check', noTools, '--report', reportPath]);
  assert.equal(code, 1);
  const md = readFileSync(reportPath, 'utf8');
  assert.match(md, /^# ggufdoctor report/m);
  assert.match(md, /## Verdict: FAIL/);
  assert.match(md, /tools-unsupported/);
});

test('unknown options exit 2 and name the option', () => {
  const { code, stderr } = run(['check', good, '--wat']);
  assert.equal(code, 2);
  assert.match(stderr, /unknown option --wat/);
});

test('--vram makes the capacity check concrete', () => {
  const { code, stdout } = run(['check', good, '--ctx', '32768', '--vram', '2', '--json']);
  const payload = JSON.parse(stdout);
  assert.equal(code, 1);
  assert.ok(payload.findings.some((f) => f.id === 'kv-over-budget'));
  assert.ok(payload.capacity.maxCtx > 0);
  assert.match(stdout, /maxCtx/);
});
