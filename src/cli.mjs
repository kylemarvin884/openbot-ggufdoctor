#!/usr/bin/env node
// ggufdoctor - check whether a GGUF file can drive an agent.
// Zero dependencies, offline, reads metadata only.

import { openSync, readSync, closeSync, statSync, writeFileSync } from 'node:fs';
import { parseGguf, GgufError } from './gguf.mjs';
import { analyze } from './analyze.mjs';
import { renderText, renderMarkdown, renderJson, exitCodeFor } from './report.mjs';

const VERSION = '0.1.0';
const DEFAULT_MAX_HEADER_MB = 256;

const USAGE = `ggufdoctor ${VERSION}

Usage:
  ggufdoctor check <model.gguf> [options]

Options:
  --ctx <n>          context length you intend to serve with (checks trained limit + KV size)
  --vram <GiB>       VRAM budget in GiB (checks weights + f16 KV cache against it)
  --json             machine-readable output
  --report <file>    also write a Markdown report
  --strict           treat warnings as failures
  --max-header-mb <n>  how much of the file to read for metadata (default ${DEFAULT_MAX_HEADER_MB})
  -h, --help         show this help

Exit codes:
  0  pass (or pass with warnings)
  1  fail: at least one error finding (or any warning with --strict)
  2  could not read the file as GGUF

What it checks:
  - chat template exists and has a tools branch (tool_calls stays empty otherwise)
  - template handles role == "tool" and add_generation_prompt
  - control tokens the template prints exist in the vocabulary
  - eos/bos ids are inside the vocabulary
  - trained context length vs the context you intend to serve
  - KV cache size against a VRAM budget, and the max ctx that fits
`;

function parseArgs(argv) {
  const opts = {
    command: null,
    file: null,
    ctx: undefined,
    vram: undefined,
    json: false,
    report: null,
    strict: false,
    maxHeaderMb: DEFAULT_MAX_HEADER_MB,
    help: false,
  };
  const rest = argv.slice();
  while (rest.length > 0) {
    const arg = rest.shift();
    if (arg === '-h' || arg === '--help') opts.help = true;
    else if (arg === '--json') opts.json = true;
    else if (arg === '--strict') opts.strict = true;
    else if (arg === '--ctx') {
      const v = Number(rest.shift());
      if (!Number.isFinite(v) || v <= 0) throw new Error('--ctx needs a positive number');
      opts.ctx = Math.floor(v);
    } else if (arg === '--vram') {
      const v = Number(rest.shift());
      if (!Number.isFinite(v) || v <= 0) throw new Error('--vram needs a positive number (GiB)');
      opts.vram = v;
    } else if (arg === '--report') {
      const v = rest.shift();
      if (!v) throw new Error('--report needs a file path');
      opts.report = v;
    } else if (arg === '--max-header-mb') {
      const v = Number(rest.shift());
      if (!Number.isFinite(v) || v <= 0) throw new Error('--max-header-mb needs a positive number');
      opts.maxHeaderMb = v;
    } else if (arg.startsWith('-')) {
      throw new Error(`unknown option ${arg}`);
    } else if (opts.command === null) {
      opts.command = arg;
    } else if (opts.file === null) {
      opts.file = arg;
    } else {
      throw new Error(`unexpected argument ${arg}`);
    }
  }
  return opts;
}

function readHeader(path, maxHeaderMb) {
  const size = statSync(path).size;
  const cap = Math.max(1, Math.min(size, maxHeaderMb * 1024 * 1024));
  const fd = openSync(path, 'r');
  try {
    const buf = Buffer.alloc(cap);
    const read = readSync(fd, buf, 0, cap, 0);
    return {
      buf: read === cap ? buf : buf.subarray(0, read),
      size,
      truncated: size > cap,
    };
  } finally {
    closeSync(fd);
  }
}

function main(argv) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (err) {
    process.stderr.write(`ggufdoctor: ${err.message}\n\n${USAGE}`);
    return 2;
  }

  if (opts.help || opts.command !== 'check' || !opts.file) {
    process.stdout.write(USAGE);
    return opts.help ? 0 : 2;
  }

  let header;
  try {
    header = readHeader(opts.file, opts.maxHeaderMb);
  } catch (err) {
    process.stderr.write(`ggufdoctor: cannot read ${opts.file}: ${err.message}\n`);
    return 2;
  }

  let model;
  try {
    model = parseGguf(header.buf);
  } catch (err) {
    if (err instanceof GgufError) {
      const hint = header.truncated
        ? ` (metadata may sit past the ${opts.maxHeaderMb} MiB read limit; retry with --max-header-mb 1024)`
        : '';
      process.stderr.write(`ggufdoctor: ${opts.file}: ${err.message}${hint}\n`);
      return 2;
    }
    throw err;
  }

  const analysis = analyze(model, {
    ctx: opts.ctx,
    vramGb: opts.vram,
    fileSizeBytes: header.size,
  });

  if (opts.json) {
    process.stdout.write(`${renderJson(analysis, { file: opts.file, version: VERSION })}\n`);
  } else {
    process.stdout.write(`${renderText(analysis, { file: opts.file, version: VERSION })}\n`);
  }

  if (opts.report) {
    writeFileSync(opts.report, renderMarkdown(analysis, { file: opts.file, version: VERSION }), 'utf8');
  }

  return exitCodeFor(analysis, { strict: opts.strict });
}

process.exitCode = main(process.argv.slice(2));
