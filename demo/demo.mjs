// Demo: builds three metadata-only GGUF fixtures and diagnoses them.
// The fixtures are spec-shaped but contain no weights, so the whole demo runs
// offline in under a second. Point the CLI at a real .gguf for a real report.

import { mkdirSync, writeFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildGguf,
  baseMetadata,
  CHATML_TOOLS_TEMPLATE,
  CHATML_NO_TOOLS_TEMPLATE,
  BASE_VOCAB,
} from './gguf-writer.mjs';
import { parseGguf } from '../src/gguf.mjs';
import { analyze } from '../src/analyze.mjs';
import { renderText } from '../src/report.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, '.run');
mkdirSync(outDir, { recursive: true });

const fixtures = [
  {
    file: 'tool-ready.gguf',
    note: 'tool-aware template, sane ids',
    buf: buildGguf({
      metadata: baseMetadata(),
      tensors: [{ name: 'token_embd.weight', dims: [4096, 7], type: 12 }],
    }),
    opts: { ctx: 8192, vramGb: 12 },
  },
  {
    file: 'no-tools.gguf',
    note: 'template without a tools branch, eos id outside the vocab',
    buf: buildGguf({
      metadata: baseMetadata({ template: CHATML_NO_TOOLS_TEMPLATE, eos: 99 }),
      tensors: [{ name: 'token_embd.weight', dims: [4096, 7], type: 12 }],
    }),
    opts: { ctx: 32768, vramGb: 2 },
  },
  {
    file: 'token-mismatch.gguf',
    note: 'template prints a control token that is not in the vocab',
    buf: buildGguf({
      metadata: baseMetadata({
        template: CHATML_TOOLS_TEMPLATE.replaceAll('<|im_end|>', '<|end_of_turn|>'),
        vocab: BASE_VOCAB,
      }),
      tensors: [{ name: 'token_embd.weight', dims: [4096, 7], type: 12 }],
    }),
    opts: { ctx: 4096 },
  },
];

const results = [];
for (const fixture of fixtures) {
  const filePath = path.join(outDir, fixture.file);
  writeFileSync(filePath, fixture.buf);
  const model = parseGguf(fixture.buf);
  const analysis = analyze(model, {
    ...fixture.opts,
    fileSizeBytes: statSync(filePath).size,
  });
  results.push({ ...fixture, filePath, analysis });
}

console.log('ggufdoctor demo — three metadata-only fixtures in demo/.run/\n');
for (const r of results) {
  const { counts, verdict } = r.analysis;
  const label = verdict === 'fail' ? 'FAIL' : verdict === 'warn' ? 'WARN' : 'PASS';
  console.log(`  ${label.padEnd(4)}  ${r.file.padEnd(20)} ${counts.error} errors, ${counts.warn} warnings  — ${r.note}`);
}

const showcase = results[1];
console.log('\n' + '-'.repeat(72) + '\n');
console.log(renderText(showcase.analysis, { file: showcase.filePath, version: '0.1.0' }));
console.log('\nRun it on a real file:  node src/cli.mjs check <model.gguf> --ctx 8192 --vram 12');
