const GIB = 1024 ** 3;

const SYMBOL = { error: '✗', warn: '!', info: '·' };

export function formatSize(bytes) {
  if (typeof bytes !== 'number') return '?';
  if (bytes >= GIB) return `${(bytes / GIB).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

function verdictLine(analysis) {
  const { counts, verdict } = analysis;
  const label = verdict === 'fail' ? 'FAIL' : verdict === 'warn' ? 'PASS WITH WARNINGS' : 'PASS';
  const parts = [];
  if (counts.error) parts.push(`${counts.error} error${counts.error === 1 ? '' : 's'}`);
  if (counts.warn) parts.push(`${counts.warn} warning${counts.warn === 1 ? '' : 's'}`);
  return `${label} (${parts.length ? parts.join(', ') : 'no findings'})`;
}

export function renderText(analysis, { file, version }) {
  const { model, capacity, template, findings } = analysis;
  const lines = [];
  lines.push(`ggufdoctor ${version}`);
  lines.push(
    `model   ${model.name}   ${formatSize(model.fileSizeBytes)}   arch=${model.arch ?? '?'}`
    + `${capacity ? `  layers=${capacity.layers}` : ''}`
    + `  ctx=${model.trainedCtx ?? '?'}`
    + `  vocab=${model.vocabSize ?? '?'}`,
  );
  lines.push(
    `file    GGUF v${model.version}, ${plural(model.tensorCount, 'tensor')}, quant ${model.quant}`
    + `${template ? `, template ${template.family}${template.count > 1 ? ` x${template.count}` : ''}` : ', no template'}`,
  );
  lines.push('');
  lines.push(`verdict ${verdictLine(analysis)}`);
  lines.push('');

  const order = { error: 0, warn: 1, info: 2 };
  const sorted = [...findings].sort((a, b) => order[a.severity] - order[b.severity]);
  for (const f of sorted) {
    lines.push(`  ${SYMBOL[f.severity]} ${f.id}  ${f.title}`);
    lines.push(`      ${f.detail}`);
    if (f.fix) lines.push(`      fix: ${f.fix}`);
  }
  lines.push('');
  lines.push(`file: ${file}`);
  return lines.join('\n');
}

export function renderMarkdown(analysis, { file, version }) {
  const { model, capacity, template, findings } = analysis;
  const lines = [];
  lines.push(`# ggufdoctor report`);
  lines.push('');
  lines.push(`- file: \`${file}\``);
  lines.push(`- tool: ggufdoctor ${version}`);
  lines.push(`- model name: ${model.name}`);
  lines.push(`- architecture: ${model.arch ?? 'unknown'}`);
  lines.push(`- size: ${formatSize(model.fileSizeBytes)}`);
  lines.push(`- quant: ${model.quant}`);
  lines.push(`- trained ctx: ${model.trainedCtx ?? 'unknown'}`);
  lines.push(`- vocab: ${model.vocabSize ?? 'unknown'}`);
  if (template) {
    lines.push(`- template: ${template.family} (${template.length} chars, ${template.count} template(s))`);
    lines.push(`- tools branch: ${template.hasToolsBranch ? 'yes' : 'no'}`);
    lines.push(`- tool result branch: ${template.handlesToolRole ? 'yes' : 'no'}`);
  }
  if (capacity) {
    lines.push(`- KV cache per token: ${capacity.bytesPerToken} bytes (f16)`);
    if (capacity.kvBytesAtCtx) lines.push(`- KV cache at requested ctx: ${formatSize(capacity.kvBytesAtCtx)}`);
    if (capacity.maxCtx !== null) lines.push(`- max ctx in budget: ${capacity.maxCtx}`);
  }
  lines.push('');
  lines.push(`## Verdict: ${verdictLine(analysis)}`);
  lines.push('');
  for (const f of findings) {
    lines.push(`### ${SYMBOL[f.severity]} ${f.title} (\`${f.id}\`)`);
    lines.push('');
    lines.push(f.detail);
    if (f.fix) {
      lines.push('');
      lines.push(`**Fix:** ${f.fix}`);
    }
    lines.push('');
  }
  return lines.join('\n');
}

export function renderJson(analysis, { file, version }) {
  return JSON.stringify(
    {
      tool: 'ggufdoctor',
      version,
      file,
      model: analysis.model,
      template: analysis.template,
      capacity: analysis.capacity,
      verdict: analysis.verdict,
      counts: analysis.counts,
      findings: analysis.findings,
    },
    null,
    2,
  );
}

export function exitCodeFor(analysis, { strict } = {}) {
  if (analysis.counts.error > 0) return 1;
  if (strict && analysis.counts.warn > 0) return 1;
  return 0;
}
