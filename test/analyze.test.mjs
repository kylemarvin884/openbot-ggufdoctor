import test from 'node:test';
import assert from 'node:assert/strict';
import { parseGguf } from '../src/gguf.mjs';
import { analyze, inspectTemplate, estimateCapacity, extractTemplates } from '../src/analyze.mjs';
import {
  buildGguf,
  baseMetadata,
  CHATML_TOOLS_TEMPLATE,
  CHATML_NO_TOOLS_TEMPLATE,
  BASE_VOCAB,
} from '../demo/gguf-writer.mjs';

const GIB = 1024 ** 3;

function run(metadata, opts = {}) {
  const model = parseGguf(buildGguf({ metadata, tensors: [{ name: 'w', dims: [8] }] }));
  return analyze(model, opts);
}

function ids(analysis) {
  return analysis.findings.map((f) => f.id);
}

test('a tool-aware chatml template passes', () => {
  const analysis = run(baseMetadata());
  assert.equal(analysis.counts.error, 0, JSON.stringify(analysis.findings, null, 2));
  assert.equal(analysis.verdict, 'pass');
  assert.equal(analysis.template.family, 'chatml');
  assert.equal(analysis.template.hasToolsBranch, true);
  assert.equal(analysis.template.handlesToolRole, true);
});

test('template without a tools branch fails, and says why', () => {
  const analysis = run(baseMetadata({ template: CHATML_NO_TOOLS_TEMPLATE }));
  const finding = analysis.findings.find((f) => f.id === 'tools-unsupported');
  assert.ok(finding);
  assert.equal(finding.severity, 'error');
  assert.match(finding.detail, /tools/);
  assert.equal(analysis.verdict, 'fail');
});

test('template that defines tools but ignores tool results is only a warning', () => {
  const partial = CHATML_TOOLS_TEMPLATE.replace(/message\.role == 'tool'/, "message.role == 'user'");
  const analysis = run(baseMetadata({ template: partial }));
  const finding = analysis.findings.find((f) => f.id === 'tool-role-unhandled');
  assert.ok(finding);
  assert.equal(finding.severity, 'warn');
  assert.equal(analysis.counts.error, 0);
});

test('missing chat template is an error', () => {
  const metadata = baseMetadata();
  delete metadata['tokenizer.chat_template'];
  const analysis = run(metadata);
  assert.ok(ids(analysis).includes('template-missing'));
  assert.equal(analysis.verdict, 'fail');
});

test('eos id outside the vocabulary is an error', () => {
  const analysis = run(baseMetadata({ eos: 99 }));
  const finding = analysis.findings.find((f) => f.id === 'eos-out-of-range');
  assert.ok(finding);
  assert.equal(finding.severity, 'error');
  assert.match(finding.detail, /0\.\.6/);
});

test('control token printed by the template but absent from the vocab warns', () => {
  const analysis = run(baseMetadata({
    template: CHATML_TOOLS_TEMPLATE.replaceAll('<|im_end|>', '<|end_of_turn|>'),
  }));
  const finding = analysis.findings.find((f) => f.id === 'template-token-missing');
  assert.ok(finding);
  assert.match(finding.detail, /<\|end_of_turn\|>/);
  assert.equal(analysis.verdict, 'warn');
});

test('requesting more context than the model was trained for is an error', () => {
  const analysis = run(baseMetadata({ ctx: 8192 }), { ctx: 32768 });
  const finding = analysis.findings.find((f) => f.id === 'ctx-exceeds-trained');
  assert.ok(finding);
  assert.equal(finding.severity, 'error');
});

test('KV cache estimate matches the f16 formula', () => {
  // 32 layers, 8 KV heads, key/value dim 128 each: 32*8*(128+128)*2 = 131072 bytes/token
  const model = parseGguf(buildGguf({ metadata: baseMetadata() }));
  const capacity = estimateCapacity(model.metadata, { ctx: 4096, fileSizeBytes: 0 });
  assert.equal(capacity.kvHeadsTotal, 256);
  assert.equal(capacity.keyLen, 128);
  assert.equal(capacity.bytesPerToken, 131072);
  assert.equal(capacity.kvBytesAtCtx, 131072 * 4096);
  assert.equal(capacity.kvBytesAtCtx, 512 * 1024 * 1024);
});

test('KV cache over the VRAM budget is an error with a usable max ctx', () => {
  const analysis = run(baseMetadata(), { ctx: 32768, vramGb: 2, fileSizeBytes: 0 });
  const finding = analysis.findings.find((f) => f.id === 'kv-over-budget');
  assert.ok(finding);
  assert.equal(finding.severity, 'error');
  assert.match(finding.fix, /Max context that fits/);
  assert.ok(analysis.capacity.maxCtx > 0);
  assert.ok(analysis.capacity.maxCtx < 32768);
});

test('KV cache fits the budget when there is room, and reports headroom', () => {
  const analysis = run(baseMetadata(), { ctx: 4096, vramGb: 12, fileSizeBytes: 4 * GIB });
  const finding = analysis.findings.find((f) => f.id === 'kv-fits');
  assert.ok(finding);
  assert.equal(finding.severity, 'info');
  assert.equal(analysis.counts.error, 0);
});

test('per-layer KV head arrays are summed, not averaged', () => {
  const metadata = baseMetadata();
  metadata['qwen3.attention.head_count_kv'] = { type: 'array', elem: 'uint32', value: [8, 8, 2, 2] };
  metadata['qwen3.block_count'] = { type: 'uint32', value: 4 };
  const model = parseGguf(buildGguf({ metadata }));
  const capacity = estimateCapacity(model.metadata, { ctx: 1024, fileSizeBytes: 0 });
  assert.equal(capacity.kvHeadsTotal, 20);
  assert.equal(capacity.bytesPerToken, 20 * 256 * 2);
});

test('missing architecture is an error and does not crash capacity math', () => {
  const metadata = baseMetadata();
  delete metadata['general.architecture'];
  const analysis = run(metadata, { ctx: 4096, vramGb: 8, fileSizeBytes: 1024 });
  assert.ok(ids(analysis).includes('arch-missing'));
  assert.equal(analysis.capacity, null);
});

test('inspectTemplate recognises llama3 and llama2 families', () => {
  const llama3 = inspectTemplate('<|start_header_id|>{% if tools %}{{ tools }}{% endif %}{% if add_generation_prompt %}<|end_of_text|>{% endif %}');
  assert.equal(llama3.family, 'llama3');
  assert.equal(llama3.hasToolsBranch, true);
  assert.equal(inspectTemplate('[INST] {{ messages }} [/INST]').family, 'llama2');
  assert.equal(inspectTemplate('plain text').family, 'unknown');
});

test('extractTemplates handles a string, and an array of named templates', () => {
  const single = new Map([['tokenizer.chat_template', { type: 'string', value: 'A' }]]);
  assert.deepEqual(extractTemplates(single).map((t) => t.name), ['default']);

  const multi = new Map([[
    'tokenizer.chat_template',
    { type: 'array', value: [{ name: 'tool_use', template: 'B' }, { name: 'default', template: 'C' }] },
  ]]);
  const templates = extractTemplates(multi);
  assert.deepEqual(templates.map((t) => t.name), ['tool_use', 'default']);
  assert.deepEqual(templates.map((t) => t.text), ['B', 'C']);
});

test('multiple templates are reported so the engine default is not a surprise', () => {
  const metadata = new Map(
    Object.entries(baseMetadata()).map(([k, v]) => [k, { type: v.type, value: v.value }]),
  );
  metadata.set('tokenizer.chat_template', {
    type: 'array',
    value: [
      { name: 'tool_use', template: CHATML_TOOLS_TEMPLATE },
      { name: 'default', template: CHATML_NO_TOOLS_TEMPLATE },
    ],
  });
  const analysis = analyze({ metadata, tensors: [], version: 3, headerBytes: 0 }, {});
  assert.ok(ids(analysis).includes('multiple-templates'));
  // at least one template is tool-aware, so no hard failure on tools
  assert.ok(!ids(analysis).includes('tools-unsupported'));
});
