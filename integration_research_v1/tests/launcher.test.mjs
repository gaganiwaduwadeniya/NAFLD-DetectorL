import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const launcher = 'scripts/run-local-research-ui.sh';

test('launcher has valid Bash syntax', () => {
  const result = spawnSync('bash', ['-n', launcher], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
});

test('launcher help documents the one-command contract without starting services', () => {
  const result = spawnSync('bash', [launcher, '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Usage: npm run app/);
  assert.match(result.stdout, /--check/);
  assert.match(result.stdout, /--prepare-only/);
  assert.match(result.stdout, /--no-open/);
});

test('package script and launcher retain fail-closed research settings', () => {
  const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
  const source = readFileSync(launcher, 'utf8');

  assert.equal(packageJson.scripts.app, 'bash scripts/run-local-research-ui.sh');
  assert.match(source, /MODEL_PROVIDER=pytorch_research_v1/);
  assert.match(source, /STORE_UPLOAD_IMAGES=false/);
  assert.match(source, /USE_LOCAL_DB=true/);
  assert.match(source, /Expected 5 selected DenseNet121 checkpoints/);
  assert.match(source, /Expected 5 selected EfficientNet-B0 checkpoints/);
});

test('unknown launcher options fail before changing the environment', () => {
  const result = spawnSync('bash', [launcher, '--not-a-real-option'], { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unknown option/);
});
