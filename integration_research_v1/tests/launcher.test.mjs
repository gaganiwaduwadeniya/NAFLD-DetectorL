import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const dispatcher = 'scripts/run-local-research-ui.mjs';
const bashLauncher = 'scripts/run-local-research-ui.sh';
const windowsLauncher = 'scripts/run-local-research-ui.ps1';

test('POSIX launcher has valid Bash syntax', () => {
  const result = spawnSync('bash', ['-n', bashLauncher], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
});

test('platform-neutral launcher help documents the one-command contract', () => {
  const result = spawnSync(process.execPath, [dispatcher, '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Usage: npm run app/);
  assert.match(result.stdout, /--check/);
  assert.match(result.stdout, /--prepare-only/);
  assert.match(result.stdout, /--no-open/);
});

test('package script dispatches by platform and both launchers retain fail-closed settings', () => {
  const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
  const dispatchSource = readFileSync(dispatcher, 'utf8');
  const bashSource = readFileSync(bashLauncher, 'utf8');
  const windowsSource = readFileSync(windowsLauncher, 'utf8');

  assert.equal(packageJson.scripts.app, 'node scripts/run-local-research-ui.mjs');
  assert.match(dispatchSource, /process\.platform === ['"]win32['"]/);
  assert.match(dispatchSource, /run-local-research-ui\.ps1/);
  assert.match(dispatchSource, /run-local-research-ui\.sh/);

  for (const source of [bashSource, windowsSource]) {
    assert.match(source, /pytorch_research_v1/);
    assert.match(source, /STORE_UPLOAD_IMAGES/);
    assert.match(source, /USE_LOCAL_DB/);
    assert.match(source, /Expected 5 selected DenseNet121 checkpoints/);
    assert.match(source, /Expected 5 selected EfficientNet-B0 checkpoints/);
    assert.match(source, /pytorch_service[\\/]research_v1/);
    assert.doesNotMatch(source, /NAFLD New/);
    assert.doesNotMatch(source, /NAFLD_RESEARCH_ROOT/);
  }
  assert.match(windowsSource, /Scripts[\\/]python\.exe/);
  assert.match(windowsSource, /pytorch-research-v1-windows/);
  assert.match(windowsSource, /node-platform-windows/);
});

test('unknown launcher options fail before changing the environment', () => {
  const result = spawnSync(process.execPath, [dispatcher, '--not-a-real-option'], { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unknown option/);
});

test('native Windows launcher parses and prints help when Windows PowerShell is available', (t) => {
  const powershell = '/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe';
  if (!existsSync(powershell)) {
    t.skip('Windows PowerShell is not reachable from this test environment');
    return;
  }

  const converted = spawnSync('wslpath', ['-w', windowsLauncher], { encoding: 'utf8' });
  assert.equal(converted.status, 0, converted.stderr);
  const windowsPath = converted.stdout.trim();
  const result = spawnSync(
    powershell,
    ['-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', windowsPath, '--help'],
    { encoding: 'utf8' },
  );
  if (result.status !== 0 && /WSL .*UtilBindVsockAnyPort|socket failed/i.test(result.stderr)) {
    t.skip('Windows interop is blocked in this WSL test environment');
    return;
  }
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Native Windows mode/);
  assert.match(result.stdout, /does not require WSL/i);
});
