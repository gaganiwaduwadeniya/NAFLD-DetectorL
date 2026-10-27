#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { writeSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const supportedOptions = new Set(['--check', '--prepare-only', '--no-open', '--help', '-h']);
const invalidOption = process.argv.slice(2).find((argument) => !supportedOptions.has(argument));

if (invalidOption) {
  writeSync(2, `[NAFLD] ERROR: Unknown option: ${invalidOption}. Run 'npm run app -- --help'.\n`);
  process.exit(1);
}

const scriptsDirectory = dirname(fileURLToPath(import.meta.url));
const rootDirectory = dirname(scriptsDirectory);
const windows = process.platform === 'win32';
const command = windows ? 'powershell.exe' : 'bash';
const launcher = join(
  scriptsDirectory,
  windows ? 'run-local-research-ui.ps1' : 'run-local-research-ui.sh',
);
const launcherArguments = windows
  ? ['-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', launcher, ...process.argv.slice(2)]
  : [launcher, ...process.argv.slice(2)];

const result = spawnSync(command, launcherArguments, {
  cwd: rootDirectory,
  env: process.env,
  stdio: 'inherit',
});

if (result.error) {
  writeSync(2, `[NAFLD] ERROR: Could not start ${command}: ${result.error.message}\n`);
  process.exit(1);
}

process.exit(result.status ?? 1);
