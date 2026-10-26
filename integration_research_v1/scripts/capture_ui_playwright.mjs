import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('/home/bimalawijekoon/n1-lms-prototype/frontend/node_modules/playwright-core');
const root = '/mnt/g/nafld-detectorl';
const fixture = path.join(root, process.argv[2]);
const evidence = path.join(root, 'integration_research_v1/evidence/screenshots');
const resultsPath = path.join(root, 'integration_research_v1/test-results/browser_capture.json');
const children = [];
const logs = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function launch(command, args, env = {}) {
  const child = spawn(command, args, { cwd: root, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => logs.push(String(chunk).replaceAll(root, '<APP_ROOT>').replaceAll('/mnt/g/NAFLD New', '<MODEL_ROOT>')));
  children.push(child);
  return child;
}

async function waitHttp(url, timeoutMs, requireOk = false) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { const response = await fetch(url); if (!requireOk || response.ok) return response; } catch {}
    await sleep(750);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

await mkdir(evidence, { recursive: true });
const service = launch(path.join(root, '.runtime/pytorch-research-v1/bin/python'), ['-m', 'uvicorn', 'pytorch_service.app:app', '--host', '127.0.0.1', '--port', '5005'], {
  PYTHONPATH: root, NAFLD_RESEARCH_ROOT: '/mnt/g/NAFLD New', PYTHONPYCACHEPREFIX: '/tmp/nafld-ch6-playwright-pyc',
});
let browser;
try {
  await waitHttp('http://127.0.0.1:5005/ready', 900000, true);
  launch('npm', ['run', 'dev'], {
    PORT: '3015', MODEL_PROVIDER: 'pytorch_research_v1', ML_SERVICE_URL: 'http://127.0.0.1:5005', ML_INFERENCE_TIMEOUT_MS: '60000',
    STORE_UPLOAD_IMAGES: 'false', LOCAL_DATA_DIR: `/tmp/nafld-ch6-playwright-db-${process.pid}`, JWT_SECRET: 'evidence-only-playwright-secret',
  });
  await waitHttp('http://127.0.0.1:3015/api/auth/me', 300000);
  const register = await fetch('http://127.0.0.1:3015/api/auth/register', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ email: 'screenshots@example.test', password: 'local-evidence-password', name: 'Research Reviewer', role: 'doctor' }) });
  const auth = await register.json();
  if (!register.ok) throw new Error(`Registration failed: ${JSON.stringify(auth)}`);

  browser = await chromium.launch({ headless: true, executablePath: '/home/bimalawijekoon/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome', args: ['--no-sandbox'] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  await context.addInitScript(({ token, user }) => { localStorage.setItem('nafld_token', token); localStorage.setItem('nafld_user', JSON.stringify(user)); }, { token: auth.token, user: auth.user });
  const page = await context.newPage();
  page.setDefaultTimeout(120000);
  page.setDefaultNavigationTimeout(120000);
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
  await page.goto('http://127.0.0.1:3015/', { waitUntil: 'domcontentloaded' });
  await sleep(3000);
  await writeFile(path.join(root, 'integration_research_v1/test-results/browser_debug.html'), await page.content());
  await page.screenshot({ path: path.join(root, 'integration_research_v1/test-results/browser_debug.png'), fullPage: true });
  console.log(JSON.stringify({ browser_url: page.url(), browser_title: await page.title(), browser_errors: browserErrors }));
  await page.locator('#doctor-dashboard').waitFor();
  await page.screenshot({ path: path.join(evidence, '01_research_dashboard.png'), fullPage: true });

  await page.locator('#input-file-upload').setInputFiles(fixture);
  await page.locator('#patient-name').fill('Research Case A');
  await page.locator('#patient-age').fill('45');
  await page.locator('#patient-gender').selectOption('Other');
  await page.screenshot({ path: path.join(evidence, '02_upload_and_validation.png'), fullPage: true });
  await page.locator('#btn-submit-diagnostic').click();
  await page.locator('#final-stage-label').waitFor({ timeout: 180000 });
  const finalLabel = await page.locator('#final-stage-label').textContent();
  await page.screenshot({ path: path.join(evidence, '04_conditional_grading_result.png'), fullPage: true });
  await page.locator('#provenance-details').evaluate(element => { element.open = true; });
  await page.screenshot({ path: path.join(evidence, '05_probability_and_provenance.png'), fullPage: true });
  await page.locator('#btn-generate-gradcam').click();
  await page.locator('#gradcam-image').waitFor({ timeout: 180000 });
  await page.screenshot({ path: path.join(evidence, '06_gradcam_example.png'), fullPage: true });

  await page.goto('http://127.0.0.1:3015/history', { waitUntil: 'domcontentloaded' });
  await page.locator('#scan-table-container').waitFor();
  await page.screenshot({ path: path.join(evidence, '07_history_and_scan_detail.png'), fullPage: true });
  await page.locator('tbody tr').first().click();
  await page.locator('#scan-detail-page').waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(evidence, '09_mobile_layout.png'), fullPage: true });

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('http://127.0.0.1:3015/', { waitUntil: 'domcontentloaded' });
  await page.locator('#input-file-upload').setInputFiles(fixture);
  await page.locator('#patient-name').fill('Research Case B');
  await page.locator('#patient-age').fill('52');
  await page.locator('#patient-gender').selectOption('Other');
  service.kill('SIGTERM'); await sleep(1500);
  await page.locator('#btn-submit-diagnostic').click();
  await page.locator('#model-unavailable-banner').waitFor({ timeout: 90000 });
  await page.screenshot({ path: path.join(evidence, '08_invalid_input_or_service_error.png'), fullPage: true });

  const files = ['01_research_dashboard.png', '02_upload_and_validation.png', '04_conditional_grading_result.png', '05_probability_and_provenance.png', '06_gradcam_example.png', '07_history_and_scan_detail.png', '08_invalid_input_or_service_error.png', '09_mobile_layout.png'];
  const result = { captured_at: new Date().toISOString(), final_label: finalLabel, files, browser_errors: browserErrors, privacy: 'approved de-identified input plus synthetic Research Reviewer and Research Case identities', genuine_model: true, training_performed: false, external_uploads: false, logs_tail: logs.join('').split('\n').filter(Boolean).slice(-100) };
  await writeFile(resultsPath, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
} finally {
  if (browser) await browser.close().catch(() => {});
  for (const child of children.reverse()) if (!child.killed) child.kill('SIGTERM');
}
