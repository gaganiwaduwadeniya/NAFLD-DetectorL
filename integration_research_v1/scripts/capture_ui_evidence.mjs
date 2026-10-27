import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const root = '/mnt/g/nafld-detectorl';
const fixtureRelative = process.argv[2];
const fixture = path.join(root, fixtureRelative);
const screenshotDir = path.join(root, 'integration_research_v1/evidence/screenshots');
const chrome = '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe';
const dataDir = `/tmp/nafld-ch6-browser-db-${process.pid}`;
const logs = { service: [], app: [], chrome: [], browserConsole: [] };
const children = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function launch(command, args, env, bucket) {
  const child = spawn(command, args, { cwd: root, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => bucket.push(String(chunk).replaceAll(root, '<APP_ROOT>')));
  children.push(child);
  return child;
}

async function waitHttp(url, timeoutMs, requireOk = false) {
  const deadline = Date.now() + timeoutMs;
  let last = 'not attempted';
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      last = `HTTP ${response.status}`;
      if (!requireOk || response.ok) return response;
    } catch (error) { last = error.message; }
    await sleep(750);
  }
  throw new Error(`Timeout waiting for ${url}: ${last}`);
}

class Cdp {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
      } else if (message.method === 'Runtime.consoleAPICalled') {
        logs.browserConsole.push(message.params);
      }
    };
  }
  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }
}

class CdpPipe {
  constructor(input, output) {
    this.input = input;
    this.nextId = 1;
    this.pending = new Map();
    this.sessionId = null;
    let buffer = '';
    output.on('data', chunk => {
      buffer += chunk.toString();
      const messages = buffer.split('\0');
      buffer = messages.pop() || '';
      for (const raw of messages) {
        if (!raw) continue;
        const message = JSON.parse(raw);
        if (message.id && this.pending.has(message.id)) {
          const { resolve, reject } = this.pending.get(message.id);
          this.pending.delete(message.id);
          message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
        } else if (message.method === 'Runtime.consoleAPICalled') logs.browserConsole.push(message.params);
      }
    });
  }
  send(method, params = {}, useSession = true) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const message = { id, method, params };
      if (useSession && this.sessionId) message.sessionId = this.sessionId;
      this.input.write(JSON.stringify(message) + '\0');
    });
  }
}

async function waitExpression(cdp, expression, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await cdp.send('Runtime.evaluate', { expression: `Boolean(${expression})`, returnByValue: true });
    if (result.result.value) return;
    await sleep(500);
  }
  throw new Error(`Browser condition timed out: ${expression}`);
}

async function evaluate(cdp, expression) {
  const result = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

async function capture(cdp, filename) {
  const shot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, fromSurface: true });
  const target = path.join(screenshotDir, filename);
  await writeFile(target, Buffer.from(shot.data, 'base64'));
  return target;
}

await mkdir(screenshotDir, { recursive: true });
const service = launch(
  path.join(root, '.runtime/pytorch-research-v1/bin/python'),
  ['-m', 'uvicorn', 'pytorch_service.app:app', '--host', '127.0.0.1', '--port', '5003'],
  { PYTHONPATH: root, PYTHONPYCACHEPREFIX: '/tmp/nafld-ch6-browser-pyc' }, logs.service,
);
let cdp;
try {
  await waitHttp('http://127.0.0.1:5003/ready', 900000, true);
  launch('npm', ['run', 'dev'], {
    PORT: '3012', MODEL_PROVIDER: 'pytorch_research_v1', ML_SERVICE_URL: 'http://127.0.0.1:5003',
    ML_INFERENCE_TIMEOUT_MS: '60000', STORE_UPLOAD_IMAGES: 'false', LOCAL_DATA_DIR: dataDir,
    JWT_SECRET: 'evidence-only-browser-secret',
  }, logs.app);
  await waitHttp('http://127.0.0.1:3012/api/auth/me', 300000);
  const registerResponse = await fetch('http://127.0.0.1:3012/api/auth/register', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: 'browser@example.test', password: 'local-evidence-password', name: 'Research Reviewer', role: 'doctor' }),
  });
  const registration = await registerResponse.json();
  if (!registerResponse.ok) throw new Error(`Registration failed: ${JSON.stringify(registration)}`);

  const chromeChild = spawn(chrome, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
    '--remote-debugging-pipe', `--user-data-dir=C:\\Windows\\Temp\\nafld-ch6-${process.pid}`, 'about:blank',
  ], { cwd: root, env: process.env, stdio: ['ignore', 'pipe', 'pipe', 'pipe', 'pipe'] });
  children.push(chromeChild);
  for (const stream of [chromeChild.stdout, chromeChild.stderr]) stream.on('data', chunk => logs.chrome.push(String(chunk)));
  cdp = new CdpPipe(chromeChild.stdio[3], chromeChild.stdio[4]);
  await cdp.send('Browser.getVersion', {}, false);
  const targetList = await cdp.send('Target.getTargets', {}, false);
  const pageTarget = targetList.targetInfos.find(item => item.type === 'page');
  const attached = await cdp.send('Target.attachToTarget', { targetId: pageTarget.targetId, flatten: true }, false);
  cdp.sessionId = attached.sessionId;
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('DOM.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp.send('Page.navigate', { url: 'http://127.0.0.1:3012/' });
  await waitExpression(cdp, `document.readyState === 'complete'`);
  await evaluate(cdp, `localStorage.setItem('nafld_token', ${JSON.stringify(registration.token)}); localStorage.setItem('nafld_user', ${JSON.stringify(JSON.stringify(registration.user))}); location.reload();`);
  await waitExpression(cdp, `document.querySelector('#doctor-dashboard')`);
  await capture(cdp, '01_research_dashboard.png');

  const documentNode = await cdp.send('DOM.getDocument', { depth: -1 });
  const fileNode = await cdp.send('DOM.querySelector', { nodeId: documentNode.root.nodeId, selector: '#input-file-upload' });
  const windowsFixture = `G:\\NAFLD-DetectorL\\${fixtureRelative.replaceAll('/', '\\')}`;
  await cdp.send('DOM.setFileInputFiles', { nodeId: fileNode.nodeId, files: [windowsFixture] });
  await waitExpression(cdp, `document.querySelector('#uploader-preview-container')`);
  await evaluate(cdp, `(() => { const set=(s,v)=>{const e=document.querySelector(s); const p=Object.getPrototypeOf(e); Object.getOwnPropertyDescriptor(p,'value').set.call(e,v); e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true}));}; set('#patient-name','Research Case A'); set('#patient-age','45'); set('#patient-gender','Other'); })()`);
  await capture(cdp, '02_upload_and_validation.png');
  await waitExpression(cdp, `!document.querySelector('#btn-submit-diagnostic').disabled`);
  await evaluate(cdp, `document.querySelector('#btn-submit-diagnostic').click()`);
  await waitExpression(cdp, `document.querySelector('#final-stage-label')`, 180000);
  const genuineModelResult = await evaluate(cdp, `document.querySelector('#final-stage-label').textContent`);
  await capture(cdp, '04_conditional_grading_result.png');
  await evaluate(cdp, `document.querySelector('#provenance-details').open=true`);
  await capture(cdp, '05_probability_and_provenance.png');
  await evaluate(cdp, `document.querySelector('#btn-generate-gradcam').click()`);
  await waitExpression(cdp, `document.querySelector('#gradcam-image')`, 180000);
  await capture(cdp, '06_gradcam_example.png');

  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await capture(cdp, '09_mobile_layout.png');
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp.send('Page.navigate', { url: 'http://127.0.0.1:3012/history' });
  await waitExpression(cdp, `document.querySelector('#scan-table-container')`);
  await capture(cdp, '07_history.png');

  service.kill('SIGTERM');
  await sleep(1500);
  await cdp.send('Page.navigate', { url: 'http://127.0.0.1:3012/' });
  await waitExpression(cdp, `document.querySelector('#doctor-dashboard')`);
  const secondDocument = await cdp.send('DOM.getDocument', { depth: -1 });
  const secondFile = await cdp.send('DOM.querySelector', { nodeId: secondDocument.root.nodeId, selector: '#input-file-upload' });
  await cdp.send('DOM.setFileInputFiles', { nodeId: secondFile.nodeId, files: [windowsFixture] });
  await evaluate(cdp, `(() => { const set=(s,v)=>{const e=document.querySelector(s); const p=Object.getPrototypeOf(e); Object.getOwnPropertyDescriptor(p,'value').set.call(e,v); e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true}));}; set('#patient-name','Research Case B'); set('#patient-age','52'); set('#patient-gender','Other'); })()`);
  await waitExpression(cdp, `!document.querySelector('#btn-submit-diagnostic').disabled`);
  await evaluate(cdp, `document.querySelector('#btn-submit-diagnostic').click()`);
  await waitExpression(cdp, `document.querySelector('#model-unavailable-banner')`, 90000);
  await capture(cdp, '08_service_error.png');

  const result = {
    captured_at: new Date().toISOString(),
    fixture: { filename: path.basename(fixture), source: 'approved anonymized consolidation example' },
    identity: 'synthetic Research Reviewer / Research Case A and B',
    screenshots: ['01_research_dashboard.png', '02_upload_and_validation.png', '04_conditional_grading_result.png', '05_probability_and_provenance.png', '06_gradcam_example.png', '07_history.png', '08_service_error.png', '09_mobile_layout.png'],
    genuine_model_result: genuineModelResult,
    browser_console_events: logs.browserConsole.length,
    training_performed: false,
    external_uploads: false,
  };
  await writeFile(path.join(root, 'integration_research_v1/test-results/browser_capture.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
} finally {
  if (cdp) cdp.input.end();
  for (const child of children.reverse()) if (!child.killed) child.kill('SIGTERM');
}
