import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, copyFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = '/mnt/g/nafld-detectorl';
const fixtureRelative = process.argv[2];
const fixture = path.join(root, fixtureRelative);
const evidence = path.join(root, 'integration_research_v1/evidence/screenshots');
const extension = path.join(root, 'integration_research_v1/test-results/chrome_evidence_extension');
const chrome = '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe';
const browserHost = process.env.WSL_BROWSER_HOST || '172.22.178.173';
const browserBase = `http://${browserHost}:3013`;
const children = [];
const logs = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function launch(command, args, env = {}) {
  const child = spawn(command, args, { cwd: root, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => logs.push(String(chunk).replaceAll(root, '<APP_ROOT>')));
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

async function chromeShot(name, url, width = 1440, height = 1000, waitMs = 10000) {
  const outputWindows = `G:\\NAFLD-DetectorL\\integration_research_v1\\evidence\\screenshots\\${name}`;
  const extensionWindows = 'G:\\NAFLD-DetectorL\\integration_research_v1\\test-results\\chrome_evidence_extension';
  const profile = `C:\\Windows\\Temp\\nafld-shot-${process.pid}-${name.replace(/\W/g, '')}`;
  await new Promise((resolve, reject) => {
    const child = spawn(chrome, [
      '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
      `--disable-extensions-except=${extensionWindows}`, `--load-extension=${extensionWindows}`,
      `--user-data-dir=${profile}`, `--window-size=${width},${height}`, `--virtual-time-budget=${waitMs}`,
      `--screenshot=${outputWindows}`, url,
    ], { stdio: ['ignore', 'pipe', 'pipe'] });
    let settled = false;
    child.on('error', error => { if (!settled) { settled = true; reject(error); } });
    child.on('exit', async code => {
      if (settled) return;
      try { const info = await stat(path.join(evidence, name)); if (info.size > 0) { settled = true; resolve(); return; } } catch {}
      settled = true; reject(new Error(`Chrome screenshot ${name} exited ${code}`));
    });
    const poll = setInterval(async () => {
      if (settled) return clearInterval(poll);
      try {
        const info = await stat(path.join(evidence, name));
        if (info.size > 0) { settled = true; clearInterval(poll); child.kill('SIGTERM'); resolve(); }
      } catch {}
    }, 500);
    setTimeout(() => { if (!settled) { settled = true; clearInterval(poll); child.kill('SIGTERM'); reject(new Error(`Chrome screenshot ${name} timed out`)); } }, Math.max(waitMs + 30000, 45000));
  });
  return path.join(evidence, name);
}

await mkdir(evidence, { recursive: true });
await mkdir(extension, { recursive: true });
const service = launch(path.join(root, '.runtime/pytorch-research-v1/bin/python'), ['-m', 'uvicorn', 'pytorch_service.app:app', '--host', '127.0.0.1', '--port', '5004'], {
  PYTHONPATH: root, PYTHONPYCACHEPREFIX: '/tmp/nafld-ch6-cli-pyc',
});
try {
  await waitHttp('http://127.0.0.1:5004/ready', 900000, true);
  launch('npm', ['run', 'dev'], {
    PORT: '3013', MODEL_PROVIDER: 'pytorch_research_v1', ML_SERVICE_URL: 'http://127.0.0.1:5004', ML_INFERENCE_TIMEOUT_MS: '60000',
    STORE_UPLOAD_IMAGES: 'false', LOCAL_DATA_DIR: `/tmp/nafld-ch6-cli-db-${process.pid}`, JWT_SECRET: 'evidence-only-cli-secret',
  });
  await waitHttp('http://127.0.0.1:3013/api/auth/me', 300000);
  const register = await fetch('http://127.0.0.1:3013/api/auth/register', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ email: 'screenshots@example.test', password: 'local-evidence-password', name: 'Research Reviewer', role: 'doctor' }) });
  const auth = await register.json();
  if (!register.ok) throw new Error(`Registration failed: ${JSON.stringify(auth)}`);
  const bytes = await readFile(fixture);
  const form = new FormData();
  form.append('image', new Blob([bytes], { type: 'image/png' }), 'approved-deidentified-input.png');
  form.append('patientName', 'Research Case A'); form.append('patientAge', '45'); form.append('patientGender', 'Other');
  const predictionResponse = await fetch('http://127.0.0.1:3013/api/predict', { method: 'POST', headers: { authorization: `Bearer ${auth.token}` }, body: form });
  const prediction = await predictionResponse.json();
  if (!predictionResponse.ok) throw new Error(`Prediction failed: ${JSON.stringify(prediction)}`);
  const explainForm = new FormData(); explainForm.append('image', new Blob([bytes], { type: 'image/png' }), 'approved-deidentified-input.png');
  const explain = await fetch('http://127.0.0.1:3013/api/explain', { method: 'POST', headers: { authorization: `Bearer ${auth.token}` }, body: explainForm });
  await writeFile(path.join(evidence, '06_real_gradcam_e2e.png'), Buffer.from(await explain.arrayBuffer()));

  await writeFile(path.join(extension, 'manifest.json'), JSON.stringify({ manifest_version: 3, name: 'NAFLD local evidence helper', version: '1.0.0', permissions: [], host_permissions: [`${browserBase}/*`], content_scripts: [{ matches: [`${browserBase}/*`], js: ['content.js'], run_at: 'document_end' }], web_accessible_resources: [{ resources: ['fixture.png'], matches: [`${browserBase}/*`] }] }, null, 2));
  await copyFile(fixture, path.join(extension, 'fixture.png'));
  const content = `
    localStorage.setItem('nafld_token', ${JSON.stringify(auth.token)});
    localStorage.setItem('nafld_user', ${JSON.stringify(JSON.stringify(auth.user))});
    const setValue=(selector,value)=>{const element=document.querySelector(selector);if(!element)return false;const setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(element),'value').set;setter.call(element,value);element.dispatchEvent(new Event('input',{bubbles:true}));element.dispatchEvent(new Event('change',{bubbles:true}));return true;};
    const prepare=async(click)=>{if(!document.querySelector('#input-file-upload'))return setTimeout(()=>prepare(click),200);const payload=await (await fetch(chrome.runtime.getURL('fixture.png'))).blob();const file=new File([payload],'approved-deidentified-input.png',{type:'image/png'});const transfer=new DataTransfer();transfer.items.add(file);const input=document.querySelector('#input-file-upload');input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));setValue('#patient-name',click?'Research Case B':'Research Case A');setValue('#patient-age',click?'52':'45');setValue('#patient-gender','Other');if(click)setTimeout(()=>document.querySelector('#btn-submit-diagnostic')?.click(),500);};
    if(location.hash==='#capture-upload')prepare(false);
    if(location.hash==='#capture-error')prepare(true);
    if(location.hash==='#provenance')setTimeout(()=>{const detail=document.querySelector('#provenance-details');if(detail)detail.open=true;},1000);
  `;
  await writeFile(path.join(extension, 'content.js'), content);

  const shots = [];
  shots.push(await chromeShot('01_research_dashboard.png', `${browserBase}/`));
  shots.push(await chromeShot('02_upload_and_validation.png', `${browserBase}/#capture-upload`));
  shots.push(await chromeShot('04_conditional_grading_result.png', `${browserBase}/scan/${prediction.id}`));
  shots.push(await chromeShot('05_probability_and_provenance.png', `${browserBase}/scan/${prediction.id}#provenance`));
  shots.push(await chromeShot('07_history_and_scan_detail.png', `${browserBase}/history`));
  shots.push(await chromeShot('09_mobile_layout.png', `${browserBase}/scan/${prediction.id}`, 390, 844));
  service.kill('SIGTERM'); await sleep(1500);
  shots.push(await chromeShot('08_invalid_input_or_service_error.png', `${browserBase}/#capture-error`, 1440, 1000, 12000));

  const result = { captured_at: new Date().toISOString(), genuine_record: { id: prediction.id, final_stage: prediction.finalStage, label: prediction.finalStageLabel }, screenshots: shots, gradcam_status: explain.status, privacy: 'approved de-identified input and synthetic identities only', logs_tail: logs.join('').split('\n').filter(Boolean).slice(-80) };
  await writeFile(path.join(root, 'integration_research_v1/test-results/browser_capture.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
} finally {
  for (const child of children.reverse()) if (!child.killed) child.kill('SIGTERM');
}
