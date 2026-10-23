import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const root = '/mnt/g/nafld-detectorl';
const output = path.join(root, 'integration_research_v1/test-results/authenticated_e2e.json');
const dataDir = `/tmp/nafld-ch6-e2e-db-${process.pid}`;
const fixture = path.join(root, process.argv[2]);
const serviceLogs = [];
const appLogs = [];

function launch(command, args, env, logs) {
  const child = spawn(command, args, { cwd: root, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => logs.push(String(chunk).replaceAll(root, '<APP_ROOT>').replaceAll('/mnt/g/NAFLD New', '<MODEL_ROOT>')));
  return child;
}

async function waitJson(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
      last = `${response.status}`;
    } catch (error) { last = error.message; }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error(`Timed out waiting for ${url}: ${last}`);
}

async function waitHttp(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { await fetch(url); return; } catch { await new Promise(resolve => setTimeout(resolve, 1000)); }
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  return { status: response.status, body };
}

await mkdir(path.dirname(output), { recursive: true });
const python = launch(
  path.join(root, '.runtime/pytorch-research-v1/bin/python'),
  ['-m', 'uvicorn', 'pytorch_service.app:app', '--host', '127.0.0.1', '--port', '5002'],
  { PYTHONPATH: root, NAFLD_RESEARCH_ROOT: '/mnt/g/NAFLD New', PYTHONPYCACHEPREFIX: '/tmp/nafld-ch6-pyc' },
  serviceLogs,
);
let app;
try {
  const ready = await waitJson('http://127.0.0.1:5002/ready', 900_000);
  app = launch(
    'npm', ['run', 'dev'],
    {
      PORT: '3011', MODEL_PROVIDER: 'pytorch_research_v1', ML_SERVICE_URL: 'http://127.0.0.1:5002',
      ML_INFERENCE_TIMEOUT_MS: '60000', STORE_UPLOAD_IMAGES: 'false', LOCAL_DATA_DIR: dataDir,
      JWT_SECRET: 'evidence-only-local-secret',
    }, appLogs,
  );
  await waitHttp('http://127.0.0.1:3011/api/auth/me', 300_000);
  const registration = await api('http://127.0.0.1:3011/api/auth/register', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: 'research@example.test', password: 'local-evidence-password', name: 'Research Reviewer', role: 'doctor' }),
  });
  if (registration.status !== 201) throw new Error(`Registration failed (${registration.status}): ${JSON.stringify(registration.body)} logs=${appLogs.join('').slice(-4000)}`);
  const token = registration.body.token;
  const auth = { authorization: `Bearer ${token}` };
  const image = await readFile(fixture);
  const predictionForm = () => {
    const form = new FormData();
    form.append('image', new Blob([image], { type: 'image/png' }), 'approved-deidentified-input.png');
    form.append('patientName', 'Research Case A');
    form.append('patientAge', '45');
    form.append('patientGender', 'Other');
    return form;
  };
  const unauthorized = await api('http://127.0.0.1:3011/api/scans');
  const prediction = await api('http://127.0.0.1:3011/api/predict', { method: 'POST', headers: auth, body: predictionForm() });
  if (prediction.status !== 200) throw new Error(`Prediction failed: ${JSON.stringify(prediction.body)}`);
  const health = await api('http://127.0.0.1:3011/api/model/health', { headers: auth });
  const history = await api('http://127.0.0.1:3011/api/scans', { headers: auth });
  const detail = await api(`http://127.0.0.1:3011/api/scans/${prediction.body.id}`, { headers: auth });
  const explainForm = new FormData();
  explainForm.append('image', new Blob([image], { type: 'image/png' }), 'approved-deidentified-input.png');
  const explainResponse = await fetch('http://127.0.0.1:3011/api/explain', { method: 'POST', headers: auth, body: explainForm });
  const explanation = Buffer.from(await explainResponse.arrayBuffer());
  const explanationPath = path.join(root, 'integration_research_v1/evidence/screenshots/06_real_gradcam_e2e.png');
  await writeFile(explanationPath, explanation);

  const beforeOutage = history.body.length;
  python.kill('SIGTERM');
  await new Promise(resolve => setTimeout(resolve, 1500));
  const outage = await api('http://127.0.0.1:3011/api/predict', { method: 'POST', headers: auth, body: predictionForm() });
  const afterOutageHistory = await api('http://127.0.0.1:3011/api/scans', { headers: auth });
  const result = {
    executed_at: new Date().toISOString(),
    exact_runtime_ready: ready,
    cases: {
      unauthorized_history: { status: unauthorized.status, passed: unauthorized.status === 401 },
      authenticated_prediction: { status: prediction.status, record: prediction.body },
      authenticated_health: health,
      versioned_history: { status: history.status, count: history.body.length, exact_record_match: detail.body.id === prediction.body.id },
      gradcam: { status: explainResponse.status, content_type: explainResponse.headers.get('content-type'), bytes: explanation.length, metadata: explainResponse.headers.get('x-nafld-explanation'), path: explanationPath },
      service_outage: { status: outage.status, body: outage.body, records_before: beforeOutage, records_after: afterOutageHistory.body.length, zero_database_writes: beforeOutage === afterOutageHistory.body.length },
    },
    logs: { model_service: serviceLogs.join('').split('\n').filter(Boolean).slice(-80), express: appLogs.join('').split('\n').filter(Boolean).slice(-80) },
    privacy: { input: 'approved anonymized consolidation example', user: 'synthetic research identity', image_retained: prediction.body.imageStored },
    training_performed: false,
    external_uploads: false,
  };
  await writeFile(output, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({ output, finalStage: prediction.body.finalStage, gradcamStatus: explainResponse.status, outageStatus: outage.status, zeroDatabaseWrites: result.cases.service_outage.zero_database_writes }, null, 2));
} finally {
  if (app) app.kill('SIGTERM');
  if (!python.killed) python.kill('SIGTERM');
}
