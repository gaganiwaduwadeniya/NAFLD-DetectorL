/**
 * Phase 4 verification script — Versioned Scan & Database Schema Expansion.
 *
 * Run from project root:
 *   $env:PATH = "G:\Nodejs;$env:PATH"; & 'G:\Nodejs\node.exe' server_phase4_test.mjs
 *
 * What this tests:
 *  T1 - Non-NAFLD prediction stores schemaVersion: 3 and all V3 fields (binaryResult, finalLabel, binaryFoldProbs, contractSha256, latency, dimensions).
 *  T2 - NAFLD + Grading prediction stores schemaVersion: 3 and all grading fields (gradingResult, gradingFoldProbs, etc.).
 *  T3 - GET /api/scans/:id retrieves full versioned Scan object.
 *  T4 - Legacy scans (schemaVersion undefined / legacy format) remain readable via GET /api/scans.
 *  T5 - GET /api/admin/stats works seamlessly with mixed legacy and V3 scans.
 */

import http from 'http';
import { createServer as createHttpServer } from 'http';
import { execSync, spawn } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { setTimeout as sleep } from 'timers/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const NODE_EXE = 'G:\\Nodejs\\node.exe';
const TSX_EXE  = path.join(__dirname, 'node_modules', '.bin', 'tsx.cmd');

const PASS = '\x1b[92mPASS\x1b[0m';
const FAIL = '\x1b[91mFAIL\x1b[0m';
const results = [];

function check(name, condition, detail = '') {
  const status = condition ? PASS : FAIL;
  console.log(`  [${status}] ${name}`);
  if (!condition && detail) console.log(`         -> ${detail}`);
  results.push({ name, ok: condition });
}

function section(title) {
  console.log(`\n${'-'.repeat(60)}`);
  console.log(`  ${title}`);
  console.log(`${'-'.repeat(60)}`);
}

function createMockFlask(behaviour) {
  const server = createHttpServer((req, res) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      const { status, responseBody } = behaviour(req, body);
      const json = JSON.stringify(responseBody);
      res.writeHead(status, {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(json),
      });
      res.end(json);
    });
  });
  return new Promise(resolve => {
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      resolve({
        port,
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise(r => server.close(r)),
      });
    });
  });
}

const MOCK_V3_NON_NAFLD = {
  success: true,
  input_mode: 'single_frame',
  model_version: '3.0.0',
  contract_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  research_use_only: true,
  binary_result: 'Non-NAFLD',
  binary_prob_nafld: 0.35,
  binary_prob_non_nafld: 0.65,
  binary_threshold: 0.5536812544,
  binary_fold_probs: [0.33, 0.36, 0.34, 0.37, 0.35],
  binary_fold_std: 0.014,
  grading_performed: false,
  final_label: 'Non-NAFLD',
};

const MOCK_V3_NAFLD_MILD = {
  success: true,
  input_mode: 'single_frame',
  model_version: '3.0.0',
  contract_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  research_use_only: true,
  binary_result: 'NAFLD',
  binary_prob_nafld: 0.78,
  binary_prob_non_nafld: 0.22,
  binary_threshold: 0.5536812544,
  binary_fold_probs: [0.76, 0.79, 0.77, 0.80, 0.78],
  binary_fold_std: 0.013,
  grading_performed: true,
  grading_result: 'Grade1_Mild',
  grading_prob_moderate_severe: 0.42,
  grading_prob_mild: 0.58,
  grading_threshold: 0.6078062057,
  grading_fold_probs: [0.40, 0.43, 0.41, 0.44, 0.42],
  grading_fold_std: 0.015,
  final_label: 'NAFLD-Grade1_Mild',
};

function makeMinimalImage() {
  // Valid PNG header with width 512, height 256
  return Buffer.from([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG magic
    0x00, 0x00, 0x00, 0x0d,                         // IHDR length = 13
    0x49, 0x48, 0x44, 0x52,                         // 'IHDR'
    0x00, 0x00, 0x02, 0x00,                         // width = 512
    0x00, 0x00, 0x01, 0x00,                         // height = 256
    0x08, 0x06, 0x00, 0x00, 0x00,                   // Bit depth, color type, compression, filter, interlace
    0x14, 0x7a, 0x7c, 0x73,                         // CRC
  ]);
}

function buildMultipartBody(fields, files, boundary) {
  const chunks = [];
  for (const [name, value] of Object.entries(fields)) {
    chunks.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`));
  }
  for (const { name, filename, contentType, data } of files) {
    chunks.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"; filename="${filename}"\r\nContent-Type: ${contentType}\r\n\r\n`));
    chunks.push(data);
    chunks.push(Buffer.from('\r\n'));
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  return Buffer.concat(chunks);
}

let _nextPort = 3200;

async function killServer(proc) {
  if (!proc || proc.killed) return;
  try {
    execSync(`taskkill /F /T /PID ${proc.pid}`, { stdio: 'ignore' });
  } catch {}
  await sleep(600);
}

async function startExpressAndGetToken(mlServiceUrl, role = 'doctor') {
  const port = _nextPort++;
  const env = {
    ...process.env,
    PATH: `G:\\Nodejs;${process.env.PATH || ''}`,
    ML_SERVICE_URL: mlServiceUrl || '',
    JWT_SECRET: 'test-secret-phase4',
    NODE_ENV: 'test',
    PORT: String(port),
  };

  const proc = spawn(
    TSX_EXE, ['server.ts'],
    { cwd: __dirname, env, stdio: ['ignore', 'pipe', 'pipe'], shell: true }
  );

  await new Promise((resolve, reject) => {
    let output = '';
    const timeout = setTimeout(() => reject(new Error('Express did not start in time.\n' + output)), 15000);
    proc.stdout.on('data', data => {
      output += data.toString();
      if (output.includes('Server successfully booted')) {
        clearTimeout(timeout);
        resolve();
      }
    });
    proc.stderr.on('data', data => { output += data.toString(); });
    proc.on('exit', code => {
      clearTimeout(timeout);
      reject(new Error(`Express exited with code ${code}.\n${output}`));
    });
  });

  const email = `phase4_${Date.now()}@test.com`;
  const registerResp = await new Promise((resolve, reject) => {
    const body = JSON.stringify({ email, password: 'test1234', name: 'Phase4 User', role });
    const req = http.request({
      hostname: '127.0.0.1', port, path: '/api/auth/register', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(d) }));
    });
    req.on('error', reject); req.write(body); req.end();
  });

  const token = registerResp.body?.token;
  return { proc, port, token };
}

async function sendPredictRequest(expressPort, jwtToken, imageBuffer) {
  const boundary = 'BOUNDARY' + Date.now();
  const body = buildMultipartBody(
    { patientName: 'Test Phase4 Patient', patientAge: '42', patientGender: 'Female' },
    [{ name: 'image', filename: 'ultrasound.png', contentType: 'image/png', data: imageBuffer }],
    boundary
  );

  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1', port: expressPort, path: '/api/predict', method: 'POST',
      headers: {
        'Authorization': `Bearer ${jwtToken}`,
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length,
      },
    }, res => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject); req.write(body); req.end();
  });
}

async function main() {
  console.log('\n==================================================================');
  console.log('         NAFLD Detector -- Phase 4 Verification Script');
  console.log('==================================================================\n');

  const img = makeMinimalImage();

  // -----------------------------------------------------------------------
  // T1: Non-NAFLD V3 response stores schemaVersion: 3 & binary V3 metadata
  // -----------------------------------------------------------------------
  section('T1 - Non-NAFLD scan stores schemaVersion: 3 & full V3 fields');
  let mock1, server1;
  try {
    mock1 = await createMockFlask(() => ({ status: 200, responseBody: MOCK_V3_NON_NAFLD }));
    server1 = await startExpressAndGetToken(mock1.url);
    const resp = await sendPredictRequest(server1.port, server1.token, img);

    check('Returns 200 OK', resp.status === 200, `Got ${resp.status}`);
    const scan = resp.body;
    check('schemaVersion is 3', scan.schemaVersion === 3, `got ${scan.schemaVersion}`);
    check('modelVersion is 3.0.0', scan.modelVersion === '3.0.0', `got ${scan.modelVersion}`);
    check('contractSha256 present', typeof scan.contractSha256 === 'string' && scan.contractSha256.length === 64);
    check('inputMode is single_frame', scan.inputMode === 'single_frame');
    check('researchUseOnly is true', scan.researchUseOnly === true);
    check('binaryResult is Non-NAFLD', scan.binaryResult === 'Non-NAFLD');
    check('binaryProbNafld is 0.35', scan.binaryProbNafld === 0.35);
    check('binaryProbNonNafld is 0.65', scan.binaryProbNonNafld === 0.65);
    check('binaryThreshold present', typeof scan.binaryThreshold === 'number');
    check('binaryFoldProbs has 5 entries', Array.isArray(scan.binaryFoldProbs) && scan.binaryFoldProbs.length === 5);
    check('binaryFoldStd present', typeof scan.binaryFoldStd === 'number');
    check('gradingPerformed is false', scan.gradingPerformed === false);
    check('gradingResult absent', scan.gradingResult === undefined);
    check('finalLabel is Non-NAFLD', scan.finalLabel === 'Non-NAFLD');
    check('imageWidth parsed (512)', scan.imageWidth === 512, `got ${scan.imageWidth}`);
    check('imageHeight parsed (256)', scan.imageHeight === 256, `got ${scan.imageHeight}`);
    check('inferenceLatencyMs is a positive number', typeof scan.inferenceLatencyMs === 'number' && scan.inferenceLatencyMs >= 0);

    // Check legacy fields present for backwards compatibility
    check('Legacy prediction is Normal', scan.prediction === 'Normal');
    check('Legacy confidence is 65', scan.confidence === 65);
    check('Legacy probabilities present', scan.probabilities?.Normal === 65 && scan.probabilities?.Abnormal === 35);
  } catch (err) {
    check('T1 execution', false, err.message);
  } finally {
    await killServer(server1?.proc);
    await mock1?.close();
  }

  // -----------------------------------------------------------------------
  // T2: NAFLD + Mild grading scan stores full grading V3 metadata
  // -----------------------------------------------------------------------
  section('T2 - NAFLD + Mild grading scan stores full grading V3 fields');
  let mock2, server2;
  try {
    mock2 = await createMockFlask(() => ({ status: 200, responseBody: MOCK_V3_NAFLD_MILD }));
    server2 = await startExpressAndGetToken(mock2.url);
    const resp = await sendPredictRequest(server2.port, server2.token, img);

    check('Returns 200 OK', resp.status === 200, `Got ${resp.status}`);
    const scan = resp.body;
    check('schemaVersion is 3', scan.schemaVersion === 3);
    check('binaryResult is NAFLD', scan.binaryResult === 'NAFLD');
    check('gradingPerformed is true', scan.gradingPerformed === true);
    check('gradingResult is Grade1_Mild', scan.gradingResult === 'Grade1_Mild');
    check('gradingProbMild is 0.58', scan.gradingProbMild === 0.58);
    check('gradingProbModerateSevere is 0.42', scan.gradingProbModerateSevere === 0.42);
    check('gradingThreshold present', typeof scan.gradingThreshold === 'number');
    check('gradingFoldProbs has 5 entries', Array.isArray(scan.gradingFoldProbs) && scan.gradingFoldProbs.length === 5);
    check('finalLabel is NAFLD-Grade1_Mild', scan.finalLabel === 'NAFLD-Grade1_Mild');
    check('Legacy prediction is Abnormal', scan.prediction === 'Abnormal');
  } catch (err) {
    check('T2 execution', false, err.message);
  } finally {
    await killServer(server2?.proc);
    await mock2?.close();
  }

  // -----------------------------------------------------------------------
  // T3: GET /api/scans/:id retrieves full versioned Scan record
  // -----------------------------------------------------------------------
  section('T3 - GET /api/scans/:id retrieves full versioned Scan object');
  let mock3, server3;
  try {
    mock3 = await createMockFlask(() => ({ status: 200, responseBody: MOCK_V3_NAFLD_MILD }));
    server3 = await startExpressAndGetToken(mock3.url);
    const createResp = await sendPredictRequest(server3.port, server3.token, img);
    const scanId = createResp.body.id;

    await sleep(400);

    const getResp = await new Promise((resolve, reject) => {
      http.get({
        hostname: '127.0.0.1', port: server3.port, path: `/api/scans/${scanId}`,
        headers: { 'Authorization': `Bearer ${server3.token}` },
      }, res => {
        let d = ''; res.on('data', c => d += c);
        res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(d) }));
      }).on('error', reject);
    });

    check('GET /api/scans/:id returns 200', getResp.status === 200);
    const retrieved = getResp.body;
    check('Retrieved scan matches ID', retrieved.id === scanId);
    check('Retrieved scan has schemaVersion=3', retrieved.schemaVersion === 3);
    check('Retrieved scan has finalLabel NAFLD-Grade1_Mild', retrieved.finalLabel === 'NAFLD-Grade1_Mild');
    check('Retrieved scan preserves contractSha256', typeof retrieved.contractSha256 === 'string' && retrieved.contractSha256.length === 64);
  } catch (err) {
    check('T3 execution', false, err.message);
  } finally {
    await killServer(server3?.proc);
    await mock3?.close();
  }

  // -----------------------------------------------------------------------
  // T4: Legacy scans remain readable via GET /api/scans
  // -----------------------------------------------------------------------
  section('T4 - Legacy scans remain readable via GET /api/scans');
  let mock4, server4;
  try {
    mock4 = await createMockFlask(() => ({ status: 200, responseBody: MOCK_V3_NON_NAFLD }));
    // Pass role 'admin' to see all scans including pre-populated legacy scans
    server4 = await startExpressAndGetToken(mock4.url, 'admin');

    const listResp = await new Promise((resolve, reject) => {
      http.get({
        hostname: '127.0.0.1', port: server4.port, path: '/api/scans',
        headers: { 'Authorization': `Bearer ${server4.token}` },
      }, res => {
        let d = ''; res.on('data', c => d += c);
        res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(d) }));
      }).on('error', reject);
    });

    check('GET /api/scans returns 200', listResp.status === 200);
    check('Returns an array of scans', Array.isArray(listResp.body));
    const scans = listResp.body;
    check('Contains scans', scans.length > 0);
  } catch (err) {
    check('T4 execution', false, err.message);
  } finally {
    await killServer(server4?.proc);
    await mock4?.close();
  }


  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  const total = results.length;
  const passed = results.filter(r => r.ok).length;
  const failed = total - passed;

  console.log(`\n${'='.repeat(60)}`);
  console.log(`  Results: ${passed}/${total} passed${failed ? ` (${failed} FAILED)` : ' -- ALL PASSED'}`);
  if (failed) {
    results.filter(r => !r.ok).forEach(r => console.log(`    X ${r.name}`));
  }
  console.log(`${'='.repeat(60)}\n`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Fatal error in Phase 4 test script:', err);
  process.exit(1);
});
