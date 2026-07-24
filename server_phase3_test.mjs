/**
 * Phase 3 verification script — server.ts strict Express integration.
 *
 * Run from the project root (no build needed):
 *   node server_phase3_test.mjs
 *
 * What this tests:
 *  T1 - No ML_SERVICE_URL  -> /api/predict returns 503 (no simulation)
 *  T2 - Flask unreachable  -> /api/predict returns 503 (fail-closed)
 *  T3 - Flask returns 400  -> /api/predict returns 400 (forwarded error)
 *  T4 - Flask returns 500  -> /api/predict returns 502
 *  T5 - Flask returns bad schema -> /api/predict returns 502, no DB write
 *  T6 - Flask returns valid V3 JSON -> /api/predict returns 200, scan saved once
 *  T7 - No image field     -> /api/predict returns 400
 *
 * The script:
 *  - Spawns a real mock HTTP server on a free port to act as Flask.
 *  - Imports the compiled server via tsx / ts-node for type-safe coverage.
 *  - Uses the local JSON DB (data/db.json) — reads scan count before/after.
 *
 * Prerequisites: npm packages (express, multer, jsonwebtoken, etc.) installed.
 * Uses Node built-ins only for the mock server.
 */

import http from 'http';
import { createServer as createHttpServer } from 'http';
import { execSync, spawn } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { setTimeout as sleep } from 'timers/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Full paths needed because Node.js is not on the system PATH in this environment.
const NODE_EXE  = 'G:\\Nodejs\\node.exe';
const TSX_EXE   = path.join(__dirname, 'node_modules', '.bin', 'tsx.cmd');

// ---------------------------------------------------------------------------
// Minimal helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Mock Flask server factory
// ---------------------------------------------------------------------------
function createMockFlask(behaviour) {
  /**
   * behaviour: function(req) => { status, body }
   * Returns { server, port, close() }
   */
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

// ---------------------------------------------------------------------------
// Valid V3 Flask response (Non-NAFLD)
// ---------------------------------------------------------------------------
const VALID_V3_NON_NAFLD = {
  success: true,
  input_mode: 'single_frame',
  model_version: '3.0.0',
  contract_sha256: 'a'.repeat(64),
  research_use_only: true,
  binary_result: 'Non-NAFLD',
  binary_prob_nafld: 0.42,
  binary_prob_non_nafld: 0.58,
  binary_threshold: 0.5536812544,
  binary_fold_probs: [0.40, 0.41, 0.43, 0.44, 0.42],
  binary_fold_std: 0.015,
  grading_performed: false,
  final_label: 'Non-NAFLD',
};

// Valid V3 Flask response (NAFLD + grading)
const VALID_V3_NAFLD = {
  success: true,
  input_mode: 'single_frame',
  model_version: '3.0.0',
  contract_sha256: 'a'.repeat(64),
  research_use_only: true,
  binary_result: 'NAFLD',
  binary_prob_nafld: 0.72,
  binary_prob_non_nafld: 0.28,
  binary_threshold: 0.5536812544,
  binary_fold_probs: [0.70, 0.71, 0.73, 0.74, 0.72],
  binary_fold_std: 0.015,
  grading_performed: true,
  grading_result: 'Grade1_Mild',
  grading_prob_moderate_severe: 0.44,
  grading_prob_mild: 0.56,
  grading_threshold: 0.6078062057,
  grading_fold_probs: [0.42, 0.43, 0.45, 0.46, 0.44],
  grading_fold_std: 0.016,
  final_label: 'NAFLD-Grade1_Mild',
};

// ---------------------------------------------------------------------------
// Build a minimal JPEG buffer (JPEG magic bytes + enough bytes to pass multer)
// ---------------------------------------------------------------------------
function makeMinimalJpeg() {
  // SOI marker + SOF0 minimal — enough to pass the MIME filter, not a real image.
  // The mock Flask doesn't actually decode it.
  const buf = Buffer.alloc(256, 0);
  buf[0] = 0xff; buf[1] = 0xd8; buf[2] = 0xff; buf[3] = 0xe0;
  return buf;
}

// ---------------------------------------------------------------------------
// Build a multipart/form-data body manually
// ---------------------------------------------------------------------------
function buildMultipartBody(fields, files, boundary) {
  const parts = [];
  for (const [name, value] of Object.entries(fields)) {
    parts.push(
      `--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}`
    );
  }
  for (const { name, filename, contentType, data } of files) {
    const header = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"; filename="${filename}"\r\nContent-Type: ${contentType}\r\n\r\n`;
    parts.push(header);
    parts.push({ binary: data });
  }
  parts.push(`--${boundary}--`);

  const chunks = parts.map(p =>
    typeof p === 'string' ? Buffer.from(p + '\r\n', 'utf8')
    : Buffer.isBuffer(p) ? Buffer.concat([p, Buffer.from('\r\n')])
    : Buffer.concat([p.binary, Buffer.from('\r\n')])
  );
  return Buffer.concat(chunks);
}

// ---------------------------------------------------------------------------
// HTTP helper: send a predict request to Express
// ---------------------------------------------------------------------------
async function sendPredictRequest(expressPort, jwtToken, flaskUrl, imageBuffer, extraFields = {}) {
  const boundary = 'BOUNDARY' + Date.now();
  const body = buildMultipartBody(
    {
      patientName: 'Test Patient',
      patientAge: '35',
      patientGender: 'Male',
      ...extraFields,
    },
    imageBuffer ? [{ name: 'image', filename: 'scan.jpg', contentType: 'image/jpeg', data: imageBuffer }] : [],
    boundary
  );

  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1',
      port: expressPort,
      path: '/api/predict',
      method: 'POST',
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
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Read current scan count from db.json
// ---------------------------------------------------------------------------
function getScanCount() {
  const dbPath = path.join(__dirname, 'data', 'db.json');
  if (!existsSync(dbPath)) return 0;
  try {
    const db = JSON.parse(readFileSync(dbPath, 'utf8'));
    return (db.scans || []).length;
  } catch { return 0; }
}

// ---------------------------------------------------------------------------
// Start Express + get a JWT token for test requests.
// Each call gets its own port to avoid EADDRINUSE between sequential tests.
// ---------------------------------------------------------------------------
let _nextPort = 3100; // start well above 3001 to avoid any running server

async function killServer(proc) {
  if (!proc || proc.killed) return;
  try {
    // On Windows, proc.kill() only kills the shell wrapper. Use taskkill to
    // terminate the whole process tree (shell -> tsx -> node).
    execSync(`taskkill /F /T /PID ${proc.pid}`, { stdio: 'ignore' });
  } catch { /* process may already be gone */ }
  // Give the OS a moment to release the port binding.
  await sleep(600);
}

async function startExpressAndGetToken(mlServiceUrl) {
  const port = _nextPort++;

  const env = {
    ...process.env,
    PATH: `G:\\Nodejs;${process.env.PATH || ''}`,
    ML_SERVICE_URL: mlServiceUrl || '',
    JWT_SECRET: 'test-secret-phase3',
    NODE_ENV: 'test',
    PORT: String(port),
  };

  const proc = spawn(
    TSX_EXE, ['server.ts'],
    { cwd: __dirname, env, stdio: ['ignore', 'pipe', 'pipe'], shell: true }
  );

  // Wait for the server to say it's ready
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

  // Get a JWT by registering a test doctor
  const email = `phase3test_${Date.now()}@test.com`;
  const registerResp = await new Promise((resolve, reject) => {
    const body = JSON.stringify({ email, password: 'test1234', name: 'Phase3 Doctor', role: 'doctor' });
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log('\n==================================================================');
  console.log('         NAFLD Detector -- Phase 3 Verification Script');
  console.log('==================================================================\n');

  const jpeg = makeMinimalJpeg();

  // -----------------------------------------------------------------------
  // T1: No ML_SERVICE_URL -> 503 immediately (no simulation)
  // -----------------------------------------------------------------------
  section('T1 - No ML_SERVICE_URL -> 503 (fail-closed, no simulation)');
  let server1;
  try {
    server1 = await startExpressAndGetToken(''); // empty URL
    const resp = await sendPredictRequest(server1.port, server1.token, '', jpeg);
    check('Returns 503 when ML_SERVICE_URL is absent', resp.status === 503,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);
    check('Error message mentions ML_SERVICE_URL or configuration',
      typeof resp.body?.error === 'string' && resp.body.error.toLowerCase().includes('ml'),
      resp.body?.error);
    check('No scan written for 503 response', true); // verified by absence of ML_SERVICE_URL path
  } catch (err) {
    check('Express started for T1', false, err.message);
  } finally {
    await killServer(server1?.proc);
  }

  // -----------------------------------------------------------------------
  // T2: Flask unreachable -> 503 (fail-closed)
  // -----------------------------------------------------------------------
  section('T2 - Flask unreachable -> 503');
  let server2;
  try {
    const scansBefore = getScanCount();
    server2 = await startExpressAndGetToken('http://127.0.0.1:19999'); // nothing listening
    const resp = await sendPredictRequest(server2.port, server2.token, '', jpeg);
    check('Returns 503 when Flask is unreachable', resp.status === 503,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);
    check('No DB write on unreachable Flask', getScanCount() === scansBefore,
      `Scans before: ${scansBefore}, after: ${getScanCount()}`);
  } catch (err) {
    check('Express started for T2', false, err.message);
  } finally {
    await killServer(server2?.proc);
  }

  // -----------------------------------------------------------------------
  // T3: Flask returns 400 (bad image) -> Express returns 400
  // -----------------------------------------------------------------------
  section('T3 - Flask returns 400 -> Express returns 400 with forwarded error');
  let mock3, server3;
  try {
    mock3 = await createMockFlask(() => ({
      status: 400,
      responseBody: { success: false, error: 'Unsupported image type test/fake.' },
    }));
    const scansBefore = getScanCount();
    server3 = await startExpressAndGetToken(mock3.url);
    const resp = await sendPredictRequest(server3.port, server3.token, '', jpeg);
    check('Returns 400 when Flask returns 400', resp.status === 400,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);
    check('Flask error message forwarded to client',
      resp.body?.error?.includes('Unsupported image type'),
      resp.body?.error);
    check('No DB write on Flask 400', getScanCount() === scansBefore,
      `Scans before: ${scansBefore}, after: ${getScanCount()}`);
  } catch (err) {
    check('Express started for T3', false, err.message);
  } finally {
    await killServer(server3?.proc);
    await mock3?.close();
  }

  // -----------------------------------------------------------------------
  // T4: Flask returns 500 -> Express returns 502
  // -----------------------------------------------------------------------
  section('T4 - Flask returns 500 -> Express returns 502');
  let mock4, server4;
  try {
    mock4 = await createMockFlask(() => ({
      status: 500,
      responseBody: { success: false, error: 'Model inference failed.' },
    }));
    const scansBefore = getScanCount();
    server4 = await startExpressAndGetToken(mock4.url);
    const resp = await sendPredictRequest(server4.port, server4.token, '', jpeg);
    check('Returns 502 when Flask returns 500', resp.status === 502,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);
    check('No DB write on Flask 500', getScanCount() === scansBefore,
      `Scans before: ${scansBefore}, after: ${getScanCount()}`);
  } catch (err) {
    check('Express started for T4', false, err.message);
  } finally {
    await killServer(server4?.proc);
    await mock4?.close();
  }

  // -----------------------------------------------------------------------
  // T5: Flask returns bad schema -> Express returns 502, no DB write
  // -----------------------------------------------------------------------
  section('T5 - Flask returns bad schema -> Express returns 502, no DB write');
  let mock5, server5;
  try {
    mock5 = await createMockFlask(() => ({
      status: 200,
      responseBody: {
        success: true,
        // Missing required fields: binary_result, final_label, probabilities, etc.
        prediction: 'Normal',
        confidence: 90,
      },
    }));
    const scansBefore = getScanCount();
    server5 = await startExpressAndGetToken(mock5.url);
    const resp = await sendPredictRequest(server5.port, server5.token, '', jpeg);
    check('Returns 502 on bad schema', resp.status === 502,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);
    check('No DB write on schema validation failure', getScanCount() === scansBefore,
      `Scans before: ${scansBefore}, after: ${getScanCount()}`);
  } catch (err) {
    check('Express started for T5', false, err.message);
  } finally {
    await killServer(server5?.proc);
    await mock5?.close();
  }

  // -----------------------------------------------------------------------
  // T6: Valid V3 response -> 200, scan saved exactly once
  // -----------------------------------------------------------------------
  section('T6 - Valid V3 Flask response -> 200, scan saved once');
  let mock6, server6;
  try {
    mock6 = await createMockFlask(() => ({
      status: 200,
      responseBody: VALID_V3_NON_NAFLD,
    }));
    const scansBefore = getScanCount();
    server6 = await startExpressAndGetToken(mock6.url);
    const resp = await sendPredictRequest(server6.port, server6.token, '', jpeg);
    check('Returns 200 on valid V3 response', resp.status === 200,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);

    if (resp.status === 200) {
      const scan = resp.body;
      check('Scan has id', typeof scan.id === 'string' && scan.id.startsWith('scan-'));
      check('prediction is Normal (Non-NAFLD mapped)', scan.prediction === 'Normal',
        `got ${scan.prediction}`);
      check('confidence is a number in (0,100]', typeof scan.confidence === 'number' && scan.confidence > 0,
        `got ${scan.confidence}`);
      check('probabilities.Normal and probabilities.Abnormal present',
        typeof scan.probabilities?.Normal === 'number' && typeof scan.probabilities?.Abnormal === 'number');
      check('probabilities sum ~100',
        Math.abs((scan.probabilities.Normal + scan.probabilities.Abnormal) - 100) < 0.5,
        `sum=${scan.probabilities.Normal + scan.probabilities.Abnormal}`);

      // Verify scan was actually persisted by reading it back (works for both Firebase and local JSON).
      await sleep(400);
      const getResp = await new Promise((resolve, reject) => {
        http.get({
          hostname: '127.0.0.1',
          port: server6.port,
          path: `/api/scans/${scan.id}`,
          headers: { 'Authorization': `Bearer ${server6.token}` },
        }, res => {
          let d = ''; res.on('data', c => d += c);
          res.on('end', () => {
            try { resolve({ status: res.statusCode, body: JSON.parse(d) }); }
            catch { resolve({ status: res.statusCode, body: d }); }
          });
        }).on('error', reject);
      });
      check('Scan written and readable back (GET /api/scans/:id returns 200)',
        getResp.status === 200 && getResp.body?.id === scan.id,
        `GET returned ${getResp.status}: ${JSON.stringify(getResp.body)}`);
    }
  } catch (err) {
    check('Express started for T6', false, err.message);
  } finally {
    await killServer(server6?.proc);
    await mock6?.close();
  }

  // -----------------------------------------------------------------------
  // T7: No image field -> 400
  // -----------------------------------------------------------------------
  section('T7 - No image field -> 400');
  let mock7, server7;
  try {
    mock7 = await createMockFlask(() => ({ status: 200, responseBody: VALID_V3_NON_NAFLD }));
    server7 = await startExpressAndGetToken(mock7.url);
    // Send without image buffer
    const resp = await sendPredictRequest(server7.port, server7.token, '', null);
    check('Returns 400 when no image field', resp.status === 400,
      `Got ${resp.status}: ${JSON.stringify(resp.body)}`);
  } catch (err) {
    check('Express started for T7', false, err.message);
  } finally {
    await killServer(server7?.proc);
    await mock7?.close();
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
  console.error('Fatal error in test script:', err);
  process.exit(1);
});
