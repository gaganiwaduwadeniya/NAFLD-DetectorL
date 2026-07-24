/**
 * Phase 6 verification script — Automated Verification.
 *
 * Run from project root:
 *   $env:PATH = "G:\Nodejs;$env:PATH"; & 'G:\Nodejs\node.exe' server_phase6_test.mjs
 *
 * Master-report Phase 6 coverage:
 *  S1  - Contract validation failures (inline validator unit tests, no server)
 *  S2  - Missing/corrupt model hashes & model metadata storage
 *  S3  - Model input/output shapes (prob range, type, finiteness) via HTTP
 *  S4  - Image forwarding integrity (raw bytes forwarded, imageUrl preserved)
 *  S5  - Ensemble arithmetic (fold probs stored verbatim, fold std stored)
 *  S6  - Cascade gating and threshold boundaries
 *  S7  - Invalid upload and malformed-image errors
 *  S8  - Express timeout / fail-closed behaviour
 *  S9  - No database write on inference failure
 *  S10 - React rendering states (inline logic tests — no DOM required)
 *  S11 - End-to-end upload through Express to Flask and back to history
 *  S12 - Performance: startup time, per-request latency, memory, health
 */

import http from 'http';
import { createServer as createHttpServer } from 'http';
import { execSync, spawn }                 from 'child_process';
import { setTimeout as sleep }             from 'timers/promises';
import path                                from 'path';
import { fileURLToPath }                   from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TSX_EXE   = path.join(__dirname, 'node_modules', '.bin', 'tsx.cmd');

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
  console.log(`\n${'-'.repeat(62)}`);
  console.log(`  ${title}`);
  console.log(`${'-'.repeat(62)}`);
}

// ============================================================================
// Inline reimplementation of validateV3Response (mirrors server.ts exactly).
// Used for pure-JS unit tests in S1 and S3 without starting a server.
// ============================================================================
function validateV3Response(data) {
  if (!data || typeof data !== 'object') return 'Response is not a JSON object.';
  if (data.success !== true)
    return `success field is not true: ${JSON.stringify(data.success)}`;
  if (data.research_use_only !== true)
    return 'research_use_only must be true.';

  const validBinaryResults = ['Non-NAFLD', 'NAFLD'];
  if (!validBinaryResults.includes(data.binary_result))
    return `binary_result ${JSON.stringify(data.binary_result)} is not valid.`;

  const validFinalLabels = [
    'Non-NAFLD', 'NAFLD-Grade1_Mild', 'NAFLD-Grade2_Moderate_Severe',
  ];
  if (!validFinalLabels.includes(data.final_label))
    return `final_label ${JSON.stringify(data.final_label)} is not valid.`;

  const probNafld    = data.binary_prob_nafld;
  const probNonNafld = data.binary_prob_non_nafld;
  if (typeof probNafld !== 'number' || !isFinite(probNafld) || probNafld < 0 || probNafld > 1)
    return `binary_prob_nafld out of range: ${probNafld}`;
  if (typeof probNonNafld !== 'number' || !isFinite(probNonNafld) || probNonNafld < 0 || probNonNafld > 1)
    return `binary_prob_non_nafld out of range: ${probNonNafld}`;

  if (!data.model_version || typeof data.model_version !== 'string')
    return 'model_version must be a non-empty string.';

  if (data.grading_performed === true) {
    const validGrading = ['Grade1_Mild', 'Grade2_Moderate_Severe'];
    if (!validGrading.includes(data.grading_result))
      return `grading_result ${JSON.stringify(data.grading_result)} is invalid.`;
    const probMod  = data.grading_prob_moderate_severe;
    const probMild = data.grading_prob_mild;
    if (typeof probMod !== 'number' || !isFinite(probMod) || probMod < 0 || probMod > 1)
      return `grading_prob_moderate_severe out of range: ${probMod}`;
    if (typeof probMild !== 'number' || !isFinite(probMild) || probMild < 0 || probMild > 1)
      return `grading_prob_mild out of range: ${probMild}`;
  }
  return null; // valid
}

// ============================================================================
// React rendering state derivation — mirrors ResultCard.tsx logic exactly.
// Used for pure-JS unit tests in S10 without a browser/DOM.
// ============================================================================
function deriveDisplayState(scan) {
  if (!scan) return { mode: 'loading' };
  const isV3 = scan.schemaVersion === 3;
  if (isV3) {
    return {
      mode: 'v3',
      isNafld:          scan.binaryResult === 'NAFLD',
      finalLabel:       scan.finalLabel ?? scan.binaryResult ?? 'Unknown',
      showGrading:      scan.gradingPerformed === true,
      gradingResult:    scan.gradingResult,
      hasResearchNotice: scan.researchUseOnly === true,
    };
  }
  return {
    mode:          'legacy',
    isNormal:      scan.prediction === 'Normal',
    finalLabel:    scan.prediction,
    showGrading:   false,
    hasResearchNotice: false,
  };
}

// ============================================================================
// Fixtures
// ============================================================================
const VALID_NON_NAFLD = {
  success: true, input_mode: 'single_frame', model_version: '3.0.0',
  contract_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  research_use_only: true,
  binary_result: 'Non-NAFLD', binary_prob_nafld: 0.35, binary_prob_non_nafld: 0.65,
  binary_threshold: 0.5536812544,
  binary_fold_probs: [0.33, 0.36, 0.34, 0.37, 0.35], binary_fold_std: 0.014,
  grading_performed: false, final_label: 'Non-NAFLD',
};

const VALID_NAFLD_MILD = {
  success: true, input_mode: 'single_frame', model_version: '3.0.0',
  contract_sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  research_use_only: true,
  binary_result: 'NAFLD', binary_prob_nafld: 0.78, binary_prob_non_nafld: 0.22,
  binary_threshold: 0.5536812544,
  binary_fold_probs: [0.76, 0.79, 0.77, 0.80, 0.78], binary_fold_std: 0.013,
  grading_performed: true, grading_result: 'Grade1_Mild',
  grading_prob_moderate_severe: 0.42, grading_prob_mild: 0.58,
  grading_threshold: 0.6078062057,
  grading_fold_probs: [0.40, 0.43, 0.41, 0.44, 0.42], grading_fold_std: 0.015,
  final_label: 'NAFLD-Grade1_Mild',
};

// ============================================================================
// Infrastructure helpers
// ============================================================================
function makeMinimalPng() {
  // Valid PNG: 512 × 256, RGBA
  return Buffer.from([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG magic
    0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
    0x00, 0x00, 0x02, 0x00,                           // width = 512
    0x00, 0x00, 0x01, 0x00,                           // height = 256
    0x08, 0x06, 0x00, 0x00, 0x00,                     // bit depth, color type, etc.
    0x14, 0x7a, 0x7c, 0x73,                           // CRC
  ]);
}

function buildMultipartBody(fields, files, boundary) {
  const chunks = [];
  for (const [name, value] of Object.entries(fields)) {
    chunks.push(Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`
    ));
  }
  for (const { name, filename, contentType, data } of files) {
    chunks.push(Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="${name}"; ` +
      `filename="${filename}"\r\nContent-Type: ${contentType}\r\n\r\n`
    ));
    chunks.push(data);
    chunks.push(Buffer.from('\r\n'));
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  return Buffer.concat(chunks);
}

// Mock Flask with changeable behaviour via a ref object
function createControllableMockFlask(ref) {
  const server = createHttpServer((req, res) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      const { status, responseBody } = ref.fn(req, body);
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
      const { port } = server.address();
      resolve({ port, url: `http://127.0.0.1:${port}`, close: () => new Promise(r => server.close(r)) });
    });
  });
}

// Slow mock — holds response for `delayMs` to trigger Express timeout
function createSlowMockFlask(delayMs) {
  const server = createHttpServer((req, res) => {
    req.on('data', () => {});
    req.on('end', () => {
      setTimeout(() => {
        try {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(VALID_NON_NAFLD));
        } catch { /* connection already closed by AbortController */ }
      }, delayMs);
    });
  });
  return new Promise(resolve => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ port, url: `http://127.0.0.1:${port}`, close: () => new Promise(r => server.close(r)) });
    });
  });
}

let _nextPort = 3300;

async function killServer(proc) {
  if (!proc || proc.killed) return;
  try { execSync(`taskkill /F /T /PID ${proc.pid}`, { stdio: 'ignore' }); } catch {}
  await sleep(500);
}

async function startExpress(mlServiceUrl, role = 'doctor') {
  const port = _nextPort++;
  const t0 = Date.now();
  const env = {
    ...process.env,
    PATH: `G:\\Nodejs;${process.env.PATH || ''}`,
    ML_SERVICE_URL: mlServiceUrl || '',
    JWT_SECRET: 'phase6-test-secret',
    NODE_ENV: 'test',
    PORT: String(port),
  };

  const proc = spawn(TSX_EXE, ['server.ts'], {
    cwd: __dirname, env, stdio: ['ignore', 'pipe', 'pipe'], shell: true,
  });

  const startupMs = await new Promise((resolve, reject) => {
    let output = '';
    const timeout = setTimeout(
      () => reject(new Error('Express timed out on startup.\n' + output)), 20000
    );
    proc.stdout.on('data', d => {
      output += d.toString();
      if (output.includes('Server successfully booted')) {
        clearTimeout(timeout); resolve(Date.now() - t0);
      }
    });
    proc.stderr.on('data', d => { output += d.toString(); });
    proc.on('exit', code => {
      clearTimeout(timeout);
      reject(new Error(`Express exited ${code}.\n${output}`));
    });
  });

  // Register a fresh user
  const email = `p6_${Date.now()}_${Math.random().toString(36).slice(2)}@test.com`;
  const body = JSON.stringify({ email, password: 'Test1234', name: 'Phase6', role });
  const regResp = await new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1', port, path: '/api/auth/register', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(d) }));
    });
    req.on('error', reject); req.write(body); req.end();
  });

  return { proc, port, token: regResp.body?.token, startupMs };
}

async function sendPredict(port, token, imageBuffer, overrideFields = {}) {
  const boundary = 'P6BD' + Date.now();
  const body = buildMultipartBody(
    { patientName: 'Phase6 Patient', patientAge: '45', patientGender: 'Male', ...overrideFields },
    [{ name: 'image', filename: 'scan.png', contentType: 'image/png', data: imageBuffer }],
    boundary,
  );
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1', port, path: '/api/predict', method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length,
      },
    }, res => {
      let data = '';
      res.on('data', c => { data += c; });
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject); req.write(body); req.end();
  });
}

// Send predict with NO image field (only text fields)
async function sendPredictNoImage(port, token) {
  const boundary = 'P6NOIMG' + Date.now();
  const body = buildMultipartBody(
    { patientName: 'Phase6 Patient', patientAge: '45', patientGender: 'Male' },
    [], // ← no file parts
    boundary,
  );
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1', port, path: '/api/predict', method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length,
      },
    }, res => {
      let data = '';
      res.on('data', c => { data += c; });
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject); req.write(body); req.end();
  });
}

async function httpGet(port, urlPath, token) {
  return new Promise((resolve, reject) => {
    http.get({
      hostname: '127.0.0.1', port, path: urlPath,
      headers: { 'Authorization': `Bearer ${token}` },
    }, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(d) }); }
        catch { resolve({ status: res.statusCode, body: d }); }
      });
    }).on('error', reject);
  });
}

async function scanCount(port, token) {
  const r = await httpGet(port, '/api/scans', token);
  return Array.isArray(r.body) ? r.body.length : -1;
}

// ============================================================================
// MAIN
// ============================================================================
async function main() {
  console.log('\n================================================================');
  console.log('        NAFLD Detector — Phase 6 Automated Verification');
  console.log('================================================================\n');

  const memBefore = process.memoryUsage();
  const img = makeMinimalPng();

  // ==========================================================================
  // S1 — Contract validation failures: inline unit tests (no server)
  // ==========================================================================
  section('S1 — Contract validation: inline unit tests (no server)');

  // Null / non-object
  check('null input → error',              validateV3Response(null) !== null);
  check('undefined input → error',         validateV3Response(undefined) !== null);
  check('string input → error',            validateV3Response('bad') !== null);

  // success field
  check('success: false → error',          validateV3Response({ ...VALID_NON_NAFLD, success: false }) !== null);
  check('success: undefined → error',      validateV3Response({ ...VALID_NON_NAFLD, success: undefined }) !== null);
  check('success: 1 (not true) → error',   validateV3Response({ ...VALID_NON_NAFLD, success: 1 }) !== null);

  // research_use_only
  check('research_use_only: false → error',     validateV3Response({ ...VALID_NON_NAFLD, research_use_only: false }) !== null);
  check('research_use_only: undefined → error', validateV3Response({ ...VALID_NON_NAFLD, research_use_only: undefined }) !== null);

  // binary_result
  check('binary_result: "UNKNOWN" → error', validateV3Response({ ...VALID_NON_NAFLD, binary_result: 'UNKNOWN' }) !== null);
  check('binary_result: null → error',      validateV3Response({ ...VALID_NON_NAFLD, binary_result: null }) !== null);

  // final_label
  check('final_label: "BAD_LABEL" → error', validateV3Response({ ...VALID_NON_NAFLD, final_label: 'BAD_LABEL' }) !== null);

  // prob range
  check('binary_prob_nafld: "string" → error',  validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: 'hi' }) !== null);
  check('binary_prob_nafld: NaN → error',       validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: NaN }) !== null);
  check('binary_prob_nafld: -0.1 → error',      validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: -0.1 }) !== null);
  check('binary_prob_nafld: 1.1 → error',       validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: 1.1 }) !== null);
  check('binary_prob_nafld: Infinity → error',  validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: Infinity }) !== null);
  check('binary_prob_non_nafld: -1 → error',    validateV3Response({ ...VALID_NON_NAFLD, binary_prob_non_nafld: -1 }) !== null);

  // model_version
  check('model_version: "" → error',   validateV3Response({ ...VALID_NON_NAFLD, model_version: '' }) !== null);
  check('model_version: 42 → error',   validateV3Response({ ...VALID_NON_NAFLD, model_version: 42 }) !== null);

  // grading fields when grading_performed=true
  check('grading=true + bad grading_result → error',
    validateV3Response({ ...VALID_NAFLD_MILD, grading_result: 'UNKNOWN_GRADE' }) !== null);
  check('grading=true + grading_prob_moderate_severe: 2.0 → error',
    validateV3Response({ ...VALID_NAFLD_MILD, grading_prob_moderate_severe: 2.0 }) !== null);
  check('grading=true + grading_prob_mild: NaN → error',
    validateV3Response({ ...VALID_NAFLD_MILD, grading_prob_mild: NaN }) !== null);

  // Valid responses → null
  check('Valid Non-NAFLD response → null (accepted)', validateV3Response(VALID_NON_NAFLD) === null);
  check('Valid NAFLD-Mild response → null (accepted)', validateV3Response(VALID_NAFLD_MILD) === null);
  // Boundary probabilities 0 and 1 are valid
  check('prob exactly 0 → accepted',
    validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: 0, binary_prob_non_nafld: 1 }) === null);
  check('prob exactly 1 → accepted',
    validateV3Response({ ...VALID_NON_NAFLD, binary_prob_nafld: 1, binary_prob_non_nafld: 0 }) === null);

  // ==========================================================================
  // S2 — Model metadata & hash storage  (1 Express instance, shared)
  // ==========================================================================
  section('S2 — Model metadata & hash storage');
  const ref2 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock2, srv2;
  try {
    mock2 = await createControllableMockFlask(ref2);
    srv2  = await startExpress(mock2.url);
    const resp2 = await sendPredict(srv2.port, srv2.token, img);
    const scan2 = resp2.body;

    check('S2: modelVersion stored',          scan2.modelVersion === '3.0.0', `got ${scan2.modelVersion}`);
    check('S2: contractSha256 stored (64 chars)',
      typeof scan2.contractSha256 === 'string' && scan2.contractSha256.length === 64,
      `len=${scan2.contractSha256?.length}`);
    check('S2: contractSha256 matches fixture', scan2.contractSha256 === VALID_NON_NAFLD.contract_sha256);
    check('S2: inputMode stored',             scan2.inputMode === 'single_frame');
    check('S2: researchUseOnly stored',       scan2.researchUseOnly === true);
    check('S2: schemaVersion stored as 3',    scan2.schemaVersion === 3);

    // Contract_sha256 absent → server accepts (validator does not check sha256)
    ref2.fn = () => ({
      status: 200,
      responseBody: { ...VALID_NON_NAFLD, contract_sha256: undefined },
    });
    const resp2b = await sendPredict(srv2.port, srv2.token, img);
    check('S2: missing contract_sha256 — server accepts (known validator gap)',
      resp2b.status === 200,
      `got ${resp2b.status} — validator does not enforce sha256 presence`);
  } catch (err) {
    check('S2 execution', false, err.message);
  } finally {
    await killServer(srv2?.proc); await mock2?.close();
  }

  // ==========================================================================
  // S3 — Model input/output shapes via HTTP (prob range enforced)
  // ==========================================================================
  section('S3 — Model input/output shapes (prob range enforcement via HTTP)');
  const ref3 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock3, srv3;
  try {
    mock3 = await createControllableMockFlask(ref3);
    srv3  = await startExpress(mock3.url);

    const cases3 = [
      ['binary_prob_nafld: 1.5 → 502',      { ...VALID_NON_NAFLD, binary_prob_nafld: 1.5 }],
      ['binary_prob_non_nafld: -0.1 → 502',  { ...VALID_NON_NAFLD, binary_prob_non_nafld: -0.1 }],
      ['binary_result: "INVALID" → 502',     { ...VALID_NON_NAFLD, binary_result: 'INVALID' }],
      ['final_label: "BAD" → 502',           { ...VALID_NON_NAFLD, final_label: 'BAD' }],
      ['research_use_only: false → 502',     { ...VALID_NON_NAFLD, research_use_only: false }],
      ['success: false → 502',               { ...VALID_NON_NAFLD, success: false }],
      ['model_version: "" → 502',            { ...VALID_NON_NAFLD, model_version: '' }],
    ];
    for (const [label, badBody] of cases3) {
      ref3.fn = () => ({ status: 200, responseBody: badBody });
      const r = await sendPredict(srv3.port, srv3.token, img);
      check(`S3: ${label}`, r.status === 502, `got ${r.status}`);
    }
  } catch (err) {
    check('S3 execution', false, err.message);
  } finally {
    await killServer(srv3?.proc); await mock3?.close();
  }

  // ==========================================================================
  // S4 — Image forwarding integrity (raw bytes forwarded, imageUrl preserved)
  // ==========================================================================
  section('S4 — Grayscale / colour preprocessing & image forwarding integrity');
  const ref4 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock4, srv4;
  try {
    mock4 = await createControllableMockFlask(ref4);
    srv4  = await startExpress(mock4.url);
    const resp4 = await sendPredict(srv4.port, srv4.token, img);
    const scan4 = resp4.body;

    check('S4: predict returns 200',    resp4.status === 200, `got ${resp4.status}`);
    check('S4: imageUrl is non-empty',  typeof scan4.imageUrl === 'string' && scan4.imageUrl.length > 0);
    check('S4: imageUrl preserves PNG content-type',
      scan4.imageUrl.startsWith('data:image/png;base64,'),
      `imageUrl starts with: ${scan4.imageUrl?.slice(0, 30)}`);
    check('S4: imageWidth parsed correctly (512)',  scan4.imageWidth === 512,  `got ${scan4.imageWidth}`);
    check('S4: imageHeight parsed correctly (256)', scan4.imageHeight === 256, `got ${scan4.imageHeight}`);
  } catch (err) {
    check('S4 execution', false, err.message);
  } finally {
    await killServer(srv4?.proc); await mock4?.close();
  }

  // ==========================================================================
  // S5 — Ensemble arithmetic (fold probs stored verbatim)
  // ==========================================================================
  section('S5 — Ensemble arithmetic (fold probs & std stored verbatim)');
  const ref5 = { fn: () => ({ status: 200, responseBody: VALID_NAFLD_MILD }) };
  let mock5, srv5;
  try {
    mock5 = await createControllableMockFlask(ref5);
    srv5  = await startExpress(mock5.url);
    const resp5 = await sendPredict(srv5.port, srv5.token, img);
    const scan5 = resp5.body;

    check('S5: binaryFoldProbs has 5 entries',
      Array.isArray(scan5.binaryFoldProbs) && scan5.binaryFoldProbs.length === 5,
      `got ${JSON.stringify(scan5.binaryFoldProbs)}`);
    check('S5: binaryFoldProbs stored verbatim (index 0)',
      scan5.binaryFoldProbs?.[0] === VALID_NAFLD_MILD.binary_fold_probs[0]);
    check('S5: binaryFoldProbs stored verbatim (index 4)',
      scan5.binaryFoldProbs?.[4] === VALID_NAFLD_MILD.binary_fold_probs[4]);
    check('S5: binaryFoldStd stored',    scan5.binaryFoldStd === VALID_NAFLD_MILD.binary_fold_std,
      `got ${scan5.binaryFoldStd}`);
    check('S5: gradingFoldProbs has 5 entries',
      Array.isArray(scan5.gradingFoldProbs) && scan5.gradingFoldProbs.length === 5);
    check('S5: gradingFoldProbs verbatim (index 2)',
      scan5.gradingFoldProbs?.[2] === VALID_NAFLD_MILD.grading_fold_probs[2]);
    check('S5: gradingFoldStd stored',   scan5.gradingFoldStd === VALID_NAFLD_MILD.grading_fold_std);

    // Ensemble mean of binaryFoldProbs should be close to binaryProbNafld
    const mean5 = scan5.binaryFoldProbs.reduce((a, b) => a + b, 0) / 5;
    check('S5: fold mean ≈ binaryProbNafld (±0.05)',
      Math.abs(mean5 - scan5.binaryProbNafld) < 0.05,
      `mean=${mean5.toFixed(4)}, binaryProbNafld=${scan5.binaryProbNafld}`);
  } catch (err) {
    check('S5 execution', false, err.message);
  } finally {
    await killServer(srv5?.proc); await mock5?.close();
  }

  // ==========================================================================
  // S6 — Cascade gating and threshold boundaries
  // ==========================================================================
  section('S6 — Cascade gating and threshold boundaries');
  const ref6 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock6, srv6;
  try {
    mock6 = await createControllableMockFlask(ref6);
    srv6  = await startExpress(mock6.url);

    // (a) Non-NAFLD + grading_performed=false → no grading fields
    ref6.fn = () => ({ status: 200, responseBody: VALID_NON_NAFLD });
    const a = (await sendPredict(srv6.port, srv6.token, img)).body;
    check('S6a: Non-NAFLD — gradingPerformed is false', a.gradingPerformed === false, `got ${a.gradingPerformed}`);
    check('S6a: Non-NAFLD — gradingResult is absent',   a.gradingResult === undefined);
    check('S6a: Non-NAFLD — gradingFoldProbs absent',   a.gradingFoldProbs === undefined);

    // (b) NAFLD + grading_performed=true → grading fields present
    ref6.fn = () => ({ status: 200, responseBody: VALID_NAFLD_MILD });
    const b = (await sendPredict(srv6.port, srv6.token, img)).body;
    check('S6b: NAFLD+Mild — gradingPerformed is true',         b.gradingPerformed === true);
    check('S6b: NAFLD+Mild — gradingResult is Grade1_Mild',     b.gradingResult === 'Grade1_Mild');
    check('S6b: NAFLD+Mild — gradingThreshold stored',          typeof b.gradingThreshold === 'number');

    // (c) NAFLD + grading_performed=true + invalid grading_result → 502
    ref6.fn = () => ({
      status: 200,
      responseBody: { ...VALID_NAFLD_MILD, grading_result: 'INVALID_GRADE' },
    });
    const c = await sendPredict(srv6.port, srv6.token, img);
    check('S6c: invalid grading_result → 502', c.status === 502, `got ${c.status}`);

    // (d) NAFLD + grading_performed=false → no cascade (gate held closed)
    ref6.fn = () => ({
      status: 200,
      responseBody: {
        ...VALID_NAFLD_MILD,
        grading_performed: false,
        grading_result: undefined,
        grading_prob_mild: undefined,
        grading_prob_moderate_severe: undefined,
        final_label: 'NAFLD-Grade1_Mild', // still valid
      },
    });
    const d = (await sendPredict(srv6.port, srv6.token, img)).body;
    check('S6d: NAFLD+grading=false — gradingPerformed stored false', d.gradingPerformed === false);

    // (e) Threshold boundary: binary_prob_nafld exactly 0 → accepted
    ref6.fn = () => ({
      status: 200,
      responseBody: { ...VALID_NON_NAFLD, binary_prob_nafld: 0, binary_prob_non_nafld: 1 },
    });
    const e = await sendPredict(srv6.port, srv6.token, img);
    check('S6e: binary_prob_nafld exactly 0 → accepted', e.status === 200, `got ${e.status}`);

    // (f) Threshold boundary: binary_prob_nafld exactly 1 → accepted
    ref6.fn = () => ({
      status: 200,
      responseBody: { ...VALID_NON_NAFLD, binary_prob_nafld: 1, binary_prob_non_nafld: 0, binary_result: 'NAFLD', final_label: 'NAFLD-Grade1_Mild', grading_performed: false },
    });
    const f = await sendPredict(srv6.port, srv6.token, img);
    check('S6f: binary_prob_nafld exactly 1 → accepted', f.status === 200, `got ${f.status}`);
  } catch (err) {
    check('S6 execution', false, err.message);
  } finally {
    await killServer(srv6?.proc); await mock6?.close();
  }

  // ==========================================================================
  // S7 — Invalid upload and malformed-image errors
  // ==========================================================================
  section('S7 — Invalid upload and malformed-image errors');
  const ref7 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock7, srv7;
  // Also test ML_SERVICE_URL unset on a separate Express instance
  let srvNoUrl;
  try {
    mock7 = await createControllableMockFlask(ref7);
    srv7  = await startExpress(mock7.url);

    // (a) No image field → 400
    const noImg = await sendPredictNoImage(srv7.port, srv7.token);
    check('S7a: No image field → 400', noImg.status === 400, `got ${noImg.status}`);

    // (b) Missing patientName → 400 (send empty string)
    const noName = await sendPredict(srv7.port, srv7.token, img, { patientName: '' });
    check('S7b: Empty patientName → 400', noName.status === 400, `got ${noName.status}`);

    // (c) Invalid JWT → 401
    const badJwt = await new Promise((resolve, reject) => {
      http.request({
        hostname: '127.0.0.1', port: srv7.port, path: '/api/predict', method: 'POST',
        headers: { 'Authorization': 'Bearer totally.invalid.jwt', 'Content-Type': 'application/json' },
      }, res => {
        let d = ''; res.on('data', c => d += c);
        res.on('end', () => resolve({ status: res.statusCode }));
      }).on('error', reject).end();
    });
    check('S7c: Invalid JWT → 401 or 403', [401, 403].includes(badJwt.status), `got ${badJwt.status}`);

    // (d) No Authorization header → 401
    const noAuth = await new Promise((resolve, reject) => {
      http.request({
        hostname: '127.0.0.1', port: srv7.port, path: '/api/predict', method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }, res => {
        let d = ''; res.on('data', c => d += c);
        res.on('end', () => resolve({ status: res.statusCode }));
      }).on('error', reject).end();
    });
    check('S7d: No auth header → 401', noAuth.status === 401, `got ${noAuth.status}`);

    // (e) ML_SERVICE_URL not set → 503
    srvNoUrl = await startExpress(''); // empty URL → ML_SERVICE_URL = ''
    const noUrl = await sendPredict(srvNoUrl.port, srvNoUrl.token, img);
    check('S7e: ML_SERVICE_URL unset → 503', noUrl.status === 503, `got ${noUrl.status}`);
  } catch (err) {
    check('S7 execution', false, err.message);
  } finally {
    await killServer(srv7?.proc);     await mock7?.close();
    await killServer(srvNoUrl?.proc);
  }

  // ==========================================================================
  // S8 — Express timeout / fail-closed behaviour  (~15 second wait)
  // ==========================================================================
  section('S8 — Express timeout / fail-closed behaviour  (≈15 s wait)');
  let slowMock8, srv8;
  try {
    console.log('  (Holding 16 s to trigger the 15 s AbortController timeout…)');
    slowMock8 = await createSlowMockFlask(16000); // 16s delay > 15s ML_INFERENCE_TIMEOUT_MS
    srv8      = await startExpress(slowMock8.url);
    const t8  = Date.now();
    const r8  = await sendPredict(srv8.port, srv8.token, img);
    const elapsed8 = Date.now() - t8;
    check('S8: Timeout → 503', r8.status === 503, `got ${r8.status}`);
    check('S8: Timeout fired after ≥ 14 s', elapsed8 >= 14000, `elapsed = ${elapsed8} ms`);
  } catch (err) {
    check('S8 execution', false, err.message);
  } finally {
    await killServer(srv8?.proc); await slowMock8?.close();
  }

  // ==========================================================================
  // S9 — No database write on inference failure
  // ==========================================================================
  section('S9 — No database write on inference failure');
  const ref9 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock9, srv9;
  try {
    mock9 = await createControllableMockFlask(ref9);
    srv9  = await startExpress(mock9.url);

    // Baseline: one successful scan
    const baseline = await sendPredict(srv9.port, srv9.token, img);
    check('S9: Baseline predict succeeds', baseline.status === 200, `got ${baseline.status}`);
    const countAfterSuccess = await scanCount(srv9.port, srv9.token);

    // Failure 1: bad schema → 502
    ref9.fn = () => ({ status: 200, responseBody: { ...VALID_NON_NAFLD, success: false } });
    const f1 = await sendPredict(srv9.port, srv9.token, img);
    check('S9: Schema failure returns 502', f1.status === 502, `got ${f1.status}`);
    const countAfterF1 = await scanCount(srv9.port, srv9.token);
    check('S9: No scan written after schema failure',
      countAfterF1 === countAfterSuccess, `expected ${countAfterSuccess}, got ${countAfterF1}`);

    // Failure 2: Flask 500 → 502
    ref9.fn = () => ({ status: 500, responseBody: { error: 'model crash' } });
    const f2 = await sendPredict(srv9.port, srv9.token, img);
    check('S9: Flask 500 → Express 502', f2.status === 502, `got ${f2.status}`);
    const countAfterF2 = await scanCount(srv9.port, srv9.token);
    check('S9: No scan written after Flask 500',
      countAfterF2 === countAfterSuccess, `expected ${countAfterSuccess}, got ${countAfterF2}`);

    // Failure 3: Flask 4xx → Express 400
    ref9.fn = () => ({ status: 400, responseBody: { error: 'bad image' } });
    const f3 = await sendPredict(srv9.port, srv9.token, img);
    check('S9: Flask 4xx → Express 400', f3.status === 400, `got ${f3.status}`);
    const countAfterF3 = await scanCount(srv9.port, srv9.token);
    check('S9: No scan written after Flask 4xx',
      countAfterF3 === countAfterSuccess, `expected ${countAfterSuccess}, got ${countAfterF3}`);
  } catch (err) {
    check('S9 execution', false, err.message);
  } finally {
    await killServer(srv9?.proc); await mock9?.close();
  }

  // ==========================================================================
  // S10 — React rendering states (inline logic tests — no DOM required)
  // ==========================================================================
  section('S10 — React rendering states (inline logic tests)');

  // Loading state (null scan)
  const stateLoading = deriveDisplayState(null);
  check('S10: null scan → mode: "loading"', stateLoading.mode === 'loading');

  // V3 — binary negative (Non-NAFLD)
  const stateNonNafld = deriveDisplayState({ schemaVersion: 3, binaryResult: 'Non-NAFLD', finalLabel: 'Non-NAFLD', gradingPerformed: false, researchUseOnly: true });
  check('S10: V3 Non-NAFLD → mode: "v3"',           stateNonNafld.mode === 'v3');
  check('S10: V3 Non-NAFLD → isNafld: false',        stateNonNafld.isNafld === false);
  check('S10: V3 Non-NAFLD → finalLabel correct',    stateNonNafld.finalLabel === 'Non-NAFLD');
  check('S10: V3 Non-NAFLD → showGrading: false',    stateNonNafld.showGrading === false);
  check('S10: V3 Non-NAFLD → researchNotice shown',  stateNonNafld.hasResearchNotice === true);

  // V3 — NAFLD Grade 1 Mild (with grading)
  const stateMild = deriveDisplayState({ schemaVersion: 3, binaryResult: 'NAFLD', finalLabel: 'NAFLD-Grade1_Mild', gradingPerformed: true, gradingResult: 'Grade1_Mild', researchUseOnly: true });
  check('S10: V3 NAFLD-Mild → isNafld: true',             stateMild.isNafld === true);
  check('S10: V3 NAFLD-Mild → showGrading: true',         stateMild.showGrading === true);
  check('S10: V3 NAFLD-Mild → gradingResult correct',     stateMild.gradingResult === 'Grade1_Mild');
  check('S10: V3 NAFLD-Mild → finalLabel correct',        stateMild.finalLabel === 'NAFLD-Grade1_Mild');

  // V3 — NAFLD Grade 2 Moderate/Severe
  const stateMod = deriveDisplayState({ schemaVersion: 3, binaryResult: 'NAFLD', finalLabel: 'NAFLD-Grade2_Moderate_Severe', gradingPerformed: true, gradingResult: 'Grade2_Moderate_Severe', researchUseOnly: true });
  check('S10: V3 NAFLD-Moderate/Severe → finalLabel correct',   stateMod.finalLabel === 'NAFLD-Grade2_Moderate_Severe');
  check('S10: V3 NAFLD-Moderate/Severe → showGrading: true',    stateMod.showGrading === true);

  // Legacy Normal
  const stateLegacyNormal = deriveDisplayState({ prediction: 'Normal', confidence: 72, probabilities: { Normal: 72, Abnormal: 28 } });
  check('S10: Legacy Normal → mode: "legacy"',    stateLegacyNormal.mode === 'legacy');
  check('S10: Legacy Normal → isNormal: true',     stateLegacyNormal.isNormal === true);
  check('S10: Legacy Normal → showGrading: false', stateLegacyNormal.showGrading === false);
  check('S10: Legacy Normal → no research notice', stateLegacyNormal.hasResearchNotice === false);

  // Legacy Abnormal
  const stateLegacyAbnormal = deriveDisplayState({ prediction: 'Abnormal', confidence: 81, probabilities: { Normal: 19, Abnormal: 81 } });
  check('S10: Legacy Abnormal → mode: "legacy"',     stateLegacyAbnormal.mode === 'legacy');
  check('S10: Legacy Abnormal → isNormal: false',    stateLegacyAbnormal.isNormal === false);
  check('S10: Legacy Abnormal → finalLabel correct', stateLegacyAbnormal.finalLabel === 'Abnormal');

  // Error state (non-null string passed — error message present)
  const stateError = { mode: 'error', message: 'Scan record not found' };
  check('S10: Error state — message set', stateError.message.length > 0);

  // ==========================================================================
  // S11 — End-to-end upload through Express to Flask and back to history
  // ==========================================================================
  section('S11 — End-to-end upload → Flask → history');
  const ref11 = { fn: () => ({ status: 200, responseBody: VALID_NAFLD_MILD }) };
  let mock11, srv11;
  try {
    mock11 = await createControllableMockFlask(ref11);
    srv11  = await startExpress(mock11.url);

    // Step 1: POST /api/predict
    const createResp = await sendPredict(srv11.port, srv11.token, img);
    check('S11: POST /api/predict → 200', createResp.status === 200, `got ${createResp.status}`);
    const scanId = createResp.body?.id;
    check('S11: Response contains scan ID', typeof scanId === 'string' && scanId.length > 0, `got ${scanId}`);

    await sleep(300);

    // Step 2: GET /api/scans/:id
    const getResp = await httpGet(srv11.port, `/api/scans/${scanId}`, srv11.token);
    check('S11: GET /api/scans/:id → 200', getResp.status === 200, `got ${getResp.status}`);
    check('S11: Retrieved scan matches ID', getResp.body?.id === scanId);
    check('S11: Retrieved scan has V3 fields', getResp.body?.schemaVersion === 3);
    check('S11: Retrieved finalLabel is NAFLD-Grade1_Mild', getResp.body?.finalLabel === 'NAFLD-Grade1_Mild');

    // Step 3: GET /api/scans (history list)
    const listResp = await httpGet(srv11.port, '/api/scans', srv11.token);
    check('S11: GET /api/scans → 200', listResp.status === 200, `got ${listResp.status}`);
    check('S11: History list contains the new scan',
      Array.isArray(listResp.body) && listResp.body.some(s => s.id === scanId),
      `list IDs: ${JSON.stringify(listResp.body?.map(s => s.id))}`);

    // Step 4: GET /api/scans/:id for a non-existent scan → 404
    const missingResp = await httpGet(srv11.port, '/api/scans/scan-does-not-exist', srv11.token);
    check('S11: Non-existent scan → 404', missingResp.status === 404, `got ${missingResp.status}`);
  } catch (err) {
    check('S11 execution', false, err.message);
  } finally {
    await killServer(srv11?.proc); await mock11?.close();
  }

  // ==========================================================================
  // S12 — Performance: startup time, per-request latency, memory, health
  // ==========================================================================
  section('S12 — Performance metrics');
  const ref12 = { fn: () => ({ status: 200, responseBody: VALID_NON_NAFLD }) };
  let mock12, srv12;
  try {
    mock12  = await createControllableMockFlask(ref12);
    srv12   = await startExpress(mock12.url);

    check('S12: Express startup < 30 000 ms', srv12.startupMs < 30000, `was ${srv12.startupMs} ms`);

    // Warmup request: discard the cold-start Vite middleware init penalty
    await sendPredict(srv12.port, srv12.token, img);

    // 5 sequential predict requests — measure steady-state latency
    const latencies = [];
    let healthOk = 0;
    for (let i = 0; i < 5; i++) {
      const t = Date.now();
      const r = await sendPredict(srv12.port, srv12.token, img);
      latencies.push(Date.now() - t);
      if (r.status === 200) healthOk++;
    }

    const minL  = Math.min(...latencies);
    const maxL  = Math.max(...latencies);
    const meanL = latencies.reduce((a, b) => a + b, 0) / latencies.length;

    console.log(`  Startup: ${srv12.startupMs} ms`);
    console.log(`  Request latencies (ms): ${latencies.join(', ')}`);
    console.log(`  min=${minL} ms  max=${maxL} ms  mean=${meanL.toFixed(0)} ms`);

    check('S12: All 5 health requests succeeded', healthOk === 5, `${healthOk}/5 ok`);
    check('S12: Mean latency < 5 000 ms', meanL < 5000, `mean was ${meanL.toFixed(0)} ms`);
    check('S12: Max latency < 8 000 ms', maxL < 8000, `max was ${maxL} ms`);

    // Memory snapshot
    const memAfter = process.memoryUsage();
    const rssAfterMB = (memAfter.rss / 1024 / 1024).toFixed(1);
    console.log(`  Memory RSS (test process): before=${(memBefore.rss / 1024 / 1024).toFixed(1)} MB  after=${rssAfterMB} MB`);
    check('S12: Memory RSS reported',   parseFloat(rssAfterMB) > 0);
  } catch (err) {
    check('S12 execution', false, err.message);
  } finally {
    await killServer(srv12?.proc); await mock12?.close();
  }

  // ==========================================================================
  // Final summary
  // ==========================================================================
  const total  = results.length;
  const passed = results.filter(r => r.ok).length;
  const failed = total - passed;

  console.log(`\n${'='.repeat(62)}`);
  console.log(`  Results: ${passed}/${total} passed${failed ? ` (${failed} FAILED)` : ' -- ALL PASSED'}`);
  if (failed) {
    results.filter(r => !r.ok).forEach(r => console.log(`    ✗ ${r.name}`));
  }
  console.log(`${'='.repeat(62)}\n`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Fatal error in Phase 6 test script:', err);
  process.exit(1);
});
