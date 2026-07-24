import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import multer from 'multer';
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { 
  dbCreateUser, 
  dbGetUserByUid, 
  dbGetUserByEmailWithPassword, 
  dbCreateScan, 
  dbGetScans, 
  dbGetScanById, 
  dbGetAdminStats,
  dbGetAllUsers
} from './src/lib/db.js';
import { Scan, User } from './src/types.js';
import { storage, isFirebaseConfigured } from './src/lib/firebase.js';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';

dotenv.config();

const app = express();
const PORT = parseInt(process.env.PORT || '3001', 10);
const JWT_SECRET = process.env.JWT_SECRET || 'nafld-diagnostic-system-secret-key-2026';

// ---------------------------------------------------------------------------
// ML service configuration — must be set in the environment before starting.
// Phase 3 requirement: if absent, /api/predict returns 503; no simulation.
// ---------------------------------------------------------------------------
const ML_SERVICE_URL = process.env.ML_SERVICE_URL?.replace(/\/$/, '') ?? '';
const ML_INFERENCE_TIMEOUT_MS = 15_000; // 15 seconds

if (!ML_SERVICE_URL) {
  console.warn(
    '[WARN] ML_SERVICE_URL is not set. ' +
    'POST /api/predict will return 503 until the Flask service URL is configured.'
  );
}

// Support JSON and urlencoded parsing
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Setup multer in-memory storage for handling image uploads.
// 20 MB ceiling matches the Flask inference service _MAX_IMAGE_BYTES limit.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 20 * 1024 * 1024, // 20 MB
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPG, PNG, and BMP are accepted.'));
    }
  }
});

// Middleware to authenticate JWT token
const authenticateToken = (req: Request | any, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, JWT_SECRET, (err: any, decoded: any) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }
    req.user = decoded;
    next();
  });
};

// --- API ROUTES ---

// 1. Auth Login
app.post('/api/auth/login', async (req: Request, res: Response) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }

  try {
    const user = await dbGetUserByEmailWithPassword(email);
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    // Verify password hash
    const crypto = await import('crypto');
    const incomingHash = crypto.createHash('sha256').update(password).digest('hex');
    if (incomingHash !== user.passwordHash) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    // Create JWT
    const payload = { uid: user.uid, email: user.email, name: user.name, role: user.role };
    const token = jwt.sign(payload, JWT_SECRET, { expiresIn: '7d' });

    res.json({
      user: {
        uid: user.uid,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt
      },
      token
    });
  } catch (error: any) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Server error during login' });
  }
});

// 2. Auth Register
app.post('/api/auth/register', async (req: Request, res: Response) => {
  const { email, password, name, role } = req.body;
  if (!email || !password || !name || !role) {
    return res.status(400).json({ error: 'All fields (email, password, name, role) are required' });
  }

  if (role !== 'doctor' && role !== 'admin') {
    return res.status(400).json({ error: 'Invalid role' });
  }

  try {
    const existingUser = await dbGetUserByEmailWithPassword(email);
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    const crypto = await import('crypto');
    const uid = 'u-' + crypto.randomBytes(8).toString('hex');
    const passwordHash = crypto.createHash('sha256').update(password).digest('hex');
    const createdAt = new Date().toISOString();

    const newUser = await dbCreateUser({
      uid,
      email,
      name,
      role,
      passwordHash,
      createdAt
    });

    const payload = { uid: newUser.uid, email: newUser.email, name: newUser.name, role: newUser.role };
    const token = jwt.sign(payload, JWT_SECRET, { expiresIn: '7d' });

    res.status(201).json({
      user: newUser,
      token
    });
  } catch (error: any) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Server error during registration' });
  }
});

// 3. Auth Get Current User Profile
app.get('/api/auth/me', authenticateToken, async (req: Request | any, res: Response) => {
  try {
    const user = await dbGetUserByUid(req.user.uid);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json({ user });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// ---------------------------------------------------------------------------
// V3 Flask response schema validator.
// Returns null on success, or a string describing the first violation.
// ---------------------------------------------------------------------------
function validateV3Response(data: any): string | null {
  if (!data || typeof data !== 'object') return 'Response is not a JSON object.';
  if (data.success !== true) return `success field is not true: ${JSON.stringify(data.success)}`;
  if (data.research_use_only !== true) return 'research_use_only must be true.';

  const validBinaryResults = ['Non-NAFLD', 'NAFLD'];
  if (!validBinaryResults.includes(data.binary_result)) {
    return `binary_result ${JSON.stringify(data.binary_result)} is not one of ${validBinaryResults}.`;
  }

  const validFinalLabels = ['Non-NAFLD', 'NAFLD-Grade1_Mild', 'NAFLD-Grade2_Moderate_Severe'];
  if (!validFinalLabels.includes(data.final_label)) {
    return `final_label ${JSON.stringify(data.final_label)} is not one of ${validFinalLabels}.`;
  }

  const probNafld = data.binary_prob_nafld;
  const probNonNafld = data.binary_prob_non_nafld;
  if (typeof probNafld !== 'number' || !isFinite(probNafld) || probNafld < 0 || probNafld > 1) {
    return `binary_prob_nafld is not a finite number in [0,1]: ${probNafld}`;
  }
  if (typeof probNonNafld !== 'number' || !isFinite(probNonNafld) || probNonNafld < 0 || probNonNafld > 1) {
    return `binary_prob_non_nafld is not a finite number in [0,1]: ${probNonNafld}`;
  }

  if (!data.model_version || typeof data.model_version !== 'string') {
    return 'model_version must be a non-empty string.';
  }

  // When grading was performed, validate grading fields too.
  if (data.grading_performed === true) {
    const validGradingResults = ['Grade1_Mild', 'Grade2_Moderate_Severe'];
    if (!validGradingResults.includes(data.grading_result)) {
      return `grading_result ${JSON.stringify(data.grading_result)} is invalid.`;
    }
    const probMod = data.grading_prob_moderate_severe;
    const probMild = data.grading_prob_mild;
    if (typeof probMod !== 'number' || !isFinite(probMod) || probMod < 0 || probMod > 1) {
      return `grading_prob_moderate_severe is not a finite number in [0,1]: ${probMod}`;
    }
    if (typeof probMild !== 'number' || !isFinite(probMild) || probMild < 0 || probMild > 1) {
      return `grading_prob_mild is not a finite number in [0,1]: ${probMild}`;
    }
  }

  return null; // valid
}

// ---------------------------------------------------------------------------
// Lightweight image dimension parser (JPEG/PNG/BMP) from raw Buffer.
// ---------------------------------------------------------------------------
function parseImageDimensions(inputBuf: any, mimetype: string): { width?: number; height?: number } {
  if (!inputBuf) return {};
  const buf = Buffer.isBuffer(inputBuf) ? inputBuf : Buffer.from(inputBuf);
  if (buf.length < 10) return {};
  try {
    // PNG (magic bytes 0x89 0x50 0x4E 0x47)
    if (buf.length >= 24 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) {
      const width = buf.readUInt32BE(16);
      const height = buf.readUInt32BE(20);
      if (width > 0 && height > 0) return { width, height };
    }
    // BMP (magic bytes 'B' 'M')
    if (buf.length >= 26 && buf[0] === 0x42 && buf[1] === 0x4d) {
      const width = Math.abs(buf.readInt32LE(18));
      const height = Math.abs(buf.readInt32LE(22));
      if (width > 0 && height > 0) return { width, height };
    }
    // JPEG (magic bytes 0xFF 0xD8)
    if (buf[0] === 0xff && buf[1] === 0xd8) {
      let offset = 2;
      while (offset < buf.length - 8) {
        if (buf[offset] !== 0xff) {
          offset++;
          continue;
        }
        const marker = buf[offset + 1];
        if (marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7)) {
          offset += 2;
          continue;
        }
        if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb].includes(marker)) {
          const height = buf.readUInt16BE(offset + 5);
          const width = buf.readUInt16BE(offset + 7);
          if (height > 0 && width > 0) {
            return { width, height };
          }
        }
        const blockLength = buf.readUInt16BE(offset + 2);
        if (blockLength < 2) break;
        offset += 2 + blockLength;
      }
    }
  } catch {
    // Graceful fallback on any byte parsing error
  }
  return {};
}

// ---------------------------------------------------------------------------
// 4. ML Prediction Forwarding & Save Scan metadata
//
// Phase 4 rules:
//   - Forward to Flask and validate response schema.
//   - Measure inference latency and extract image dimensions.
//   - Store full V3 fields (schemaVersion: 3, probabilities, thresholds,
//     fold statistics, finalLabel, contractSha256, etc.).
//   - Maintain legacy fields (prediction, confidence, probabilities) for
//     backwards compatibility with existing UI.
// ---------------------------------------------------------------------------
app.post('/api/predict', authenticateToken, upload.single('image'), async (req: Request | any, res: Response) => {
  const { patientName, patientAge, patientGender } = req.body;

  if (!req.file) {
    return res.status(400).json({ error: 'An ultrasound image is required.' });
  }
  if (!patientName || !patientAge || !patientGender) {
    return res.status(400).json({ error: 'Patient demographic information (name, age, gender) is required.' });
  }

  // --- Fail closed: ML service URL must be configured. ---
  if (!ML_SERVICE_URL) {
    return res.status(503).json({
      error: 'The ML inference service is not configured. Set the ML_SERVICE_URL environment variable and restart the server.'
    });
  }

  const doctorId = req.user.uid;
  const doctorName = req.user.name;

  // --- Forward to Flask with timeout & latency measurement ---
  let flaskData: any;
  let inferenceLatencyMs = 0;
  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), ML_INFERENCE_TIMEOUT_MS);
  const fetchStartTime = Date.now();

  try {
    console.log(`[predict] Forwarding image (${req.file.size} bytes, ${req.file.mimetype}) to Flask at ${ML_SERVICE_URL}`);

    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
    formData.append('image', blob, req.file.originalname || 'upload');

    const flaskResponse = await fetch(`${ML_SERVICE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutHandle);
    inferenceLatencyMs = Date.now() - fetchStartTime;

    // Flask 4xx = bad image from client — forward the Flask error message.
    if (flaskResponse.status >= 400 && flaskResponse.status < 500) {
      let flaskError = 'The uploaded image was rejected by the inference service.';
      try {
        const errBody = await flaskResponse.json();
        if (errBody?.error) flaskError = errBody.error;
      } catch { /* ignore parse failure */ }
      console.warn(`[predict] Flask returned ${flaskResponse.status}: ${flaskError}`);
      return res.status(400).json({ error: flaskError });
    }

    // Flask 5xx = model/server failure.
    if (flaskResponse.status >= 500) {
      console.error(`[predict] Flask returned ${flaskResponse.status}. Returning 502.`);
      return res.status(502).json({ error: 'The inference service encountered an internal error. Please retry.' });
    }

    // Parse 200 response.
    try {
      flaskData = await flaskResponse.json();
    } catch (parseErr) {
      console.error('[predict] Flask 200 response was not valid JSON.');
      return res.status(502).json({ error: 'The inference service returned an unreadable response.' });
    }

  } catch (fetchErr: any) {
    clearTimeout(timeoutHandle);
    if (fetchErr.name === 'AbortError') {
      console.error(`[predict] Flask request timed out after ${ML_INFERENCE_TIMEOUT_MS}ms.`);
      return res.status(503).json({ error: 'The inference service did not respond in time. Please retry.' });
    }
    console.error('[predict] Could not reach Flask service:', fetchErr.message);
    return res.status(503).json({ error: 'The inference service is unavailable. Please retry.' });
  }

  // --- Validate the V3 response schema before touching the database. ---
  const schemaError = validateV3Response(flaskData);
  if (schemaError) {
    console.error('[predict] Flask response failed schema validation:', schemaError, JSON.stringify(flaskData));
    return res.status(502).json({ error: 'The inference service returned an unexpected response format.' });
  }

  // --- Derived legacy fields for backwards compatibility ---
  const isNafld = flaskData.binary_result === 'NAFLD';
  const prediction: 'Normal' | 'Abnormal' = isNafld ? 'Abnormal' : 'Normal';
  const probNafld: number = flaskData.binary_prob_nafld;
  const probNonNafld: number = flaskData.binary_prob_non_nafld;
  const confidence = parseFloat(((isNafld ? probNafld : probNonNafld) * 100).toFixed(1));
  const probabilities = {
    Normal: parseFloat((probNonNafld * 100).toFixed(1)),
    Abnormal: parseFloat((probNafld * 100).toFixed(1)),
  };

  // --- Image dimension parsing & Storage ---
  const imageDims = parseImageDimensions(req.file.buffer, req.file.mimetype);

  let imageUrl = '';
  try {
    if (isFirebaseConfigured() && storage) {
      console.log('[predict] Uploading scan image to Firebase Storage...');
      const uniqueFilename = `scans/${Date.now()}-${req.file.originalname}`;
      const storageRef = ref(storage, uniqueFilename);
      const metadata = { contentType: req.file.mimetype };
      await uploadBytes(storageRef, req.file.buffer, metadata);
      imageUrl = await getDownloadURL(storageRef);
      console.log('[predict] Firebase Storage upload succeeded:', imageUrl);
    } else {
      console.log('[predict] Firebase not configured. Storing image as inline base64.');
      const base64Data = req.file.buffer.toString('base64');
      imageUrl = `data:${req.file.mimetype};base64,${base64Data}`;
    }
  } catch (storageErr: any) {
    console.error('[predict] Image storage failed, falling back to base64:', storageErr.message);
    const base64Data = req.file.buffer.toString('base64');
    imageUrl = `data:${req.file.mimetype};base64,${base64Data}`;
  }

  const crypto = await import('crypto');
  const scanId = 'scan-' + crypto.randomBytes(8).toString('hex');

  // --- Phase 4: Construct complete versioned Scan record ---
  const newScan: Scan = {
    id: scanId,
    patientName,
    patientAge: parseInt(patientAge, 10),
    patientGender: patientGender as 'Male' | 'Female' | 'Other',
    doctorId,
    doctorName,
    imageUrl,

    // Legacy compatibility fields
    prediction,
    confidence,
    probabilities,
    timestamp: new Date().toISOString(),

    // Phase 4 V3 Schema Versioned Metadata
    schemaVersion: 3,
    modelVersion: flaskData.model_version,
    contractSha256: flaskData.contract_sha256,
    inputMode: flaskData.input_mode || 'single_frame',
    researchUseOnly: flaskData.research_use_only ?? true,

    binaryResult: flaskData.binary_result,
    binaryProbNafld: flaskData.binary_prob_nafld,
    binaryProbNonNafld: flaskData.binary_prob_non_nafld,
    binaryThreshold: flaskData.binary_threshold,
    binaryFoldProbs: flaskData.binary_fold_probs || [],
    binaryFoldStd: flaskData.binary_fold_std,

    gradingPerformed: flaskData.grading_performed === true,
    ...(flaskData.grading_performed === true ? {
      gradingResult: flaskData.grading_result,
      gradingProbModerateSevere: flaskData.grading_prob_moderate_severe,
      gradingProbMild: flaskData.grading_prob_mild,
      gradingThreshold: flaskData.grading_threshold,
      gradingFoldProbs: flaskData.grading_fold_probs || [],
      gradingFoldStd: flaskData.grading_fold_std,
    } : {}),

    finalLabel: flaskData.final_label,
    ...(imageDims.width ? { imageWidth: imageDims.width } : {}),
    ...(imageDims.height ? { imageHeight: imageDims.height } : {}),
    inferenceLatencyMs,
  };

  try {
    await dbCreateScan(newScan);
  } catch (dbErr: any) {
    console.error('[predict] DB write failed after successful inference:', dbErr.message);
    return res.status(500).json({ error: 'Inference succeeded but the scan record could not be saved.' });
  }

  console.log(
    `[predict] Scan ${scanId} saved (schemaVersion=3, final_label=${flaskData.final_label}, ` +
    `latency=${inferenceLatencyMs}ms, size=${imageDims.width}x${imageDims.height})`
  );
  res.json(newScan);
});

// 5. Get List of Scans (doctor-specific or all for admin)
app.get('/api/scans', authenticateToken, async (req: Request | any, res: Response) => {
  try {
    const scans = await dbGetScans(req.user.uid, req.user.role);
    res.json(scans);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Error fetching scan directory' });
  }
});

// 6. Get Single Scan Details
app.get('/api/scans/:id', authenticateToken, async (req: Request, res: Response) => {
  const { id } = req.params;
  try {
    const scan = await dbGetScanById(id);
    if (!scan) {
      return res.status(404).json({ error: 'Diagnostic scan not found' });
    }
    res.json(scan);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Error fetching scan details' });
  }
});

// 7. Get Admin Users Directory (admin only)
app.get('/api/admin/users', authenticateToken, async (req: Request | any, res: Response) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Administrative privileges required' });
  }

  try {
    const users = await dbGetAllUsers();
    res.json(users);
  } catch (error: any) {
    res.status(500).json({ error: 'Failed to fetch user directory' });
  }
});

// 8. Get Dashboard Statistics (admin only)
app.get('/api/dashboard/stats', authenticateToken, async (req: Request | any, res: Response) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Administrative privileges required' });
  }

  try {
    const stats = await dbGetAdminStats();
    res.json(stats);
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Error fetching dashboard stats' });
  }
});


// --- VITE MIDDLEWARE SETUP / STATIC FILE SERVING ---

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    
    // Inject Vite development middlewares
    app.use(vite.middlewares);
  } else {
    // In production, serve pre-built assets
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server successfully booted on Port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal dev server bootstrap failure:', err);
});
