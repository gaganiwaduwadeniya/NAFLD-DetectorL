# NAFLD Detector — How to Run the Full System

## Prerequisites

- **Node.js** installed at `G:\Nodejs` (or on your PATH)
- **Python `.venv`** set up at `G:\NAFLD-DetectorL\.venv`
- **`firebase-applet-config.json`** populated with your Firebase credentials (local file, not in git)
- **`nafld_v3/models/`** containing the 10 `.keras` model files (pulled via Git LFS)

---

## Step 1 — Install Node dependencies (first time only)

```powershell
cd G:\NAFLD-DetectorL
$env:PATH = "G:\Nodejs;$env:PATH"
cmd /c "npm install"
```

---

## Step 2 — Set up Python environment (first time only)

```powershell
cd G:\NAFLD-DetectorL
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r ml_service\requirements.txt
```

---

## Step 3 — Start the Flask ML Service

Open **Terminal 1** and run:

```powershell
cd G:\NAFLD-DetectorL
.venv\Scripts\python.exe -m ml_service.app
```

Expected output:
```
INFO  nafld_model_bundle: Loading binary fold 0 ...
...
INFO  nafld_model_bundle: All 10 models loaded and verified.
 * Running on http://127.0.0.1:5001
```

> The ML service runs on **port 5001** by default.
> Model loading takes ~15–20 seconds (TensorFlow initialises all 10 folds).

---

## Step 4 — Start the Express + React Server

Open **Terminal 2** and run:

```powershell
cd G:\NAFLD-DetectorL
$env:PATH = "G:\Nodejs;$env:PATH"
$env:ML_SERVICE_URL = "http://127.0.0.1:5001"
$env:JWT_SECRET = "your-local-secret-change-this"
$env:PORT = "3001"
cmd /c "node_modules\.bin\tsx server.ts"
```

Expected output:
```
Firebase successfully initialized. Project: symmetric-arcana-7n50x
Server successfully booted on Port 3001
```

> If Firebase is not configured, it falls back to `data/db.json` automatically.

---

## Step 5 — Open the App

Navigate to **http://localhost:3001** in your browser.

Default login credentials:

| Role | Email | Password |
|---|---|---|
| Admin | `admin@system.com` | `admin123` |
| Doctor | `doctor@system.com` | `doctor123` |

---

## Full startup — copy-paste block (two separate terminals)

**Terminal 1 (Flask):**
```powershell
cd G:\NAFLD-DetectorL
.venv\Scripts\python.exe -m ml_service.app
```

**Terminal 2 (Express + React):**
```powershell
cd G:\NAFLD-DetectorL
$env:PATH = "G:\Nodejs;$env:PATH"
$env:ML_SERVICE_URL = "http://127.0.0.1:5001"
$env:JWT_SECRET = "your-local-secret-change-this"
cmd /c "node_modules\.bin\tsx server.ts"
```

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ML_SERVICE_URL` | **Yes** | *(none)* | URL of the Flask ML service. Without this, `/api/predict` returns 503. |
| `JWT_SECRET` | Recommended | `nafld-diagnostic-system-secret-key-2026` | Secret used to sign JWT tokens. Change for any shared environment. |
| `PORT` | No | `3001` | Port the Express server listens on. |
| `ML_SERVICE_HOST` | No | `127.0.0.1` | Flask service bind host (set in Flask, not Express). |
| `ML_SERVICE_PORT` | No | `5001` | Flask service port (set in Flask, not Express). |

---

## Run the automated test suite (no live Flask needed)

```powershell
cd G:\NAFLD-DetectorL
$env:PATH = "G:\Nodejs;$env:PATH"
& 'G:\Nodejs\node.exe' server_phase6_test.mjs
```

Expected: `Results: 111/111 passed -- ALL PASSED`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/predict` returns 503 | `ML_SERVICE_URL` not set | Set the env var and restart Express |
| Flask crashes on startup | Model files missing | Run `git lfs pull` to download `.keras` files |
| "Firebase not configured" in logs | `firebase-applet-config.json` is empty | Restore your Firebase credentials to the file |
| 10 000 changes in VS Code Source Control | VS Code showing untracked `.venv` files | Add `"git.untrackedChanges": "hidden"` to VS Code settings |
| `ImportError: attempted relative import` | Running `app.py` directly | Use `python -m ml_service.app` from project root |
